use crate::actor::command::Command;
use crate::actor::worker::WorkerMessage;
use crate::config::ServerConfig;
use crate::traits::{Input, Output};
use anyhow::Result;
use ndarray::ArrayD;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task::JoinSet;

/// Task that collects requests, batches them, and dispatches to workers.
///
/// The batcher runs as a single Tokio task and performs:
/// 1. Collects incoming requests until batch is full or timeout
/// 2. Runs preprocessing in parallel using `JoinSet`
/// 3. Batches preprocessed inputs into a single tensor
/// 4. Sends batched tensor to worker pool
/// 5. Receives outputs and runs postprocessing in parallel
/// 6. Sends results back to callers via oneshot channels
pub struct BatcherTask;

impl BatcherTask {
    /// Spawn the batcher task.
    ///
    /// # Arguments
    ///
    /// * `command_rx` - Channel for receiving commands from the server.
    /// * `worker_txs` - Per-worker channels for lock-free dispatch.
    /// * `config` - Server configuration.
    /// * `input_name` - Name of the ONNX input tensor.
    /// * `output_name` - Name of the ONNX output tensor.
    ///
    /// # Returns
    ///
    /// A `JoinHandle` for the spawned task.
    pub fn spawn<I: Input, O: Output>(
        command_rx: kanal::AsyncReceiver<Command<I, O>>,
        worker_txs: Vec<kanal::AsyncSender<WorkerMessage>>,
        config: Arc<ServerConfig>,
        input_name: Arc<str>,
        output_name: Arc<str>,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut batch_buffer: Vec<Command<I, O>> = Vec::with_capacity(config.max_batch_size);
            let num_workers = worker_txs.len();
            let mut dispatch_index: usize = 0;

            loop {
                let first_cmd = match command_rx.recv().await {
                    Ok(cmd) => cmd,
                    Err(_) => break,
                };
                batch_buffer.push(first_cmd);

                let sleep = tokio::time::sleep(config.max_wait_time);
                tokio::pin!(sleep);

                while batch_buffer.len() < config.max_batch_size {
                    tokio::select! {
                        cmd = command_rx.recv() => match cmd {
                            Ok(c) => batch_buffer.push(c),
                            Err(_) => break,
                        },
                        () = &mut sleep => break,
                    }
                }

                if batch_buffer.is_empty() {
                    continue;
                }

                // Check minimum batch size
                if batch_buffer.len() < config.min_batch_size {
                    tracing::warn!(
                        "Batcher: Only {} items collected, less than min_batch_size {}.",
                        batch_buffer.len(),
                        config.min_batch_size
                    );
                    for cmd in batch_buffer.drain(..) {
                        let _ = cmd
                            .responder
                            .send(Err(anyhow::Error::msg("Insufficient batch size")));
                    }
                    continue;
                }

                let batch_len = batch_buffer.len();
                tracing::info!("Batcher: Processing batch of {} items...", batch_len);

                // Run preprocessing in parallel with index tracking to preserve order
                let mut preprocess_tasks = JoinSet::new();
                let mut responders: Vec<tokio::sync::oneshot::Sender<Result<O>>> =
                    Vec::with_capacity(batch_len);

                for (idx, cmd) in batch_buffer.drain(..).enumerate() {
                    responders.push(cmd.responder);
                    preprocess_tasks.spawn(async move { (idx, cmd.input.preprocess().await) });
                }

                // Collect results with index tracking
                let mut results: HashMap<usize, I::Preprocessed> = HashMap::new();
                let mut failed_indices: Vec<usize> = Vec::new();

                while let Some(result) = preprocess_tasks.join_next().await {
                    match result {
                        Ok((idx, Ok(item))) => {
                            results.insert(idx, item);
                        }
                        Ok((idx, Err(e))) => {
                            tracing::error!("Batcher: Preprocessing failed for item {idx}: {e:#}");
                            failed_indices.push(idx);
                        }
                        Err(e) => {
                            tracing::error!("Batcher: Preprocessing task panicked: {e:#}");
                        }
                    }
                }

                // Track panicked tasks: spawned count minus completed count
                let panicked_count = batch_len - results.len() - failed_indices.len();
                if panicked_count > 0 {
                    tracing::error!("Batcher: {panicked_count} preprocessing tasks panicked");
                }

                // Track successful indices for responder filtering
                let success_indices: Vec<usize> = results.keys().cloned().collect();

                // Build preprocessed items in original order
                let preprocessed_items: Vec<_> = (0..batch_len)
                    .filter_map(|idx| results.remove(&idx))
                    .collect();

                // Filter responders: send errors to failed/panicked ones, keep valid ones
                let responders: Vec<_> = responders
                    .into_iter()
                    .enumerate()
                    .filter_map(|(idx, resp)| {
                        if success_indices.contains(&idx) {
                            Some(resp)
                        } else {
                            let _ = resp.send(Err(anyhow::Error::msg("Preprocessing failed")));
                            None
                        }
                    })
                    .collect();

                if preprocessed_items.is_empty() {
                    tracing::error!("Batcher: All preprocessing tasks failed.");
                    continue;
                }

                // Batch preprocessed inputs
                let batched_tensor = match I::batch(preprocessed_items) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::error!("Batcher: Batching failed: {e:#}");
                        for responder in responders {
                            let _ = responder
                                .send(Err(anyhow::Error::msg(format!("Batching failed: {e:#}"))));
                        }
                        continue;
                    }
                };

                // Send to worker pool (round-robin dispatch)
                let (result_tx, result_rx) =
                    tokio::sync::oneshot::channel::<Result<Vec<ArrayD<f32>>>>();

                let worker_msg = WorkerMessage {
                    batched_tensor,
                    input_name: input_name.clone(),
                    output_name: output_name.clone(),
                    result_tx,
                };

                let worker_idx = dispatch_index % num_workers;
                dispatch_index = dispatch_index.wrapping_add(1);

                if let Err(e) = worker_txs[worker_idx].send(worker_msg).await {
                    tracing::error!("Batcher: Failed to send to worker {worker_idx}: {e:#}");
                    for responder in responders {
                        let _ = responder.send(Err(anyhow::Error::msg("Worker unavailable")));
                    }
                    continue;
                }

                // Receive sliced outputs from worker
                let sliced_outputs = match result_rx.await {
                    Ok(Ok(outputs)) => outputs,
                    Ok(Err(e)) => {
                        tracing::error!("Batcher: Worker inference failed: {e:#}");
                        let err = e.to_string();
                        for responder in responders {
                            let _ = responder.send(Err(anyhow::Error::msg(err.clone())));
                        }
                        continue;
                    }
                    Err(_) => {
                        tracing::error!("Batcher: Worker channel closed unexpectedly");
                        for responder in responders {
                            let _ =
                                responder.send(Err(anyhow::Error::msg("Worker channel closed")));
                        }
                        continue;
                    }
                };

                // Verify output count matches responder count
                if sliced_outputs.len() != responders.len() {
                    tracing::error!(
                        "Batcher: Output count mismatch: {} outputs for {} responders",
                        sliced_outputs.len(),
                        responders.len()
                    );
                    for responder in responders {
                        let _ = responder.send(Err(anyhow::Error::msg("Output count mismatch")));
                    }
                    continue;
                }

                // Run postprocessing in parallel
                let mut postprocess_tasks = JoinSet::new();
                for (raw_output, responder) in
                    sliced_outputs.into_iter().zip(responders.into_iter())
                {
                    postprocess_tasks.spawn(async move {
                        let result = O::postprocess(raw_output.view()).await;
                        let _ = responder.send(result);
                    });
                }

                while let Some(result) = postprocess_tasks.join_next().await {
                    if let Err(e) = result {
                        tracing::error!("Batcher: Postprocess task panicked: {e:#}");
                    }
                }
            }

            tracing::info!("Batcher: Shutting down.");
        })
    }
}
