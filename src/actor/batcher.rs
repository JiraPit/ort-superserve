use crate::actor::command::Command;
use crate::actor::worker::WorkerMessage;
use crate::config::ServerConfig;
use crate::traits::{Input, Output};
use anyhow::Result;
use ndarray::ArrayD;
use std::sync::Arc;
use std::sync::mpsc::Sender as StdSender;
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
    /// * `worker_tx` - Channel for sending batched tensors to workers.
    /// * `config` - Server configuration.
    /// * `input_name` - Name of the ONNX input tensor.
    /// * `output_name` - Name of the ONNX output tensor.
    ///
    /// # Returns
    ///
    /// A `JoinHandle` for the spawned task.
    pub fn spawn<I: Input, O: Output>(
        mut command_rx: tokio::sync::mpsc::UnboundedReceiver<Command<I, O>>,
        worker_tx: StdSender<WorkerMessage>,
        config: Arc<ServerConfig>,
        input_name: String,
        output_name: String,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut batch_buffer: Vec<Command<I, O>> = Vec::with_capacity(config.max_batch_size);

            loop {
                // Wait for first command
                let first_cmd = match command_rx.recv().await {
                    Some(cmd) => cmd,
                    None => break,
                };
                batch_buffer.push(first_cmd);

                // Collect more commands until batch is full or timeout
                let deadline = tokio::time::Instant::now() + config.max_wait_time;
                while batch_buffer.len() < config.max_batch_size {
                    if tokio::time::Instant::now() >= deadline {
                        break;
                    }

                    match command_rx.try_recv() {
                        Ok(cmd) => batch_buffer.push(cmd),
                        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                            tokio::task::yield_now().await;
                        }
                        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
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

                // Run preprocessing in parallel
                let mut preprocess_tasks = JoinSet::new();
                let mut responders: Vec<tokio::sync::oneshot::Sender<Result<O>>> =
                    Vec::with_capacity(batch_len);

                for cmd in batch_buffer.drain(..) {
                    responders.push(cmd.responder);
                    preprocess_tasks.spawn(async move { cmd.input.preprocess().await });
                }

                let mut preprocessed_items = Vec::with_capacity(batch_len);

                while let Some(result) = preprocess_tasks.join_next().await {
                    match result {
                        Ok(Ok(item)) => preprocessed_items.push(item),
                        Ok(Err(e)) => {
                            tracing::error!("Batcher: Preprocessing failed: {e:#}");
                        }
                        Err(e) => {
                            tracing::error!("Batcher: Preprocessing task panicked: {e:#}");
                        }
                    }
                }

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

                // Send to worker pool
                let (result_tx, result_rx) = std::sync::mpsc::channel::<Result<Vec<ArrayD<f32>>>>();

                let worker_msg = WorkerMessage {
                    batched_tensor,
                    input_name: input_name.clone(),
                    output_name: output_name.clone(),
                    result_tx,
                };

                if let Err(e) = worker_tx.send(worker_msg) {
                    tracing::error!("Batcher: Failed to send to worker: {e:#}");
                    for responder in responders {
                        let _ = responder.send(Err(anyhow::Error::msg("Worker unavailable")));
                    }
                    continue;
                }

                // Receive sliced outputs from worker
                let sliced_outputs = match result_rx.recv() {
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
