use crate::config::ServerConfig;
use crate::environment::Environment;
use crate::io::Outputs;
use crate::model::{ModelPlan, PreparedRequest, session_input, tensor_from_ort};
use crate::session::SessionBuilder;
use crate::tensor::Tensor;
use anyhow::{Context, Result};
use ort::session::{Session, SessionInputs};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::thread;

/// Task that runs ONNX inference in a dedicated thread.
///
/// Each worker owns a single ONNX session and processes batches
/// sequentially. Multiple workers can run in parallel for better
/// hardware utilization.
pub struct WorkerTask;

impl WorkerTask {
    /// Spawn a worker thread with its own ONNX session.
    ///
    /// The worker initializes its session and reports the outcome on `ready_tx`.
    /// If initialization fails, the worker exits. Otherwise it processes
    /// batches from its dedicated channel until the channel closes. A panic
    /// while processing a batch fails that batch only; the worker keeps running.
    ///
    /// # Arguments
    ///
    /// * `worker_rx` - Dedicated receiver for this worker's messages.
    /// * `ready_tx` - Receives `Ok(())` once the session is ready, or the
    ///   initialization error.
    /// * `environment` - ONNX Runtime environment to create the session in.
    /// * `model_path` - Path to the ONNX model file.
    /// * `config` - Server configuration.
    /// * `worker_id` - Unique identifier for logging.
    ///
    /// # Returns
    ///
    /// A `JoinHandle` for the spawned thread.
    pub fn spawn(
        worker_rx: kanal::AsyncReceiver<WorkerMessage>,
        ready_tx: tokio::sync::oneshot::Sender<Result<()>>,
        environment: Environment,
        model_path: Arc<String>,
        config: Arc<ServerConfig>,
        worker_id: usize,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            tracing::info!("Worker {}: Initializing ONNX session...", worker_id);

            let mut session = match SessionBuilder::build(&environment, &model_path, &config) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Worker {}: Failed to initialize session: {e:#}", worker_id);
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };

            tracing::info!("Worker {}: Session initialized successfully.", worker_id);
            let _ = ready_tx.send(Ok(()));

            let sync_rx = worker_rx.clone().to_sync();
            loop {
                let msg = match sync_rx.recv() {
                    Ok(m) => m,
                    Err(_) => {
                        tracing::info!("Worker {}: Channel closed, shutting down.", worker_id);
                        break;
                    }
                };

                // A panic drops the message's result channel, so the batcher
                // fails just this batch; the session stays usable.
                let processed = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    Self::process_message(worker_id, &mut session, msg)
                }));
                if processed.is_err() {
                    tracing::error!("Worker {}: Panicked while processing a batch.", worker_id);
                }
            }
        })
    }

    /// Process a single batch message.
    ///
    /// Assembles the batched input tensors, runs inference, splits the outputs
    /// per request and sends them back to the batcher for postprocessing.
    fn process_message(worker_id: usize, session: &mut Session, msg: WorkerMessage) {
        let WorkerMessage {
            requests,
            plan,
            result_tx,
        } = msg;

        tracing::info!("Worker {}: Processing inference...", worker_id);
        let num_requests = requests.len();

        let result = Self::run_batch(session, &plan, requests);
        match &result {
            Ok(_) => tracing::info!(
                "Worker {}: Successfully processed batch of {} items",
                worker_id,
                num_requests
            ),
            Err(e) => tracing::error!("Worker {}: Batch failed: {e:#}", worker_id),
        }
        let _ = result_tx.send(result);
    }

    fn run_batch(
        session: &mut Session,
        plan: &ModelPlan,
        requests: Vec<PreparedRequest>,
    ) -> Result<Vec<Outputs>> {
        plan.check_outputs()?;
        let batch = plan.assemble(requests)?;

        let inputs = batch
            .tensors
            .iter()
            .map(|(name, tensor)| {
                Ok((
                    std::borrow::Cow::Borrowed(&**name),
                    session_input(tensor)
                        .with_context(|| format!("Failed to create input tensor '{name}'"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let outputs = session
            .run(SessionInputs::<0>::ValueMap(inputs))
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;

        let mut raw: HashMap<Arc<str>, Tensor> = HashMap::new();
        for name in plan.needed_outputs() {
            if let Some(value) = outputs.get(&**name) {
                raw.insert(name.clone(), tensor_from_ort(name, value)?);
            }
        }
        drop(outputs);

        plan.split_outputs(&batch, &raw)
    }
}

/// Message sent from the batcher to a worker.
///
/// Contains the validated requests of one batch and the channel for the
/// per-request outputs.
pub struct WorkerMessage {
    /// The batch's requests, validated against the model.
    pub(crate) requests: Vec<PreparedRequest>,
    /// How to batch the inputs and split the outputs.
    pub(crate) plan: Arc<ModelPlan>,
    /// Channel to send the per-request outputs back to the batcher.
    pub(crate) result_tx: tokio::sync::oneshot::Sender<Result<Vec<Outputs>>>,
}
