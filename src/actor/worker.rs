use crate::config::ServerConfig;
use crate::session::SessionBuilder;
use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ort::session::{Session, SessionInputValue, SessionInputs};
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
    /// The worker will initialize the session, then process batches
    /// from its dedicated channel until shutdown. Failed sessions are retried
    /// with exponential backoff.
    ///
    /// # Arguments
    ///
    /// * `worker_rx` - Dedicated receiver for this worker's messages.
    /// * `model_path` - Path to the ONNX model file.
    /// * `config` - Server configuration.
    /// * `worker_id` - Unique identifier for logging.
    ///
    /// # Returns
    ///
    /// A `JoinHandle` for the spawned thread.
    pub fn spawn(
        mut worker_rx: tokio::sync::mpsc::UnboundedReceiver<WorkerMessage>,
        model_path: Arc<String>,
        config: Arc<ServerConfig>,
        worker_id: usize,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            tracing::info!("Worker {}: Initializing ONNX session...", worker_id);

            // Initialize session with retry on failure
            let mut session = loop {
                match SessionBuilder::build(&model_path, &config) {
                    Ok(s) => break s,
                    Err(e) => {
                        tracing::error!(
                            "Worker {}: Failed to initialize session: {e:#}. Retrying in 5s...",
                            worker_id
                        );
                        thread::sleep(std::time::Duration::from_secs(5));
                    }
                }
            };

            tracing::info!("Worker {}: Session initialized successfully.", worker_id);

            // Process batches until channel closes
            loop {
                let msg = match worker_rx.blocking_recv() {
                    Some(m) => m,
                    None => {
                        tracing::info!("Worker {}: Channel closed, shutting down.", worker_id);
                        break;
                    }
                };

                Self::process_message(worker_id, &mut session, msg);
            }
        })
    }

    /// Process a single batch message.
    ///
    /// Runs inference on the batched tensor and sends sliced outputs
    /// back to the batcher for postprocessing.
    fn process_message(worker_id: usize, session: &mut Session, msg: WorkerMessage) {
        let WorkerMessage {
            batched_tensor,
            input_name,
            output_name,
            result_tx,
        } = msg;

        tracing::info!("Worker {}: Processing inference...", worker_id);

        // Create input tensor
        let input_value = match ort::value::Value::from_array(batched_tensor.clone()) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Worker {}: Failed to create input tensor: {e:#}", worker_id);
                let _ = result_tx.send(Err(anyhow::Error::msg(e.to_string())));
                return;
            }
        };

        // Create input map
        let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
            std::borrow::Cow::Borrowed(input_name.as_str()),
            SessionInputValue::Owned(input_value.into()),
        )]);

        // Run inference
        let outputs = match session.run(inputs) {
            Ok(o) => o,
            Err(e) => {
                tracing::error!("Worker {}: Inference failed: {e:#}", worker_id);
                let _ = result_tx.send(Err(anyhow::Error::msg(e.to_string())));
                return;
            }
        };

        // Extract output tensor
        let output_tensor = match outputs.get(&output_name) {
            Some(v) => v,
            None => {
                tracing::error!(
                    "Worker {}: Output tensor '{}' not found",
                    worker_id,
                    output_name
                );
                let _ = result_tx.send(Err(anyhow::Error::msg(format!(
                    "Output tensor '{}' not found",
                    output_name
                ))));
                return;
            }
        };

        // Extract tensor data
        let (shape, data) = match output_tensor.try_extract_tensor::<f32>() {
            Ok((s, d)) => (s, d),
            Err(e) => {
                tracing::error!("Worker {}: Failed to extract tensor: {e:#}", worker_id);
                let _ = result_tx.send(Err(anyhow::Error::msg(e.to_string())));
                return;
            }
        };

        // Slice tensor by batch dimension
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let batch_size = shape_usize[0];

        let batch_view = match ndarray::ArrayView::from_shape(IxDyn(&shape_usize), data) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("Worker {}: Failed to create array view: {e:#}", worker_id);
                let _ = result_tx.send(Err(anyhow::Error::msg(e.to_string())));
                return;
            }
        };

        // Create sliced outputs for each batch item
        let mut sliced_outputs = Vec::with_capacity(batch_size);
        for idx in 0..batch_size {
            let single_output = batch_view.index_axis(ndarray::Axis(0), idx);
            sliced_outputs.push(
                ArrayD::from_shape_vec(
                    IxDyn(&shape_usize[1..]),
                    single_output.iter().cloned().collect(),
                )
                .unwrap(),
            );
        }

        tracing::info!(
            "Worker {}: Successfully processed batch of {} items",
            worker_id,
            batch_size
        );

        let _ = result_tx.send(Ok(sliced_outputs));
    }
}

/// Message sent from the batcher to a worker.
///
/// Contains the batched input tensor and metadata for inference.
pub struct WorkerMessage {
    /// Batched input tensor with shape `[batch_size, ...]`.
    pub batched_tensor: ArrayD<f32>,
    /// Name of the ONNX input tensor.
    pub input_name: String,
    /// Name of the ONNX output tensor.
    pub output_name: String,
    /// Channel to send sliced outputs back to batcher.
    pub result_tx: tokio::sync::oneshot::Sender<Result<Vec<ArrayD<f32>>>>,
}
