//! Benchmark server using Actix actors with manual batching implementation.
//!
//! This implementation demonstrates the complexity of implementing dynamic
//! batching manually using Actix actors. Two actors are used:
//! - BatcherActor: Collects requests and dispatches batches
//! - WorkerActor: Runs inference on batched inputs
//!
//! Requires significantly more code than ort-superserve for equivalent functionality.

use actix::{Actor, Addr, AsyncContext, Handler, Message, SyncArbiter, SyncContext};
use anyhow::Result;
use axum::{Json, Router, extract::State, routing::post};
use ndarray::ArrayD;
use ort::{
    session::{Session, SessionInputValue, SessionInputs, builder::GraphOptimizationLevel},
    value::Value,
};
use shared::{ImageInput, ImageOutput, apply_imagenet_normalization};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::oneshot;
use tokio::time::Duration;
use tower_http::cors::CorsLayer;

/// Maximum number of requests to batch together.
const MAX_BATCH_SIZE: usize = 32;
/// Maximum time to wait for a full batch (milliseconds).
const MAX_WAIT_MS: u64 = 10;

/// Worker actor that owns the ONNX session and processes batched inference.
struct WorkerActor {
    session: Session,
}

impl Actor for WorkerActor {
    type Context = SyncContext<Self>;
}

impl WorkerActor {
    /// Creates a new worker actor with the given model.
    fn new(model_path: &PathBuf) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(Self { session })
    }
}

/// Message containing a batch of preprocessed inputs for inference.
struct BatchInferMessage {
    /// Batched input tensors.
    batch: Vec<ArrayD<f32>>,
    /// Response channels for each batch item.
    result_txs: Vec<oneshot::Sender<Vec<f64>>>,
}

impl Message for BatchInferMessage {
    type Result = ();
}

impl Handler<BatchInferMessage> for WorkerActor {
    type Result = ();

    /// Runs inference on the batched inputs and sends results to each caller.
    fn handle(&mut self, msg: BatchInferMessage, _ctx: &mut Self::Context) {
        let batch_size = msg.batch.len();
        if batch_size == 0 {
            return;
        }

        let views: Vec<_> = msg.batch.iter().map(|a| a.view()).collect();
        let batched = match ndarray::stack(ndarray::Axis(0), &views) {
            Ok(b) => b.into_dyn(),
            Err(_) => {
                for tx in msg.result_txs {
                    let _ = tx.send(Vec::new());
                }
                return;
            }
        };

        let input_value = match Value::from_array(batched) {
            Ok(v) => v,
            Err(_) => {
                for tx in msg.result_txs {
                    let _ = tx.send(Vec::new());
                }
                return;
            }
        };

        let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
            std::borrow::Cow::Borrowed("input"),
            SessionInputValue::Owned(input_value.into()),
        )]);

        let outputs = match self.session.run(inputs) {
            Ok(o) => o,
            Err(_) => {
                for tx in msg.result_txs {
                    let _ = tx.send(Vec::new());
                }
                return;
            }
        };

        let output_tensor = match outputs.get("output") {
            Some(t) => t,
            None => {
                for tx in msg.result_txs {
                    let _ = tx.send(Vec::new());
                }
                return;
            }
        };

        let (shape, data) = match output_tensor.try_extract_tensor::<f32>() {
            Ok((s, d)) => (s, d),
            Err(_) => {
                for tx in msg.result_txs {
                    let _ = tx.send(Vec::new());
                }
                return;
            }
        };

        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let num_classes = if shape_usize.len() > 1 {
            shape_usize[1]
        } else {
            shape_usize[0]
        };

        for (i, tx) in msg.result_txs.into_iter().enumerate() {
            let start = i * num_classes;
            let end = start + num_classes;
            let slice = &data[start..end];
            let result: Vec<f64> = slice.iter().map(|&x| x as f64).collect();
            let _ = tx.send(result);
        }
    }
}

/// Batcher actor that collects requests and dispatches batches to the worker.
struct BatcherActor {
    /// Address of the worker actor.
    worker: Addr<WorkerActor>,
    /// Accumulated batch inputs.
    batch: Vec<ArrayD<f32>>,
    /// Response channels for pending requests.
    result_txs: Vec<oneshot::Sender<Vec<f64>>>,
    /// Maximum batch size before forced dispatch.
    max_batch_size: usize,
    /// Maximum time to wait for more requests.
    max_wait: Duration,
}

impl Actor for BatcherActor {
    type Context = actix::Context<Self>;
}

/// Message containing a single inference request.
struct InferMessage {
    input_array: ArrayD<f32>,
}

impl Message for InferMessage {
    type Result = Result<Vec<f64>, anyhow::Error>;
}

impl Handler<InferMessage> for BatcherActor {
    type Result = actix::Response<Result<Vec<f64>, anyhow::Error>>;

    /// Adds the request to the current batch and dispatches if full or timeout.
    fn handle(&mut self, msg: InferMessage, ctx: &mut Self::Context) -> Self::Result {
        let (tx, rx) = oneshot::channel();
        self.batch.push(msg.input_array);
        self.result_txs.push(tx);

        if self.batch.len() >= self.max_batch_size {
            self.dispatch_batch();
        } else {
            ctx.run_later(self.max_wait, |act, _ctx| {
                act.dispatch_batch();
            });
        }

        actix::Response::fut(
            async move { rx.await.map_err(|_| anyhow::Error::msg("Channel closed")) },
        )
    }
}

impl BatcherActor {
    /// Dispatches the current batch to the worker actor.
    fn dispatch_batch(&mut self) {
        if self.batch.is_empty() {
            return;
        }

        let batch = std::mem::take(&mut self.batch);
        let txs = std::mem::take(&mut self.result_txs);

        self.worker.do_send(BatchInferMessage {
            batch,
            result_txs: txs,
        });
    }
}

/// Application state containing the batcher actor address.
struct AppState {
    batcher: Addr<BatcherActor>,
}

#[actix::main]
async fn main() -> Result<()> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/resnet50-v1-12-int8.onnx");

    let model_path_clone = model_path.clone();
    let worker = SyncArbiter::start(1, move || {
        WorkerActor::new(&model_path_clone).expect("Failed to create worker")
    });

    let batcher = BatcherActor::create(|_ctx| BatcherActor {
        worker: worker.clone(),
        batch: Vec::new(),
        result_txs: Vec::new(),
        max_batch_size: MAX_BATCH_SIZE,
        max_wait: Duration::from_millis(MAX_WAIT_MS),
    });

    let state = Arc::new(AppState { batcher });

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3002").await?;
    println!("actix-with-batching-server listening on http://0.0.0.0:3002");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Handles inference requests by preprocessing and sending to the batcher.
async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<ImageInput>,
) -> Json<ImageOutput> {
    let start = Instant::now();

    let input_array = tokio::task::spawn_blocking(move || {
        let arr = input.to_input_array().expect("Failed to process image");
        apply_imagenet_normalization(arr)
    })
    .await
    .expect("Preprocessing task failed");

    let logits: Vec<f64> = state
        .batcher
        .send(InferMessage { input_array })
        .await
        .expect("Batcher send failed")
        .expect("Inference failed");

    if logits.is_empty() {
        panic!("Inference returned empty logits");
    }

    let logits_f32: Vec<f32> = logits.iter().map(|&x| x as f32).collect();
    let output = ImageOutput::from_logits(&logits_f32);

    let _elapsed = start.elapsed();

    Json(output)
}
