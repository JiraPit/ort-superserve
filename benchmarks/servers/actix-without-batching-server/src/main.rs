//! Benchmark server using Actix actors without batching.
//!
//! This implementation uses actix SyncArbiter to run inference on a dedicated
//! thread, avoiding async runtime blocking. Each request is processed
//! individually without batching, making it simpler but less efficient under load.

use actix::{Actor, Addr, Handler, Message, SyncArbiter, SyncContext};
use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue, SessionInputs},
    value::Value,
};
use shared::{MnistInput, MnistOutput};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tower_http::cors::CorsLayer;

/// Actor that owns the ONNX session and processes inference requests.
struct InferenceActor {
    session: Session,
}

impl Actor for InferenceActor {
    type Context = SyncContext<Self>;
}

impl InferenceActor {
    /// Creates a new inference actor with the given model.
    fn new(model_path: &PathBuf) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(Self { session })
    }
}

/// Message containing preprocessed input for inference.
struct InferMessage {
    input_array: ndarray::ArrayD<f32>,
}

impl Message for InferMessage {
    type Result = Vec<f32>;
}

impl Handler<InferMessage> for InferenceActor {
    type Result = Vec<f32>;

    /// Runs inference on the input tensor and returns logits.
    fn handle(&mut self, msg: InferMessage, _ctx: &mut Self::Context) -> Self::Result {
        let input_value =
            Value::from_array(msg.input_array.clone()).expect("Failed to create input tensor");

        let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
            std::borrow::Cow::Borrowed("input"),
            SessionInputValue::Owned(input_value.into()),
        )]);

        let outputs = self.session.run(inputs).expect("Inference failed");

        let output_tensor = outputs.get("output").expect("Output not found");
        let (_shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        data.to_vec()
    }
}

/// Application state containing the inference actor address.
struct AppState {
    actor: Addr<InferenceActor>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/mobilenetv2-12-int8.onnx");

    let model_path_clone = model_path.clone();
    let actor = SyncArbiter::start(1, move || {
        InferenceActor::new(&model_path_clone).expect("Failed to create actor")
    });

    let state = Arc::new(AppState { actor });

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3003").await?;
    println!("actix-without-batching-server listening on http://0.0.0.0:3003");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Handles inference requests by sending to the actor and awaiting response.
async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<MnistInput>,
) -> Json<MnistOutput> {
    let start = Instant::now();

    let input_array = input.to_input_array().expect("Failed to process image");

    let logits = state
        .actor
        .send(InferMessage { input_array })
        .await
        .expect("Actor send failed");

    let output = MnistOutput::from_logits(&logits);

    let elapsed = start.elapsed();
    println!("Request completed in {:?}", elapsed);

    Json(output)
}