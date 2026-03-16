//! Benchmark server using Actix actors without batching.
//!
//! This implementation uses actix SyncArbiter to run inference on a dedicated
//! thread, avoiding async runtime blocking. Each request is processed
//! individually without batching, making it simpler but less efficient under load.

use actix::{Actor, Addr, Handler, Message, SyncArbiter, SyncContext};
use anyhow::Result;
use axum::{Json, Router, extract::State, routing::post};
use ort::{
    session::{Session, SessionInputValue, SessionInputs, builder::GraphOptimizationLevel},
    value::Value,
};
use shared::{ImageInput, ImageOutput, apply_imagenet_normalization};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tower_http::cors::CorsLayer;

/// Actor that owns the ONNX session and processes inference requests.
struct InferenceActor {
    session: Session,
    input_name: String,
    output_name: String,
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
            .with_intra_threads(1)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        
        let input_name = session.inputs().first().map(|i| i.name().to_string()).expect("Model has no inputs");
        let output_name = session.outputs().first().map(|o| o.name().to_string()).expect("Model has no outputs");
        
        Ok(Self { session, input_name, output_name })
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
        let shape = msg.input_array.shape();
        let batched = if shape.len() == 3 {
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]),
                msg.input_array.clone().into_raw_vec_and_offset().0,
            )
            .expect("Failed to add batch dimension")
        } else {
            msg.input_array.clone()
        };

        let input_value = Value::from_array(batched).expect("Failed to create input tensor");

        let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
            std::borrow::Cow::Borrowed(&self.input_name),
            SessionInputValue::Owned(input_value.into()),
        )]);

        let outputs = self.session.run(inputs).expect("Inference failed");

        let output_tensor = outputs.get(&self.output_name).expect("Output not found");
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

#[actix::main]
async fn main() -> Result<()> {
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "resnet50-v1-12-int8".to_string());
    let model_filename = format!("{}.onnx", model_name);
    
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_assets")
        .join(&model_name)
        .join(&model_filename);

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
    Json(input): Json<ImageInput>,
) -> Json<ImageOutput> {
    let start = Instant::now();

    let input_array = tokio::task::spawn_blocking(move || {
        let arr = input.to_input_array().expect("Failed to process image");
        apply_imagenet_normalization(arr)
    })
    .await
    .expect("Preprocessing task failed");

    let logits: Vec<f32> = state
        .actor
        .send(InferMessage { input_array })
        .await
        .expect("Actor send failed");

    if logits.is_empty() {
        panic!("Inference returned empty logits");
    }

    let output = ImageOutput::from_logits(&logits);

    let _elapsed = start.elapsed();

    Json(output)
}
