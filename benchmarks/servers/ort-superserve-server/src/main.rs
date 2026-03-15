//! Benchmark server using the ort-superserve library.
//!
//! This implementation demonstrates the minimal code required to build a
//! production-grade inference server with dynamic batching and parallel
//! preprocessing/postprocessing.

use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::time::Duration;

use anyhow::Result;
use axum::{Json, Router, extract::State, routing::post};
use ndarray::{ArrayD, ArrayViewD};
use ort::session::builder::GraphOptimizationLevel;
use ort_superserve::{Input, Output, Server, ServerConfig, helpers::batch_array};
use shared::{MnistInput, MnistOutput};
use tower_http::cors::CorsLayer;

/// Input wrapper implementing the ort-superserve `Input` trait.
struct MyInput {
    /// Raw PNG image bytes.
    image_bytes: Vec<u8>,
}

impl Input for MyInput {
    type Preprocessed = ArrayD<f32>;

    /// Preprocesses the PNG bytes into a normalized tensor.
    async fn preprocess(self) -> Result<Self::Preprocessed> {
        let input = MnistInput::from_png_bytes(self.image_bytes);
        input.to_input_array()
    }

    /// Stacks multiple preprocessed inputs into a batched tensor.
    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        batch_array(&items)
    }
}

/// Output wrapper implementing the ort-superserve `Output` trait.
struct MyOutput {
    /// Predicted digit class.
    digit: usize,
    /// Confidence score.
    confidence: f32,
}

impl Output for MyOutput {
    /// Postprocesses raw logits into predicted digit and confidence.
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
        let logits: Vec<f32> = raw.iter().copied().collect();
        let output = MnistOutput::from_logits(&logits);
        Ok(MyOutput {
            digit: output.digit,
            confidence: output.confidence,
        })
    }
}

/// Application state containing the ort-superserve server instance.
struct AppState {
    server: Server<MyInput, MyOutput>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/mobilenetv2-12-int8.onnx");

    let config = ServerConfig::new()
        .with_num_sessions(1)
        .with_threads_per_session(4)
        .with_max_batch_size(32)
        .with_min_batch_size(1)
        .with_max_wait_time(Duration::from_millis(10))
        .with_optimization_level(GraphOptimizationLevel::Level3);

    let server = Server::<MyInput, MyOutput>::from_file(&model_path, config).await?;

    let state = Arc::new(AppState { server });

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await?;
    println!("ort-superserve-server listening on http://0.0.0.0:3001");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Handles inference requests by delegating to the ort-superserve server.
async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<MnistInput>,
) -> Json<MnistOutput> {
    let start = Instant::now();

    let my_input = MyInput {
        image_bytes: input.image_bytes,
    };

    let output = state
        .server
        .infer(my_input)
        .await
        .expect("Inference failed");

    let _elapsed = start.elapsed();

    Json(MnistOutput {
        digit: output.digit,
        confidence: output.confidence,
    })
}
