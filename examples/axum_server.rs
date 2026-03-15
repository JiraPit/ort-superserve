//! Axum HTTP server example showing how to share an ONNX session pool
//! across all requests using `Arc<Server>`.

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use ndarray::{Array1, ArrayD, ArrayViewD};
use ort_superserve::{Input, Output, Server, ServerConfig, helpers::batch_array};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::signal;

struct ArrayInput {
    data: Array1<f32>,
}

impl Input for ArrayInput {
    type Preprocessed = Array1<f32>;

    async fn preprocess(self) -> Result<Self::Preprocessed> {
        // Preprocessing logic goes here
        Ok(self.data)
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        batch_array(&items)
    }
}

#[derive(Debug, Serialize)]
struct ArrayOutput {
    values: Vec<f32>,
}

impl Output for ArrayOutput {
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
        // Postprocessing logic goes here
        Ok(ArrayOutput {
            values: raw.iter().cloned().collect(),
        })
    }
}

#[derive(Deserialize)]
struct InferRequest {
    data: Vec<f32>,
}

#[derive(Serialize)]
struct InferResponse {
    values: Vec<f32>,
}

async fn infer_handler(
    State(server): State<Arc<Server<ArrayInput, ArrayOutput>>>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    let input = ArrayInput {
        data: Array1::from_vec(req.data),
    };

    let output = server.infer(input).await.expect("Inference failed");

    Json(InferResponse {
        values: output.values,
    })
}

async fn health_handler() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: axum_server <model.onnx>");

    let config = ServerConfig::new()
        .with_num_sessions(4)
        .with_threads_per_session(2)
        .with_max_batch_size(16);

    let server = Arc::new(Server::<ArrayInput, ArrayOutput>::from_file(&model_path, config).await?);

    println!("Listening on http://0.0.0.0:3000");

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .route("/health", get(health_handler))
        .with_state(server);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
