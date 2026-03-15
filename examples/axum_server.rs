//! Axum HTTP server example showing how to share an ONNX session pool
//! across all requests using `Arc<Server>`.

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    routing::{get, post},
};
use ndarray::{Array3, ArrayD, ArrayViewD, Axis};
use ort_superserve::{Input, Output, Server, ServerConfig};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::signal;

/// Input type for the inference request.
struct ArrayInput {
    data: Array3<f32>,
}

impl Input for ArrayInput {
    type Preprocessed = Array3<f32>;

    async fn preprocess(self) -> Result<Self::Preprocessed> {
        tokio::task::spawn_blocking(move || Ok(self.data)).await?
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        let views: Vec<_> = items.iter().map(|a| a.view()).collect();
        let batched = ndarray::stack(Axis(0), &views)?;
        Ok(batched.into_dyn())
    }
}

/// Output type for the inference response.
#[derive(Debug, Serialize)]
struct ArrayOutput {
    scores: Vec<f32>,
}

impl Output for ArrayOutput {
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
        let scores: Vec<f32> = raw.iter().cloned().collect();
        Ok(ArrayOutput { scores })
    }
}

/// HTTP request body for inference.
#[derive(Deserialize)]
struct InferRequest {
    data: Vec<f32>,
    height: usize,
    width: usize,
    channels: usize,
}

/// HTTP response body for inference.
#[derive(Serialize)]
struct InferResponse {
    scores: Vec<f32>,
}

/// Handler for POST /infer endpoint.
async fn infer_handler(
    State(server): State<Arc<Server<ArrayInput, ArrayOutput>>>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    let data = Array3::from_shape_vec((req.channels, req.height, req.width), req.data)
        .expect("Invalid input shape");

    let input = ArrayInput { data };

    let output = server.infer(input).await.expect("Inference failed");

    Json(InferResponse {
        scores: output.scores,
    })
}

/// Handler for GET /health endpoint.
async fn health_handler() -> &'static str {
    "ok"
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: axum_server <model.onnx>");

    println!("Loading model from {}...", model_path);

    let config = ServerConfig::new()
        .with_num_sessions(4)
        .with_threads_per_session(2)
        .with_max_batch_size(16)
        .with_min_batch_size(1);

    println!("Server initialized with {} sessions", config.num_sessions);

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

    println!("Server shut down gracefully");
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
