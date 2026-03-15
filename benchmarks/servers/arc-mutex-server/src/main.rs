//! Benchmark server using naive Arc<Mutex<Session>> approach.
//!
//! This implementation represents the simplest but least scalable approach
//! to ONNX session management. A single session is wrapped in a mutex,
/// creating contention under concurrent load.
use anyhow::Result;
use axum::{Json, Router, extract::State, routing::post};
use ort::{
    session::{Session, SessionInputValue, SessionInputs, builder::GraphOptimizationLevel},
    value::Value,
};
use shared::{MnistInput, MnistOutput};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

/// Application state containing the mutex-protected ONNX session.
struct AppState {
    session: Arc<Mutex<Session>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/mobilenetv2-12-int8.onnx");

    let session = Session::builder()
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .with_intra_threads(4)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .commit_from_file(&model_path)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;

    let state = Arc::new(AppState {
        session: Arc::new(Mutex::new(session)),
    });

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3004").await?;
    println!("arc-mutex-server listening on http://0.0.0.0:3004");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Handles inference requests with mutex-locked session access.
///
/// Preprocessing and postprocessing run sequentially within each request.
/// The mutex is held for the entire duration of inference, blocking
/// other concurrent requests.
async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<MnistInput>,
) -> Json<MnistOutput> {
    let start = Instant::now();

    let input_array = input.to_input_array().expect("Failed to process image");

    let shape = input_array.shape();
    let batched = if shape.len() == 3 {
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&[1, shape[0], shape[1], shape[2]]),
            input_array.clone().into_raw_vec_and_offset().0,
        )
        .expect("Failed to add batch dimension")
    } else {
        input_array.clone()
    };

    let logits = {
        let mut session = state.session.lock().await;

        let input_value = Value::from_array(batched).expect("Failed to create input tensor");

        let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
            std::borrow::Cow::Borrowed("input"),
            SessionInputValue::Owned(input_value.into()),
        )]);

        let outputs = session.run(inputs).expect("Inference failed");

        let output_tensor = outputs.get("output").expect("Output not found");
        let (_shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        data.to_vec()
    };

    let output = MnistOutput::from_logits(&logits);

    let _elapsed = start.elapsed();

    Json(output)
}
