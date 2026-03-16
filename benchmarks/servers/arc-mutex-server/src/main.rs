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
use shared::{ImageInput, ImageOutput, apply_imagenet_normalization};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;

/// Application state containing the mutex-protected ONNX session.
struct AppState {
    session: Arc<Mutex<Session>>,
    input_name: String,
    output_name: String,
}

#[tokio::main]
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

    let session = Session::builder()
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .with_intra_threads(1)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?
        .commit_from_file(&model_path)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;

    let input_name = session.inputs().first().map(|i| i.name().to_string()).expect("Model has no inputs");
    let output_name = session.outputs().first().map(|o| o.name().to_string()).expect("Model has no outputs");

    let state = Arc::new(AppState {
        session: Arc::new(Mutex::new(session)),
        input_name,
        output_name,
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
    Json(input): Json<ImageInput>,
) -> Json<ImageOutput> {
    let start = Instant::now();

    let input_array = tokio::task::spawn_blocking(move || {
        let arr = input.to_input_array().expect("Failed to process image");
        apply_imagenet_normalization(arr)
    })
    .await
    .expect("Preprocessing task failed");

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
            std::borrow::Cow::Borrowed(&state.input_name),
            SessionInputValue::Owned(input_value.into()),
        )]);

        let outputs = session.run(inputs).expect("Inference failed");

        let output_tensor = outputs.get(&state.output_name).expect("Output not found");
        let (_shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        data.to_vec()
    };

    if logits.is_empty() {
        panic!("Inference returned empty logits");
    }

    let output = ImageOutput::from_logits(&logits);

    let _elapsed = start.elapsed();

    Json(output)
}
