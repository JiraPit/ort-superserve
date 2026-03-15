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
        .join("data/mnist-12.onnx");

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

async fn infer_handler(
    State(state): State<Arc<AppState>>,
    Json(input): Json<MnistInput>,
) -> Json<MnistOutput> {
    let start = Instant::now();

    // Preprocessing (sequential)
    let input_array = input.to_input_array().expect("Failed to process image");

    // Acquire lock and run inference
    let logits = {
        let mut session = state.session.lock().await;

        // Create input tensor
        let input_value =
            Value::from_array(input_array.clone()).expect("Failed to create input tensor");

        let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
            std::borrow::Cow::Borrowed("Input3"),
            SessionInputValue::Owned(input_value.into()),
        )]);

        // Run inference
        let outputs = session.run(inputs).expect("Inference failed");

        // Extract output
        let output_tensor = outputs.get("Plus214_Output_0").expect("Output not found");
        let (_shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        // Copy data while holding the lock
        data.to_vec()
    };

    // Postprocessing (sequential)
    let output = MnistOutput::from_logits(&logits);

    let elapsed = start.elapsed();
    println!("Request completed in {:?}", elapsed);

    Json(output)
}
