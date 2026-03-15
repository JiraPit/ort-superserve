//! Benchmark server using the batched-fn crate for transparent batching.
//!
//! This implementation uses the `batched-fn` macro to automatically batch
//! incoming requests. The batching logic is handled by the macro, reducing
//! boilerplate compared to manual implementation.

use anyhow::Result;
use axum::{Json, Router, extract::State, routing::post};
use batched_fn::batched_fn;
use ndarray::ArrayD;
use once_cell::sync::Lazy;
use ort::{
    session::{Session, SessionInputValue, SessionInputs, builder::GraphOptimizationLevel},
    value::Value,
};
use shared::{ImageInput, ImageOutput, apply_imagenet_normalization};
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::{path::PathBuf, sync::Arc};
use tokio::time::Instant as TokioInstant;
use tower_http::cors::CorsLayer;

/// Global ONNX session protected by a mutex.
static SESSION: Lazy<Mutex<Session>> = Lazy::new(|| {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/resnet50-v1-12-int8.onnx");

    let session = Session::builder()
        .expect("Failed to create builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("Failed to set optimization")
        .with_intra_threads(1)
        .expect("Failed to set threads")
        .commit_from_file(&model_path)
        .expect("Failed to load model");

    Mutex::new(session)
});

/// Type alias for a batch of items.
type Batch<T> = Vec<T>;
/// Preprocessed input tensor type.
type PreprocessedInput = ArrayD<f32>;
/// Inference output type.
type InferenceResult = Vec<f32>;

/// Processes a batch of preprocessed inputs through the ONNX model.
///
/// This function stacks individual inputs into a batched tensor,
/// runs inference, and splits the results back to individual outputs.
fn predict_batch(batch: Batch<PreprocessedInput>) -> Batch<InferenceResult> {
    if batch.is_empty() {
        return Vec::new();
    }

    let views: Vec<_> = batch.iter().map(|a| a.view()).collect();
    let batched = match ndarray::stack(ndarray::Axis(0), &views) {
        Ok(b) => b.into_dyn(),
        Err(_) => {
            return vec![vec![0.0; 1000]; batch.len()];
        }
    };

    let input_value = match Value::from_array(batched) {
        Ok(v) => v,
        Err(_) => {
            return vec![vec![0.0; 1000]; batch.len()];
        }
    };

    let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
        std::borrow::Cow::Borrowed("input"),
        SessionInputValue::Owned(input_value.into()),
    )]);

    let all_logits = {
        let mut session = SESSION.lock().unwrap();
        let outputs = match session.run(inputs) {
            Ok(o) => o,
            Err(_) => {
                return vec![vec![0.0; 1000]; batch.len()];
            }
        };

        let output_tensor = match outputs.get("output") {
            Some(t) => t,
            None => {
                return vec![vec![0.0; 1000]; batch.len()];
            }
        };

        let (_shape, data) = match output_tensor.try_extract_tensor::<f32>() {
            Ok((s, d)) => (s, d),
            Err(_) => {
                return vec![vec![0.0; 1000]; batch.len()];
            }
        };

        data.to_vec()
    };

    let num_classes = 1000;
    let mut results = Vec::with_capacity(batch.len());
    for i in 0..batch.len() {
        let start = i * num_classes;
        let end = start + num_classes;
        results.push(all_logits[start..end].to_vec());
    }
    results
}

/// Type alias for the batched prediction function.
type BatchPredictFn = Arc<
    dyn Fn(
            PreprocessedInput,
        )
            -> Pin<Box<dyn Future<Output = Result<InferenceResult, batched_fn::Error>> + Send>>
        + Send
        + Sync,
>;
type AppState = BatchPredictFn;

/// Global batched prediction function created by the batched-fn macro.
static BATCH_PREDICT: Lazy<AppState> = Lazy::new(|| {
    let batched = batched_fn! {
        handler = |batch: Batch<PreprocessedInput>, _ctx: &()| -> Batch<InferenceResult> {
            predict_batch(batch)
        };
        config = {
            max_batch_size: 32,
            max_delay: 10,
        };
        context = {
            _ctx: (),
        };
    };
    Arc::new(move |input: PreprocessedInput| Box::pin(batched(input)))
});

#[tokio::main]
async fn main() -> Result<()> {
    Lazy::force(&SESSION);
    Lazy::force(&BATCH_PREDICT);

    let state: AppState = Arc::clone(&BATCH_PREDICT);

    let app = Router::new()
        .route("/infer", post(infer_handler))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3005").await?;
    println!("batched-fn-server listening on http://0.0.0.0:3005");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Handles inference requests by preprocessing and calling the batched function.
async fn infer_handler(
    State(batch_predict): State<AppState>,
    Json(input): Json<ImageInput>,
) -> Json<ImageOutput> {
    let start = TokioInstant::now();

    let input_array = tokio::task::spawn_blocking(move || {
        let arr = input.to_input_array().expect("Failed to preprocess image");
        apply_imagenet_normalization(arr)
    })
    .await
    .expect("Preprocessing task failed");

    let logits = batch_predict(input_array)
        .await
        .expect("Batched inference failed");

    if logits.is_empty() {
        panic!("Inference returned empty logits");
    }

    let output = ImageOutput::from_logits(&logits);

    let _elapsed = start.elapsed();

    Json(output)
}
