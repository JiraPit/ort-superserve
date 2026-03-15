use anyhow::Result;
use axum::{Json, Router, extract::State, routing::post};
use batched_fn::batched_fn;
use ndarray::ArrayD;
use once_cell::sync::Lazy;
use ort::{
    session::{Session, SessionInputValue, SessionInputs, builder::GraphOptimizationLevel},
    value::Value,
};
use shared::{MnistInput, MnistOutput};
use std::future::Future;
use std::pin::Pin;
use std::sync::Mutex;
use std::{path::PathBuf, sync::Arc};
use tokio::time::Instant as TokioInstant;
use tower_http::cors::CorsLayer;

static SESSION: Lazy<Mutex<Session>> = Lazy::new(|| {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("data/mnist-12.onnx");

    let session = Session::builder()
        .expect("Failed to create builder")
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("Failed to set optimization")
        .with_intra_threads(4)
        .expect("Failed to set threads")
        .commit_from_file(&model_path)
        .expect("Failed to load model");

    Mutex::new(session)
});

type Batch<T> = Vec<T>;
type PreprocessedInput = ArrayD<f32>;
type InferenceResult = Vec<f32>;

fn predict_batch(batch: Batch<PreprocessedInput>) -> Batch<InferenceResult> {
    if batch.is_empty() {
        return Vec::new();
    }

    // === STACK ARRAYS (no lock) ===
    let views: Vec<_> = batch.iter().map(|a| a.view()).collect();
    let batched = match ndarray::stack(ndarray::Axis(0), &views) {
        Ok(b) => b.into_dyn(),
        Err(_) => {
            return vec![vec![0.0; 10]; batch.len()];
        }
    };

    let input_value = match Value::from_array(batched) {
        Ok(v) => v,
        Err(_) => {
            return vec![vec![0.0; 10]; batch.len()];
        }
    };

    let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
        std::borrow::Cow::Borrowed("Input3"),
        SessionInputValue::Owned(input_value.into()),
    )]);

    // === INFERENCE (hold lock only here) ===
    let all_logits = {
        let mut session = SESSION.lock().unwrap();
        let outputs = match session.run(inputs) {
            Ok(o) => o,
            Err(_) => {
                return vec![vec![0.0; 10]; batch.len()];
            }
        };

        let output_tensor = match outputs.get("Plus214_Output_0") {
            Some(t) => t,
            None => {
                return vec![vec![0.0; 10]; batch.len()];
            }
        };

        let (_shape, data) = match output_tensor.try_extract_tensor::<f32>() {
            Ok((s, d)) => (s, d),
            Err(_) => {
                return vec![vec![0.0; 10]; batch.len()];
            }
        };

        data.to_vec()
    };
    // === LOCK RELEASED ===

    // === SPLIT LOGITS (no lock) ===
    let num_classes = 10;
    let mut results = Vec::with_capacity(batch.len());
    for i in 0..batch.len() {
        let start = i * num_classes;
        let end = start + num_classes;
        results.push(all_logits[start..end].to_vec());
    }
    results
}

type BatchPredictFn = Arc<
    dyn Fn(
            PreprocessedInput,
        )
            -> Pin<Box<dyn Future<Output = Result<InferenceResult, batched_fn::Error>> + Send>>
        + Send
        + Sync,
>;
type AppState = BatchPredictFn;

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

async fn infer_handler(
    State(batch_predict): State<AppState>,
    Json(input): Json<MnistInput>,
) -> Json<MnistOutput> {
    let start = TokioInstant::now();

    // === PREPROCESSING (parallel in handler) ===
    let input_array = input.to_input_array().expect("Failed to preprocess image");

    // === BATCHED INFERENCE ===
    let logits = batch_predict(input_array)
        .await
        .expect("Batched inference failed");

    // === POSTPROCESSING (parallel in handler) ===
    let output = MnistOutput::from_logits(&logits);

    let elapsed = start.elapsed();
    println!("Request completed in {:?}", elapsed);

    Json(output)
}
