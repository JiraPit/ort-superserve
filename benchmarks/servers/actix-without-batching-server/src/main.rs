use actix::{Actor, Addr, Handler, Message, SyncArbiter, SyncContext};
use anyhow::Result;
use axum::{Json, Router, extract::State, routing::post};
use ort::{
    session::{Session, SessionInputValue, SessionInputs, builder::GraphOptimizationLevel},
    value::Value,
};
use shared::{MnistInput, MnistOutput};
use std::{path::PathBuf, sync::Arc, time::Instant};
use tower_http::cors::CorsLayer;

struct InferenceActor {
    session: Session,
}

impl Actor for InferenceActor {
    type Context = SyncContext<Self>;
}

impl InferenceActor {
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

struct InferMessage {
    input_array: ndarray::ArrayD<f32>,
}

impl Message for InferMessage {
    type Result = Vec<f32>;
}

impl Handler<InferMessage> for InferenceActor {
    type Result = Vec<f32>;

    fn handle(&mut self, msg: InferMessage, _ctx: &mut Self::Context) -> Self::Result {
        let input_value =
            Value::from_array(msg.input_array.clone()).expect("Failed to create input tensor");

        let inputs: SessionInputs = SessionInputs::ValueMap(vec![(
            std::borrow::Cow::Borrowed("Input3"),
            SessionInputValue::Owned(input_value.into()),
        )]);

        let outputs = self.session.run(inputs).expect("Inference failed");

        let output_tensor = outputs.get("Plus214_Output_0").expect("Output not found");
        let (_shape, data) = output_tensor
            .try_extract_tensor::<f32>()
            .expect("Failed to extract tensor");

        data.to_vec()
    }
}

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
        .join("data/mnist-12.onnx");

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
