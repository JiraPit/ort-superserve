use anyhow::Result;
use ndarray::Array3;
use ort_superserve::{ExecutionProvider, Input, Output, Outputs, Server, ServerConfig, create_env};
use std::sync::Arc;

struct ArrayInput {
    data: Array3<f32>,
}

impl Input for ArrayInput {
    type Preprocessed = Array3<f32>;
    type Context = ();

    async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> {
        // Preprocessing logic goes here
        Ok((self.data, ()))
    }
}

#[derive(Debug)]
struct ArrayOutput {
    #[allow(dead_code)]
    values: Vec<f32>,
}

impl Output for ArrayOutput {
    async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
        let raw = outputs.first::<f32>()?;
        // Postprocessing logic goes here
        Ok(ArrayOutput {
            values: raw.iter().cloned().collect(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let custom_provider = ExecutionProvider::Custom(Arc::new(|builder| {
        Ok(builder
            .with_memory_pattern(true)?
            .with_execution_providers([ort::ep::CPU::default().build()])?)
    }));

    // Create the ONNX Runtime environment explicitly. ONNX Runtime allows one per
    // process, so applications with several servers should share (clone) it.
    let env = create_env()?;

    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_threads_per_session(2)
        .with_max_batch_size(8)
        .with_min_batch_size(1)
        .with_execution_provider(custom_provider)
        .with_environment(env);

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: custom_provider <model.onnx>");

    let server = Server::<ArrayInput, ArrayOutput>::from_file(&model_path, config).await?;

    println!("Server initialized with custom execution provider!");
    println!("Running inference...");

    let input_data = Array3::<f32>::zeros((3, 224, 224));
    let input = ArrayInput { data: input_data };

    let result = server.infer(input).await?;

    println!("Output: {:?}", result);

    server.shutdown();

    Ok(())
}
