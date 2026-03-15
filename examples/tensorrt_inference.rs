use anyhow::Result;
use ndarray::{Array3, ArrayD, ArrayViewD, Axis};
use ort_superserve::{ExecutionProvider, Input, Output, Server, ServerConfig};

struct ArrayInput {
    data: Array3<f32>,
}

impl Input for ArrayInput {
    type Preprocessed = Array3<f32>;

    async fn preprocess(self) -> Result<Self::Preprocessed> {
        // Preprocessing logic goes here
        Ok(self.data)
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        let views: Vec<_> = items.iter().map(|a| a.view()).collect();
        let batched = ndarray::stack(Axis(0), &views)?;
        Ok(batched.into_dyn())
    }
}

#[derive(Debug)]
struct ArrayOutput {
    #[allow(dead_code)]
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let config = ServerConfig::new()
        .with_num_sessions(4)
        .with_threads_per_session(1)
        .with_max_batch_size(32)
        .with_min_batch_size(1)
        .with_execution_provider(ExecutionProvider::TensorRT {
            device_id: 0,
            fp16: true,
        });

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: tensorrt_inference <model.onnx>");

    let server = Server::<ArrayInput, ArrayOutput>::from_file(&model_path, config).await?;

    println!("Server initialized with TensorRT execution provider (FP16 enabled)!");
    println!("Running optimized inference on GPU...");

    let input_data = Array3::<f32>::zeros((3, 224, 224));
    let input = ArrayInput { data: input_data };

    let result = server.infer(input).await?;

    println!("Output: {:?}", result);

    server.shutdown();

    Ok(())
}
