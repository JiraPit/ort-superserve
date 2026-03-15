use anyhow::Result;
use ndarray::{Array3, ArrayD, ArrayViewD, Axis};
use ort_superserve::{Input, Output, Server, ServerConfig};

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

#[derive(Debug)]
struct ArrayOutput {
    #[allow(dead_code)]
    scores: Vec<f32>,
}

impl Output for ArrayOutput {
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
        let shape = raw.shape();
        let batch_size = shape[0];

        let mut scores = Vec::new();
        for i in 0..batch_size {
            let slice = raw.index_axis(Axis(0), i);
            scores.push(slice[[0]]);
        }

        Ok(ArrayOutput { scores })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_threads_per_session(2)
        .with_max_batch_size(8)
        .with_min_batch_size(1)
        .with_input_name("images")
        .with_output_name("output");

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: custom_names <model.onnx>");

    let server = Server::<ArrayInput, ArrayOutput>::from_file(&model_path, config).await?;

    println!("Server initialized with custom input/output names!");
    println!("Running inference...");

    let input_data = Array3::<f32>::zeros((3, 224, 224));
    let input = ArrayInput { data: input_data };

    let result = server.infer(input).await?;

    println!("Output: {:?}", result);

    server.shutdown();

    Ok(())
}
