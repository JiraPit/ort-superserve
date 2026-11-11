//! Token sequences of different lengths sharing one batch.
//!
//! For a model with inputs `input_ids: [batch, sequence]` and
//! `attention_mask: [batch, sequence]` (both i64) and an output
//! `logits: [batch, sequence, num_labels]` (e.g. token classification), every
//! request is padded to the longest sequence in its batch, with `pad_token_id`
//! for the ids and `0` for the attention mask. Each caller receives `logits` cut
//! back to its own sequence length.
//!
//! Output axes are matched to the padded input dimension by their symbolic
//! name in the model. If your export uses auto-generated names (e.g. `logits:
//! [batch, unk__0, num_labels]`), the server refuses to start and names the
//! axis; declare the relation by replacing `.with_output_name("logits")` with
//! `.with_output(OutputSpec::new("logits").with_crop_axis(1, "input_ids", 1))`.

use anyhow::Result;
use ndarray::Array1;
use ort_superserve::{Input, InputSpec, Inputs, Output, Outputs, Server, ServerConfig, create_env};

struct TextInput {
    token_ids: Vec<i64>,
}

impl Input for TextInput {
    type Preprocessed = Inputs;
    /// Sequence length, to check the output was cut back.
    type Context = usize;

    async fn preprocess(self) -> Result<(Inputs, usize)> {
        let len = self.token_ids.len();
        let inputs = Inputs::new()
            .with("input_ids", Array1::from(self.token_ids))
            .with("attention_mask", Array1::<i64>::ones(len));
        Ok((inputs, len))
    }
}

#[derive(Debug)]
struct TokenLabels(#[allow(dead_code)] Vec<usize>);

impl Output<usize> for TokenLabels {
    async fn postprocess(outputs: Outputs, len: usize) -> Result<Self> {
        // `[sequence, num_labels]` for this request only, without padding.
        let logits = outputs.get::<f32>("logits")?;
        anyhow::ensure!(logits.shape()[0] == len, "output was not cut back");

        let labels = logits
            .outer_iter()
            .map(|token| {
                token
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(i, _)| i)
            })
            .collect();
        Ok(TokenLabels(labels))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: variable_length_text <model.onnx>");
    let pad_token_id = 0.0;

    let config = ServerConfig::new()
        .with_max_batch_size(16)
        .with_input(InputSpec::new("input_ids").with_pad_value(pad_token_id))
        .with_input(InputSpec::new("attention_mask").with_pad_value(0.0))
        .with_output_name("logits")
        .with_environment(create_env()?);

    let server = Server::<TextInput, TokenLabels>::from_file(&model_path, config).await?;

    let sentences: Vec<Vec<i64>> = vec![
        vec![101, 7592, 102],
        vec![101, 2023, 2003, 1037, 2936, 6251, 102],
        vec![101, 2460, 102, 0, 0],
    ];
    let requests = sentences
        .into_iter()
        .map(|token_ids| server.infer(TextInput { token_ids }));
    for result in futures::future::join_all(requests).await {
        println!("{:?}", result?);
    }

    server.shutdown();
    Ok(())
}
