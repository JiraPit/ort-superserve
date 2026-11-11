//! A "thin" asynchronous ONNX Runtime session orchestrator for high-throughput model serving.
//!
//! Orchestrates a pool of ONNX sessions with dynamic batching and parallel processing,
//! enabling thousands of concurrent requests to share the same model without mutex
//! contention on the hot path.
//!
//! # Features
//!
//! - **Lock-free**: Unlike the naive approach of wrapping an ONNX session in
//!   `Arc<Mutex<Session>>` which creates a bottleneck under load, ort-superserve
//!   uses an actor model where requests flow through channels instead of locks,
//!   and a session pool where multiple ONNX sessions run in parallel on dedicated
//!   threads—zero mutex on the hot path.
//! - **Dynamic batching**: Incoming requests are collected into batches before being
//!   sent to inference. The batcher waits until either `max_batch_size` is reached
//!   or `max_wait_time` elapses, ensuring optimal GPU/CPU utilization.
//! - **Built on [`ort`] crate**: Uses the mature ONNX Runtime bindings directly,
//!   supporting all execution providers: CPU (default), CUDA, TensorRT, XNNPACK,
//!   and CoreML.
//! - **Any inputs and outputs**: Models can have any number of named inputs and
//!   outputs of any common element type (`f32`, `i64`, `bool`, ...). Each output
//!   is split back per request by its [`Split`]: along the batch axis (default),
//!   by a batch-index column for NMS models, or with your own function.
//! - **Fixed batch sizes and variable shapes**: Partial batches are padded up to
//!   a model's fixed batch size, and requests with different lengths along a
//!   dynamic dimension (e.g. token sequences) are padded to the longest one in
//!   the batch with a configurable value per input. Padded rows never reach a
//!   caller, and outputs are cut back to each request's own length.
//! - **Safe splitting**: When an output cannot be split one entry per request,
//!   requests fail with a clear error instead of receiving someone else's data.
//! - **Custom pre/post-processing**: The `Input` and `Output` traits let you
//!   define preprocessing and postprocessing for any data type, with
//!   per-request context (like the original image size) passed from
//!   preprocessing to postprocessing.
//! - **Session pool**: Multiple ONNX sessions run in parallel, each with fewer
//!   threads, achieving near-linear scaling with core count.
//! - **Parallel pre/post-processing**: Preprocessing and postprocessing run
//!   concurrently using `JoinSet`, pipelining CPU work while the GPU handles
//!   inference.
//!
//! [`ort`]: https://crates.io/crates/ort
//!
//! # Example
//!
//! The common case of one `f32` input and one `f32` output:
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use ndarray::Array3;
//! use ort_superserve::{create_env, Input, Output, Outputs, Server, ServerConfig};
//!
//! struct ImageInput { data: Array3<f32> }
//!
//! impl Input for ImageInput {
//!     type Preprocessed = Array3<f32>;
//!     type Context = ();
//!
//!     async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> {
//!         Ok((self.data, ()))
//!     }
//! }
//!
//! struct ClassScores { scores: Vec<f32> }
//!
//! impl Output for ClassScores {
//!     async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
//!         Ok(ClassScores { scores: outputs.first::<f32>()?.iter().cloned().collect() })
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create the ONNX Runtime environment explicitly (one per process; clone it
//!     // to share it between servers).
//!     let env = create_env()?;
//!
//!     let config = ServerConfig::new()
//!         .with_num_sessions(4)
//!         .with_max_batch_size(16)
//!         .with_environment(env);
//!
//!     let server = Server::<ImageInput, ClassScores>::from_file("model.onnx", config).await?;
//!     let output = server.infer(ImageInput { data: Array3::zeros((3, 224, 224)) }).await?;
//!     server.shutdown();
//!     Ok(())
//! }
//! ```
//!
//! # Several inputs and outputs
//!
//! A text model with two `i64` inputs of varying length: requests are padded to
//! the longest sequence in the batch (ids with the pad token, the attention
//! mask with 0), and outputs along the sequence axis are cut back per request.
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use ndarray::Array1;
//! use ort_superserve::{Input, InputSpec, Inputs, Output, Outputs, Server, ServerConfig};
//!
//! struct Tokens(Vec<i64>);
//!
//! impl Input for Tokens {
//!     type Preprocessed = Inputs;
//!     type Context = ();
//!
//!     async fn preprocess(self) -> Result<(Inputs, ())> {
//!         let len = self.0.len();
//!         Ok((
//!             Inputs::new()
//!                 .with("input_ids", Array1::from(self.0))
//!                 .with("attention_mask", Array1::<i64>::ones(len)),
//!             (),
//!         ))
//!     }
//! }
//!
//! struct TokenLabels(Vec<i64>);
//!
//! impl Output for TokenLabels {
//!     async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
//!         // `[seq_len, num_labels]`, already cut back to this request's length.
//!         let logits = outputs.get::<f32>("logits")?;
//!         let labels = logits
//!             .outer_iter()
//!             .map(|row| {
//!                 row.iter()
//!                     .enumerate()
//!                     .max_by(|a, b| a.1.total_cmp(b.1))
//!                     .map_or(0, |(i, _)| i as i64)
//!             })
//!             .collect();
//!         Ok(TokenLabels(labels))
//!     }
//! }
//!
//! # async fn run() -> Result<()> {
//! let config = ServerConfig::new()
//!     .with_input(InputSpec::new("input_ids").with_pad_value(0.0))
//!     .with_input(InputSpec::new("attention_mask").with_pad_value(0.0));
//! let server = Server::<Tokens, TokenLabels>::from_file("ner.onnx", config).await?;
//! let labels = server.infer(Tokens(vec![101, 7592, 2088, 102])).await?;
//! # Ok(())
//! # }
//! ```
//!
//! Output axes are linked to the padded input dimensions by their symbolic
//! names in the model. If the exporter generated the names instead, the server
//! refuses to start until you declare the relation; see
//! [Outputs of models with padded inputs](OutputSpec#outputs-of-models-with-padded-inputs).
//!
//! See [`Split`] for detectors with NMS outputs and custom splitting.

pub mod actor;
pub mod config;
pub mod environment;
pub mod error;
pub mod helpers;
pub mod io;
mod model;
pub mod server;
pub mod session;
pub mod split;
pub mod tensor;
pub mod traits;

pub use config::{
    CropAxis, ExecutionProvider, InputSpec, OutputSpec, ServerConfig, SessionBuilderCallback,
};
pub use environment::{Environment, create_env};
pub use error::Error;
pub use io::{Inputs, Outputs};
pub use server::Server;
pub use split::{PaddedDim, Split, SplitContext, SplitFn, split_batch_axis, split_by_batch_index};
pub use tensor::{Element, ElementType, Tensor};
pub use traits::{Input, Output};
