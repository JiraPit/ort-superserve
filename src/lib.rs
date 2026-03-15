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
//! - **Works with all input/output types**: The `Input` and `Output` traits let you
//!   define custom preprocessing and postprocessing logic for any data type.
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
//! ```rust,no_run
//! use anyhow::Result;
//! use ndarray::{ArrayD, ArrayViewD, Array3};
//! use ort_superserve::{helpers::batch_array, Input, Output, Server, ServerConfig};
//!
//! struct ImageInput { data: Array3<f32> }
//!
//! impl Input for ImageInput {
//!     type Preprocessed = Array3<f32>;
//!
//!     async fn preprocess(self) -> Result<Self::Preprocessed> {
//!         Ok(self.data)
//!     }
//!
//!     fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
//!         batch_array(&items)
//!     }
//! }
//!
//! struct DetectionOutput { scores: Vec<f32> }
//!
//! impl Output for DetectionOutput {
//!     async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
//!         Ok(DetectionOutput { scores: raw.iter().cloned().collect() })
//!     }
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let config = ServerConfig::new()
//!         .with_num_sessions(4)
//!         .with_max_batch_size(16);
//!
//!     let server = Server::<ImageInput, DetectionOutput>::from_file("model.onnx", config).await?;
//!     let output = server.infer(ImageInput { data: Array3::zeros((3, 224, 224)) }).await?;
//!     server.shutdown();
//!     Ok(())
//! }
//! ```

pub mod actor;
pub mod config;
pub mod error;
pub mod helpers;
pub mod server;
pub mod session;
pub mod traits;

pub use config::{ExecutionProvider, ServerConfig, SessionBuilderCallback};
pub use error::Error;
pub use server::Server;
pub use traits::{Input, Output};
