//! A high-level ONNX Runtime session orchestrator for serving models with
//! dynamic batching and parallel processing.
//!
//! # Features
//!
//! - **Actor model with dynamic batching**: Incoming requests are collected
//!   into batches before inference. The batcher waits until either
//!   `max_batch_size` is reached or `max_wait_time` elapses.
//! - **Built on `ort` crate**: Supports all execution providers (CPU, CUDA,
//!   TensorRT, XNNPACK, CoreML).
//! - **Works with all input/output types**: Define custom `Input` and `Output`
//!   traits for any data type.
//! - **Session pool**: Multiple ONNX sessions for full hardware utilization.
//! - **Parallel pre/post-processing**: Concurrent preprocessing and
//!   postprocessing with `JoinSet`.
//!
//! # Example
//!
//! ```rust,no_run
//! use anyhow::Result;
//! use ndarray::{ArrayD, ArrayViewD, Array3, Axis};
//! use ort_superserve::{Input, Output, Server, ServerConfig};
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
//!         let views: Vec<_> = items.iter().map(|a| a.view()).collect();
//!         Ok(ndarray::stack(Axis(0), &views)?.into_dyn())
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
pub mod server;
pub mod session;
pub mod traits;

pub use config::{ExecutionProvider, ServerConfig};
pub use error::Error;
pub use server::Server;
pub use traits::{Input, Output};
