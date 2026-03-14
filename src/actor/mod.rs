//! Actor system for request batching and worker management.
//!
//! This module contains the internal actors that handle:
//! - **Batcher**: Collects requests, runs preprocessing in parallel,
//!   and dispatches batches to workers.
//! - **Worker**: Runs ONNX inference in dedicated threads.
//! - **Command**: Internal message type for communication.

pub mod batcher;
pub mod command;
pub mod worker;

pub use batcher::BatcherTask;
pub use command::Command;
pub use worker::{WorkerMessage, WorkerTask};
