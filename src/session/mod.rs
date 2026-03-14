//! ONNX Runtime session management.
//!
//! This module provides session construction with support for various
//! execution providers (CPU, CUDA, TensorRT, XNNPACK, CoreML).

mod builder;

pub use builder::SessionBuilder;
pub use ort::session::Session;
