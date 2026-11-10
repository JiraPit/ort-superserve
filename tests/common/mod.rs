//! Shared fixtures for integration tests.
//!
//! Provides tiny ONNX models built in-process (no downloads required) and an
//! `Input`/`Output` pair that can record batch sizes and inject failures.

#![allow(dead_code)]

pub mod onnx;

use anyhow::Result;
use ndarray::{Array1, ArrayD, ArrayViewD};
use ort_superserve::{
    Environment, Input, Output, Server, ServerConfig, create_env, helpers::batch_array,
};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use tempfile::NamedTempFile;

/// Number of features per item expected by the fixture models (`[N, 4]`).
pub const FEATURES: usize = 4;

/// Output value that makes `TestOutput::postprocess` return an error.
pub const POSTPROCESS_ERROR_MARKER: f32 = -1000.0;

/// Output value that makes `TestOutput::postprocess` panic.
pub const POSTPROCESS_PANIC_MARKER: f32 = -2000.0;

/// Environment shared by every server in a test binary.
///
/// ONNX Runtime allows one environment per process, so it is created lazily
/// once and cloned, the same way an application is expected to use it.
pub fn env() -> Environment {
    static ENV: OnceLock<Environment> = OnceLock::new();
    ENV.get_or_init(|| create_env().expect("Failed to create environment"))
        .clone()
}

/// An ONNX model written to a temporary file.
///
/// Workers load the model from disk after `Server::from_file` returns, so the
/// fixture must outlive the server.
pub struct ModelFixture {
    file: NamedTempFile,
}

impl ModelFixture {
    pub fn new(bytes: Vec<u8>) -> Self {
        let file = tempfile::Builder::new()
            .suffix(".onnx")
            .tempfile()
            .expect("Failed to create temp model file");
        std::fs::write(file.path(), bytes).expect("Failed to write temp model file");
        Self { file }
    }

    /// `y = x * 2`, input `x: [N, 4]`, output `y: [N, 4]`.
    pub fn doubling() -> Self {
        Self::new(onnx::doubling_model())
    }

    /// Outputs `doubled = x * 2` (first) and `incremented = x + 1` (second).
    pub fn two_outputs() -> Self {
        Self::new(onnx::two_outputs_model())
    }

    /// Reduces over the batch axis, so the output is always `[1, 4]`.
    pub fn batch_collapsing() -> Self {
        Self::new(onnx::batch_collapsing_model())
    }

    /// Casts the input to int64, so the output is not `f32`.
    pub fn int64_output() -> Self {
        Self::new(onnx::int64_output_model())
    }

    pub fn path(&self) -> &Path {
        self.file.path()
    }
}

/// Records the size of every batch passed to `TestInput::batch`.
#[derive(Clone, Default)]
pub struct BatchRecorder(Arc<Mutex<Vec<usize>>>);

impl BatchRecorder {
    pub fn sizes(&self) -> Vec<usize> {
        self.0.lock().unwrap().clone()
    }

    fn record(&self, size: usize) {
        self.0.lock().unwrap().push(size);
    }
}

#[derive(Clone, Copy)]
pub enum Behavior {
    Succeed,
    FailPreprocess,
    PanicPreprocess,
    FailBatch,
}

pub struct TestInput {
    pub values: Vec<f32>,
    pub behavior: Behavior,
    pub recorder: BatchRecorder,
}

impl TestInput {
    /// Input whose values are `[start, start + 1, start + 2, start + 3]`.
    pub fn seq(start: f32) -> Self {
        Self::from_values((0..FEATURES).map(|i| start + i as f32).collect())
    }

    pub fn from_values(values: Vec<f32>) -> Self {
        Self {
            values,
            behavior: Behavior::Succeed,
            recorder: BatchRecorder::default(),
        }
    }

    /// Input that the doubling model maps to `marker` in every position.
    pub fn producing(marker: f32) -> Self {
        Self::from_values(vec![marker / 2.0; FEATURES])
    }

    pub fn with_behavior(mut self, behavior: Behavior) -> Self {
        self.behavior = behavior;
        self
    }

    pub fn with_recorder(mut self, recorder: &BatchRecorder) -> Self {
        self.recorder = recorder.clone();
        self
    }
}

pub struct Prepared {
    values: Array1<f32>,
    fail_batch: bool,
    recorder: BatchRecorder,
}

impl Input for TestInput {
    type Preprocessed = Prepared;

    async fn preprocess(self) -> Result<Self::Preprocessed> {
        match self.behavior {
            Behavior::FailPreprocess => anyhow::bail!("injected preprocess failure"),
            Behavior::PanicPreprocess => panic!("injected preprocess panic"),
            Behavior::Succeed | Behavior::FailBatch => Ok(Prepared {
                values: Array1::from_vec(self.values),
                fail_batch: matches!(self.behavior, Behavior::FailBatch),
                recorder: self.recorder,
            }),
        }
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        if let Some(first) = items.first() {
            first.recorder.record(items.len());
        }
        if items.iter().any(|item| item.fail_batch) {
            anyhow::bail!("injected batch failure");
        }
        let arrays: Vec<_> = items.into_iter().map(|item| item.values).collect();
        batch_array(&arrays)
    }
}

#[derive(Debug, PartialEq)]
pub struct TestOutput(pub Vec<f32>);

impl Output for TestOutput {
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
        let values: Vec<f32> = raw.iter().copied().collect();
        match values.first() {
            Some(&v) if v == POSTPROCESS_ERROR_MARKER => {
                anyhow::bail!("injected postprocess failure")
            }
            Some(&v) if v == POSTPROCESS_PANIC_MARKER => panic!("injected postprocess panic"),
            _ => Ok(TestOutput(values)),
        }
    }
}

pub async fn start_server(
    model: &ModelFixture,
    config: ServerConfig,
) -> Server<TestInput, TestOutput> {
    Server::from_file(model.path(), config.with_environment(env()))
        .await
        .expect("Failed to start server")
}

/// Expected doubling-model output for `TestInput::seq(start)`.
pub fn doubled_seq(start: f32) -> TestOutput {
    TestOutput((0..FEATURES).map(|i| (start + i as f32) * 2.0).collect())
}
