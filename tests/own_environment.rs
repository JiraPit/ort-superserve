//! Tests for a server whose config has no ONNX Runtime environment.
//!
//! ONNX Runtime allows only one environment per process, so this lives in its
//! own test binary where nothing else has created one.

mod common;

use common::*;
use ort_superserve::{Server, ServerConfig};
use std::time::Duration;

#[tokio::test]
async fn server_without_configured_environment_creates_its_own_once_per_process() {
    let model = ModelFixture::doubling();
    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_max_wait_time(Duration::from_millis(5));

    let server = Server::<TestInput, TestOutput>::from_file(model.path(), config.clone())
        .await
        .expect("first server should create the environment");
    assert_eq!(
        server.infer(TestInput::seq(1.0)).await.unwrap(),
        doubled_seq(1.0)
    );

    let second = Server::<TestInput, TestOutput>::from_file(model.path(), config).await;
    let message = format!("{:#}", second.err().expect("second environment must fail"));
    assert!(
        message.contains("create_env()"),
        "unexpected error: {message}"
    );
}
