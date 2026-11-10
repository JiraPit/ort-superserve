//! Tests for ONNX Runtime session construction and execution providers.

mod common;

use common::{ModelFixture, env};
use ort_superserve::session::SessionBuilder;
use ort_superserve::{ExecutionProvider, ServerConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn custom_provider_callback_is_applied_to_the_session() {
    let model = ModelFixture::doubling();
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let config = ServerConfig::new().with_execution_provider(ExecutionProvider::Custom(Arc::new(
        move |builder| {
            counter.fetch_add(1, Ordering::SeqCst);
            Ok(builder)
        },
    )));

    let session = SessionBuilder::build(&env(), model.path().to_str().unwrap(), &config);

    assert!(session.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn custom_provider_callback_error_fails_the_build() {
    let model = ModelFixture::doubling();
    let config = ServerConfig::new().with_execution_provider(ExecutionProvider::Custom(
        // ONNX Runtime rejects empty config keys, so the callback itself fails.
        Arc::new(|builder| Ok(builder.with_config_entry("", "value")?)),
    ));

    let err = SessionBuilder::build(&env(), model.path().to_str().unwrap(), &config)
        .expect_err("callback error should fail the build");

    assert!(
        err.to_string().contains("Config key"),
        "unexpected error: {err}"
    );
}

#[test]
fn providers_not_compiled_in_are_rejected() {
    let model = ModelFixture::doubling();
    let path = model.path().to_str().unwrap();

    let mut uncompiled: Vec<(ExecutionProvider, &str)> = Vec::new();
    #[cfg(not(feature = "cuda"))]
    uncompiled.push((ExecutionProvider::Cuda { device_id: 0 }, "cuda"));
    #[cfg(not(feature = "tensorrt"))]
    uncompiled.push((
        ExecutionProvider::TensorRT {
            device_id: 0,
            fp16: false,
        },
        "tensorrt",
    ));
    #[cfg(not(feature = "xnnpack"))]
    uncompiled.push((ExecutionProvider::Xnnpack, "xnnpack"));
    #[cfg(not(feature = "coreml"))]
    uncompiled.push((ExecutionProvider::CoreML, "coreml"));

    for (provider, feature) in uncompiled {
        let config = ServerConfig::new().with_execution_provider(provider);
        let err = SessionBuilder::build(&env(), path, &config).expect_err(feature);
        assert!(
            err.to_string().contains(&format!("{feature} feature")),
            "unexpected error for {feature}: {err}"
        );
    }
}
