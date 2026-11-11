//! End-to-end server behavior tests using tiny in-process ONNX models.

mod common;

use common::*;
use futures::future::join_all;
use ort_superserve::{Server, ServerConfig};
use std::time::{Duration, Instant};

/// Config whose batching window is long enough that concurrently submitted
/// requests are guaranteed to land in the same batch.
fn single_batch_config(max_batch_size: usize) -> ServerConfig {
    ServerConfig::new()
        .with_max_batch_size(max_batch_size)
        .with_max_wait_time(Duration::from_secs(2))
}

fn error_message(result: anyhow::Result<TestOutput>) -> String {
    format!("{:#}", result.expect_err("expected inference to fail"))
}

mod batching {
    use super::*;

    #[tokio::test]
    async fn each_caller_receives_its_own_result() {
        let model = ModelFixture::doubling();
        let config = ServerConfig::new()
            .with_max_batch_size(8)
            .with_max_wait_time(Duration::from_millis(20));
        let server = start_server(&model, config).await;

        let starts: Vec<f32> = (0..32).map(|i| i as f32 * 10.0).collect();
        let results = join_all(starts.iter().map(|&s| server.infer(TestInput::seq(s)))).await;

        for (start, result) in starts.into_iter().zip(results) {
            assert_eq!(result.unwrap(), doubled_seq(start));
        }
    }

    #[tokio::test]
    async fn full_batch_dispatches_without_waiting_for_timeout() {
        let model = ModelFixture::doubling();
        let config = ServerConfig::new()
            .with_max_batch_size(4)
            .with_max_wait_time(Duration::from_secs(30));
        let server = start_server(&model, config).await;
        let recorder = BatchRecorder::default();

        let started = Instant::now();
        let results = join_all(
            (0..4).map(|i| server.infer(TestInput::seq(i as f32).with_recorder(&recorder))),
        )
        .await;

        assert!(results.iter().all(Result::is_ok));
        assert_eq!(recorder.sizes(), vec![4]);
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[tokio::test]
    async fn partial_batch_flushes_after_max_wait_time() {
        let model = ModelFixture::doubling();
        let wait = Duration::from_millis(200);
        let config = ServerConfig::new()
            .with_max_batch_size(8)
            .with_max_wait_time(wait);
        let server = start_server(&model, config).await;
        let recorder = BatchRecorder::default();

        let started = Instant::now();
        let result = server
            .infer(TestInput::seq(1.0).with_recorder(&recorder))
            .await;

        assert_eq!(result.unwrap(), doubled_seq(1.0));
        assert_eq!(recorder.sizes(), vec![1]);
        assert!(started.elapsed() >= wait);
    }

    #[tokio::test]
    async fn batches_never_exceed_max_batch_size() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(4)).await;
        let recorder = BatchRecorder::default();

        let results = join_all(
            (0..10).map(|i| server.infer(TestInput::seq(i as f32).with_recorder(&recorder))),
        )
        .await;

        assert!(results.iter().all(Result::is_ok));
        let sizes = recorder.sizes();
        assert!(sizes.iter().all(|&s| s <= 4), "batch sizes: {sizes:?}");
        assert_eq!(sizes.iter().sum::<usize>(), 10);
    }

    #[tokio::test]
    async fn batch_below_min_batch_size_is_rejected() {
        let model = ModelFixture::doubling();
        let config = ServerConfig::new()
            .with_min_batch_size(2)
            .with_max_wait_time(Duration::from_millis(20));
        let server = start_server(&model, config).await;
        let recorder = BatchRecorder::default();

        let result = server
            .infer(TestInput::seq(0.0).with_recorder(&recorder))
            .await;

        assert!(error_message(result).contains("Insufficient batch size"));
        assert!(
            recorder.sizes().is_empty(),
            "rejected batch must not be built"
        );
    }

    #[tokio::test]
    async fn requests_are_spread_across_multiple_sessions() {
        let model = ModelFixture::doubling();
        let config = ServerConfig::new()
            .with_num_sessions(3)
            .with_max_batch_size(2)
            .with_max_wait_time(Duration::from_millis(5));
        let server = start_server(&model, config).await;

        let starts: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let results = join_all(starts.iter().map(|&s| server.infer(TestInput::seq(s)))).await;

        for (start, result) in starts.into_iter().zip(results) {
            assert_eq!(result.unwrap(), doubled_seq(start));
        }
    }

    #[tokio::test]
    async fn cloned_server_shares_the_same_pipeline() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(2)).await;
        let clone = server.clone();
        let recorder = BatchRecorder::default();

        let (a, b) = tokio::join!(
            server.infer(TestInput::seq(0.0).with_recorder(&recorder)),
            clone.infer(TestInput::seq(5.0).with_recorder(&recorder)),
        );

        assert_eq!(a.unwrap(), doubled_seq(0.0));
        assert_eq!(b.unwrap(), doubled_seq(5.0));
        assert_eq!(recorder.sizes(), vec![2]);
    }
}

mod failure_isolation {
    use super::*;

    #[tokio::test]
    async fn preprocess_error_fails_only_that_request() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(3)).await;

        let (ok_a, failed, ok_b) = tokio::join!(
            server.infer(TestInput::seq(1.0)),
            server.infer(TestInput::seq(2.0).with_behavior(Behavior::FailPreprocess)),
            server.infer(TestInput::seq(3.0)),
        );

        assert_eq!(ok_a.unwrap(), doubled_seq(1.0));
        assert!(error_message(failed).contains("Preprocessing failed"));
        assert_eq!(ok_b.unwrap(), doubled_seq(3.0));
    }

    #[tokio::test]
    async fn preprocess_panic_fails_only_that_request() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(3)).await;

        let (ok_a, panicked, ok_b) = tokio::join!(
            server.infer(TestInput::seq(1.0)),
            server.infer(TestInput::seq(2.0).with_behavior(Behavior::PanicPreprocess)),
            server.infer(TestInput::seq(3.0)),
        );

        assert_eq!(ok_a.unwrap(), doubled_seq(1.0));
        assert!(error_message(panicked).contains("Preprocessing failed"));
        assert_eq!(ok_b.unwrap(), doubled_seq(3.0));
    }

    #[tokio::test]
    async fn batch_where_every_preprocess_fails_does_not_reach_batching() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(2)).await;
        let recorder = BatchRecorder::default();

        let results = join_all((0..2).map(|i| {
            server.infer(
                TestInput::seq(i as f32)
                    .with_behavior(Behavior::FailPreprocess)
                    .with_recorder(&recorder),
            )
        }))
        .await;

        assert!(results.into_iter().all(|r| r.is_err()));
        assert!(recorder.sizes().is_empty());
    }

    #[tokio::test]
    async fn batching_error_fails_every_request_in_the_batch() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(2)).await;

        let (good, bad) = tokio::join!(
            server.infer(TestInput::seq(1.0)),
            server.infer(TestInput::seq(2.0).with_behavior(Behavior::FailBatch)),
        );

        assert!(error_message(good).contains("injected batch failure"));
        assert!(error_message(bad).contains("injected batch failure"));
    }

    #[tokio::test]
    async fn postprocess_error_fails_only_that_request() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(2)).await;

        let (ok, failed) = tokio::join!(
            server.infer(TestInput::seq(1.0)),
            server.infer(TestInput::producing(POSTPROCESS_ERROR_MARKER)),
        );

        assert_eq!(ok.unwrap(), doubled_seq(1.0));
        assert!(error_message(failed).contains("injected postprocess failure"));
    }

    #[tokio::test]
    async fn postprocess_panic_fails_only_that_request() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(2)).await;

        let (ok, panicked) = tokio::join!(
            server.infer(TestInput::seq(1.0)),
            server.infer(TestInput::producing(POSTPROCESS_PANIC_MARKER)),
        );

        assert_eq!(ok.unwrap(), doubled_seq(1.0));
        assert!(panicked.is_err());
    }

    #[tokio::test]
    async fn server_keeps_serving_after_an_inference_failure() {
        let model = ModelFixture::doubling();
        let config = ServerConfig::new().with_max_wait_time(Duration::from_millis(5));
        let server = start_server(&model, config).await;

        // The model expects 4 features; 5 makes ONNX Runtime reject the batch.
        let wrong_shape = server.infer(TestInput::from_values(vec![1.0; 5])).await;
        let afterwards = server.infer(TestInput::seq(1.0)).await;

        assert!(wrong_shape.is_err());
        assert_eq!(afterwards.unwrap(), doubled_seq(1.0));
    }
}

mod worker_failures {
    use super::*;

    #[tokio::test]
    async fn worker_survives_a_panic_while_processing_a_batch() {
        use ort_superserve::{Split, split_batch_axis};

        let model = ModelFixture::squeezing();
        // A lone request squeezes to a scalar output with no batch axis; the
        // custom split panics the worker while it slices such a batch.
        let split = Split::custom(|output, ctx| {
            assert!(output.ndim() > 0, "scalar output has no batch axis");
            split_batch_axis(output, ctx)
        });
        let config = ServerConfig::new()
            .with_num_sessions(1)
            .with_max_batch_size(2)
            .with_max_wait_time(Duration::from_millis(200))
            .with_output_split("y", split);
        let server = start_server(&model, config).await;

        let lone = server.infer(TestInput::from_values(vec![1.0])).await;
        // A batch of two keeps its batch axis; the only worker must still serve it.
        let (a, b) = tokio::join!(
            server.infer(TestInput::from_values(vec![2.0])),
            server.infer(TestInput::from_values(vec![3.0])),
        );

        assert!(error_message(lone).contains("worker returned no result"));
        assert_eq!(a.unwrap(), TestOutput(vec![2.0]));
        assert_eq!(b.unwrap(), TestOutput(vec![3.0]));
    }

    #[tokio::test]
    async fn startup_fails_when_a_worker_session_cannot_be_built() {
        use ort_superserve::ExecutionProvider;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let model = ModelFixture::doubling();
        // The first build (used to detect tensor names) succeeds; every worker's
        // build then fails because ONNX Runtime rejects an empty config key.
        let builds = Arc::new(AtomicUsize::new(0));
        let config = ServerConfig::new()
            .with_num_sessions(2)
            .with_environment(env())
            .with_execution_provider(ExecutionProvider::Custom(Arc::new(move |builder| {
                if builds.fetch_add(1, Ordering::SeqCst) == 0 {
                    Ok(builder)
                } else {
                    Ok(builder.with_config_entry("", "value")?)
                }
            })));

        let result = tokio::time::timeout(
            Duration::from_secs(30),
            Server::<TestInput, TestOutput>::from_file(model.path(), config),
        )
        .await
        .expect("startup must fail instead of waiting on a broken worker");

        let message = format!("{:#}", result.err().expect("expected startup failure"));
        assert!(
            message.contains("failed to initialize its session"),
            "unexpected error: {message}"
        );
    }
}

mod shutdown {
    use super::*;

    #[tokio::test]
    async fn clones_keep_serving_after_another_handle_shuts_down() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, single_batch_config(2)).await;
        let clone = server.clone();

        // Queue a request on the clone (it waits for the batch window), then
        // shut down the original handle while it is in flight.
        let in_flight = tokio::spawn(async move {
            let result = clone.infer(TestInput::seq(1.0)).await;
            (result, clone)
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        server.shutdown();

        let (result, clone) = in_flight.await.unwrap();
        assert_eq!(result.unwrap(), doubled_seq(1.0));
        assert_eq!(
            clone.infer(TestInput::seq(2.0)).await.unwrap(),
            doubled_seq(2.0)
        );
    }
}

mod eager_preprocessing {
    use super::*;

    fn eager_config() -> ServerConfig {
        single_batch_config(3).with_preprocess_on_infer(true)
    }

    #[tokio::test]
    async fn preprocessed_requests_are_batched_and_answered() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, eager_config()).await;
        let recorder = BatchRecorder::default();

        let results = join_all(
            (0..3).map(|i| server.infer(TestInput::seq(i as f32).with_recorder(&recorder))),
        )
        .await;

        for (i, result) in results.into_iter().enumerate() {
            assert_eq!(result.unwrap(), doubled_seq(i as f32));
        }
        assert_eq!(recorder.sizes(), vec![3]);
    }

    #[tokio::test]
    async fn preprocess_error_is_returned_directly_to_caller() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, eager_config()).await;
        let recorder = BatchRecorder::default();

        let result = server
            .infer(
                TestInput::seq(0.0)
                    .with_behavior(Behavior::FailPreprocess)
                    .with_recorder(&recorder),
            )
            .await;

        assert!(error_message(result).contains("injected preprocess failure"));
        assert!(
            recorder.sizes().is_empty(),
            "failed input must never be queued"
        );
    }
}

mod tensor_names {
    use super::*;

    fn quick_config() -> ServerConfig {
        ServerConfig::new().with_max_wait_time(Duration::from_millis(5))
    }

    #[tokio::test]
    async fn first_model_output_is_used_by_default() {
        let model = ModelFixture::two_outputs();
        let server = start_server(&model, quick_config()).await;

        let output = server.infer(TestInput::seq(1.0)).await.unwrap();

        assert_eq!(output, doubled_seq(1.0));
    }

    #[tokio::test]
    async fn configured_output_name_selects_that_output() {
        let model = ModelFixture::two_outputs();
        let server = start_server(&model, quick_config().with_output_name("incremented")).await;

        let output = server.infer(TestInput::seq(1.0)).await.unwrap();

        assert_eq!(output, TestOutput(vec![2.0, 3.0, 4.0, 5.0]));
    }

    #[tokio::test]
    async fn unknown_output_name_fails_inference() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, quick_config().with_output_name("missing")).await;

        let result = server.infer(TestInput::seq(1.0)).await;

        assert!(error_message(result).contains("Output tensor 'missing' not found"));
    }

    #[tokio::test]
    async fn unknown_input_name_fails_inference() {
        let model = ModelFixture::doubling();
        let server = start_server(&model, quick_config().with_input_name("missing")).await;

        let result = server.infer(TestInput::seq(1.0)).await;

        assert!(result.is_err());
    }
}

mod model_outputs {
    use super::*;

    #[tokio::test]
    async fn non_f32_output_fails_inference() {
        let model = ModelFixture::int64_output();
        let config = ServerConfig::new().with_max_wait_time(Duration::from_millis(5));
        let server = start_server(&model, config).await;

        let result = server.infer(TestInput::seq(1.0)).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn output_not_batched_on_first_axis_fails_every_request() {
        let model = ModelFixture::batch_collapsing();
        let server = start_server(&model, single_batch_config(2)).await;

        let (a, b) = tokio::join!(
            server.infer(TestInput::seq(1.0)),
            server.infer(TestInput::seq(2.0)),
        );

        assert!(error_message(a).contains("Output count mismatch"));
        assert!(error_message(b).contains("Output count mismatch"));
    }
}

mod loading {
    use super::*;
    use axum::{Router, http::StatusCode, routing::get};
    use std::collections::HashSet;
    use std::path::PathBuf;

    /// Serializes tests that download a model, so each can check the temp
    /// directory for files left behind by its own download.
    static DOWNLOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn downloaded_model_files() -> HashSet<PathBuf> {
        std::fs::read_dir(std::env::temp_dir())
            .unwrap()
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("ort-superserve-model-"))
            })
            .collect()
    }

    /// Serves `bytes` at `/model.onnx` on a random local port; everything
    /// else returns 404.
    async fn serve_model(bytes: Vec<u8>) -> String {
        let app = Router::new().route("/model.onnx", get(move || async move { bytes }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn missing_model_file_fails_to_start() {
        let result = Server::<TestInput, TestOutput>::from_file(
            "does/not/exist.onnx",
            ServerConfig::new().with_environment(env()),
        )
        .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn model_downloaded_from_url_serves_requests() {
        let _download = DOWNLOAD_LOCK.lock().await;
        let base = serve_model(common::onnx::doubling_model()).await;
        let config = ServerConfig::new().with_max_wait_time(Duration::from_millis(5));

        let server = Server::<TestInput, TestOutput>::from_url(
            &format!("{base}/model.onnx"),
            config.with_environment(env()),
        )
        .await
        .expect("Failed to start server from URL");

        assert_eq!(
            server.infer(TestInput::seq(1.0)).await.unwrap(),
            doubled_seq(1.0)
        );
    }

    #[tokio::test]
    async fn unsuccessful_http_status_fails_to_start() {
        let base = serve_model(common::onnx::doubling_model()).await;

        let result = Server::<TestInput, TestOutput>::from_url(
            &format!("{base}/missing.onnx"),
            ServerConfig::new().with_environment(env()),
        )
        .await;

        let message = format!("{:#}", result.err().expect("expected download failure"));
        assert!(message.contains(&StatusCode::NOT_FOUND.as_u16().to_string()));
    }

    #[tokio::test]
    async fn downloaded_model_file_is_removed_after_startup() {
        let _download = DOWNLOAD_LOCK.lock().await;
        let base = serve_model(common::onnx::doubling_model()).await;
        let before = downloaded_model_files();

        let server = Server::<TestInput, TestOutput>::from_url(
            &format!("{base}/model.onnx"),
            ServerConfig::new().with_environment(env()),
        )
        .await
        .expect("Failed to start server from URL");

        assert!(downloaded_model_files().is_subset(&before));
        drop(server);
    }

    #[tokio::test]
    async fn downloaded_model_file_is_removed_when_startup_fails() {
        let _download = DOWNLOAD_LOCK.lock().await;
        let base = serve_model(b"not an onnx model".to_vec()).await;
        let before = downloaded_model_files();

        let result = Server::<TestInput, TestOutput>::from_url(
            &format!("{base}/model.onnx"),
            ServerConfig::new().with_environment(env()),
        )
        .await;

        assert!(result.is_err());
        assert!(downloaded_model_files().is_subset(&before));
    }
}

mod request_context {
    use super::*;

    fn tagged_results(results: Vec<anyhow::Result<TaggedOutput>>) -> Vec<Option<(u32, Vec<f32>)>> {
        results
            .into_iter()
            .map(|r| r.ok().map(|o| (o.tag, o.values)))
            .collect()
    }

    #[tokio::test]
    async fn contexts_stay_with_their_requests_when_others_fail_preprocessing() {
        let model = ModelFixture::doubling();
        let server = start_server_with_output::<TaggedOutput>(&model, single_batch_config(5)).await;

        // Failures at the front and middle shift every later request's position
        // in the batch tensor.
        let behaviors = [
            Behavior::FailPreprocess,
            Behavior::Succeed,
            Behavior::PanicPreprocess,
            Behavior::Succeed,
            Behavior::Succeed,
        ];
        let results = join_all(behaviors.iter().enumerate().map(|(i, &behavior)| {
            server.infer(
                TestInput::seq(i as f32 * 10.0)
                    .with_tag(100 + i as u32)
                    .with_behavior(behavior),
            )
        }))
        .await;

        let expected: Vec<_> = behaviors
            .iter()
            .enumerate()
            .map(|(i, behavior)| {
                matches!(behavior, Behavior::Succeed)
                    .then(|| (100 + i as u32, doubled_seq(i as f32 * 10.0).0))
            })
            .collect();
        assert_eq!(tagged_results(results), expected);
    }

    #[tokio::test]
    async fn eager_preprocessing_delivers_each_requests_context() {
        let model = ModelFixture::doubling();
        let config = single_batch_config(3).with_preprocess_on_infer(true);
        let server = start_server_with_output::<TaggedOutput>(&model, config).await;

        let results =
            join_all((0..3).map(|i| server.infer(TestInput::seq(i as f32).with_tag(7 * (i + 1)))))
                .await;

        let expected: Vec<_> = (0..3)
            .map(|i| Some((7 * (i + 1), doubled_seq(i as f32).0)))
            .collect();
        assert_eq!(tagged_results(results), expected);
    }
}
