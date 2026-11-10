//! Tests for servers sharing one application-created ONNX Runtime environment.

mod common;

use common::*;
use futures::future::join_all;
use ort_superserve::ServerConfig;
use std::time::Duration;

fn config(num_sessions: usize) -> ServerConfig {
    ServerConfig::new()
        .with_num_sessions(num_sessions)
        .with_max_batch_size(2)
        .with_max_wait_time(Duration::from_millis(5))
}

#[tokio::test]
async fn multiple_servers_run_side_by_side_on_one_environment() {
    let model = ModelFixture::doubling();
    let first = start_server(&model, config(1)).await;
    let second = start_server(&model, config(1)).await;

    let (a, b) = tokio::join!(
        first.infer(TestInput::seq(1.0)),
        second.infer(TestInput::seq(2.0)),
    );

    assert_eq!(a.unwrap(), doubled_seq(1.0));
    assert_eq!(b.unwrap(), doubled_seq(2.0));
}

#[tokio::test]
async fn server_can_be_dropped_and_recreated_with_the_same_environment() {
    let model = ModelFixture::doubling();

    let original = start_server(&model, config(1)).await;
    assert_eq!(
        original.infer(TestInput::seq(1.0)).await.unwrap(),
        doubled_seq(1.0)
    );
    original.shutdown();

    let recreated = start_server(&model, config(1)).await;
    assert_eq!(
        recreated.infer(TestInput::seq(2.0)).await.unwrap(),
        doubled_seq(2.0)
    );
}

#[tokio::test]
async fn servers_with_multiple_ort_sessions_share_one_environment() {
    let model = ModelFixture::doubling();
    let servers = [
        start_server(&model, config(3)).await,
        start_server(&model, config(3)).await,
    ];

    let requests = servers.iter().enumerate().flat_map(|(server_idx, server)| {
        (0..12).map(move |i| {
            let start = (server_idx * 100 + i) as f32;
            async move { (start, server.infer(TestInput::seq(start)).await) }
        })
    });

    for (start, result) in join_all(requests).await {
        assert_eq!(result.unwrap(), doubled_seq(start));
    }
}
