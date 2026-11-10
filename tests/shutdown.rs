//! Tests that shutting down the last server handle stops the batcher and workers.
//!
//! Their only externally observable signal is the shutdown log lines, so this
//! test installs a process-wide log capture. It lives in its own test binary so
//! no other server in the process can emit matching lines.

mod common;

use common::*;
use ort_superserve::ServerConfig;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Default)]
struct CapturedLogs(Arc<Mutex<Vec<u8>>>);

impl CapturedLogs {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
}

impl Write for CapturedLogs {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn shutting_down_the_last_handle_stops_batcher_and_workers() {
    let logs = CapturedLogs::default();
    let writer = logs.clone();
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .init();

    let model = ModelFixture::doubling();
    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_max_wait_time(Duration::from_millis(5));
    let server = start_server(&model, config).await;
    assert_eq!(
        server.infer(TestInput::seq(1.0)).await.unwrap(),
        doubled_seq(1.0)
    );

    server.shutdown();

    let expected = [
        "Batcher: Shutting down.",
        "Worker 0: Channel closed, shutting down.",
        "Worker 1: Channel closed, shutting down.",
    ];
    let deadline = Instant::now() + Duration::from_secs(10);
    while !expected.iter().all(|line| logs.text().contains(line)) {
        assert!(
            Instant::now() < deadline,
            "batcher and workers did not stop; captured logs:\n{}",
            logs.text()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
