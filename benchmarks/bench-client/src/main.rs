use anyhow::Result;
use clap::Parser;
use csv::Writer;
use shared::ImageInput;
use shared::ImageOutput;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tokio::sync::mpsc;
use tokio::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "bench-client")]
#[command(about = "Benchmark client for ONNX session management comparison")]
struct Args {
    /// Server to benchmark (ort-superserve, actix-with-batching, actix-without-batching, arc-mutex, batched-fn)
    #[arg(short, long)]
    server: String,

    /// Server port
    #[arg(short, long, default_value = "3001")]
    port: u16,

    /// Server host (default: localhost)
    #[arg(long, default_value = "localhost")]
    host: String,

    /// Output CSV file
    #[arg(short, long, default_value = "results/default.csv")]
    output: PathBuf,

    /// Data directory containing MNIST images
    #[arg(long, default_value = "data/images")]
    data_dir: PathBuf,

    /// Ramp duration in seconds (concurrency goes from 1 to max)
    #[arg(long, default_value = "60")]
    ramp_duration: u64,

    /// Hold duration in seconds (concurrency stays at max)
    #[arg(long, default_value = "30")]
    hold_duration: u64,

    /// Maximum concurrency
    #[arg(long, default_value = "2048")]
    max_concurrency: usize,

    /// Number of warmup requests before benchmark
    #[arg(long, default_value = "10")]
    warmup_requests: usize,

    /// Sampling interval in milliseconds
    #[arg(long, default_value = "500")]
    sample_interval_ms: u64,
}

#[derive(Clone)]
struct Metrics {
    latency_us: Arc<AtomicU64>,
    request_count: Arc<AtomicU64>,
    active_workers: Arc<AtomicUsize>,
}

struct Sample {
    timestamp_sec: f64,
    concurrency: usize,
    latency_p50_ms: f64,
    latency_p90_ms: f64,
    latency_p99_ms: f64,
    throughput_rps: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Server: {}:{}", args.host, args.port);
    println!("Ramp duration: {}s", args.ramp_duration);

    let base_url = format!("http://{}:{}", args.host, args.port);
    let health_url = format!("{}/health", base_url);
    let infer_url = format!("{}/infer", base_url);

    println!("Waiting for server...");
    for i in 0..30 {
        match reqwest::get(&health_url).await {
            Ok(resp) if resp.status().is_success() => break,
            _ => {
                if i == 29 {
                    anyhow::bail!("Server not ready after 30 seconds");
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }

    let all_images: Arc<Vec<Vec<u8>>> = Arc::new(load_all_images(&args.data_dir)?);
    if all_images.is_empty() {
        anyhow::bail!("No images found in {}", args.data_dir.display());
    }
    println!("Loaded {} images", all_images.len());

    // Warmup
    for image in all_images.iter().take(args.warmup_requests) {
        let input = ImageInput::from_png_bytes(image.clone());
        let client = reqwest::Client::new();
        let _ = client.post(&infer_url).json(&input).send().await?;
    }

    let metrics = Metrics {
        latency_us: Arc::new(AtomicU64::new(0)),
        request_count: Arc::new(AtomicU64::new(0)),
        active_workers: Arc::new(AtomicUsize::new(0)),
    };

    let metrics_for_sampler = metrics.clone();
    let metrics_for_summary = metrics.clone();

    let (latency_tx, mut latency_rx) = mpsc::unbounded_channel::<(f64, u64)>();

    let benchmark_start = Instant::now();
    let sample_interval = Duration::from_millis(args.sample_interval_ms);
    let ramp_duration = args.ramp_duration;
    let max_concurrency = args.max_concurrency;

    let sampler_handle = tokio::spawn(async move {
        let mut samples: Vec<Sample> = Vec::new();
        let mut interval_latencies: Vec<(f64, u64)> = Vec::new();
        let mut last_request_count: u64 = 0;
        let mut last_sample_time = benchmark_start;

        while let Some((ts, lat)) = latency_rx.recv().await {
            interval_latencies.push((ts, lat));

            let now = Instant::now();
            if now.duration_since(last_sample_time) >= sample_interval {
                let timestamp_sec = benchmark_start.elapsed().as_secs_f64();
                let ramp_end = ramp_duration as f64;
                let concurrency = if timestamp_sec < ramp_end {
                    ((timestamp_sec / ramp_end) * max_concurrency as f64).ceil() as usize
                } else {
                    max_concurrency
                };

                interval_latencies.sort_by_key(|(_, l)| *l);
                let count = interval_latencies.len();
                if count > 0 {
                    let p50_idx = ((count as f64) * 0.50).min(count as f64 - 1.0) as usize;
                    let p90_idx = ((count as f64) * 0.90).min(count as f64 - 1.0) as usize;
                    let p99_idx = ((count as f64) * 0.99).min(count as f64 - 1.0) as usize;

                    let current_requests =
                        metrics_for_sampler.request_count.load(Ordering::Relaxed);
                    let elapsed = last_sample_time.elapsed().as_secs_f64();
                    let throughput = (current_requests - last_request_count) as f64 / elapsed;

                    samples.push(Sample {
                        timestamp_sec,
                        concurrency,
                        latency_p50_ms: interval_latencies[p50_idx].1 as f64 / 1000.0,
                        latency_p90_ms: interval_latencies[p90_idx].1 as f64 / 1000.0,
                        latency_p99_ms: interval_latencies[p99_idx].1 as f64 / 1000.0,
                        throughput_rps: throughput,
                    });

                    last_request_count = current_requests;
                }

                interval_latencies.clear();
                last_sample_time = now;
            }
        }

        samples
    });

    println!("Running benchmark...");
    let start_time = Instant::now();

    let mut worker_handles = Vec::new();

    for worker_id in 0..args.max_concurrency {
        let client = reqwest::Client::new();
        let url = infer_url.clone();
        let metrics = metrics.clone();
        let latency_tx = latency_tx.clone();
        let images = Arc::clone(&all_images);
        let ramp_duration = args.ramp_duration;
        let hold_duration = args.hold_duration;
        let max_concurrency = args.max_concurrency;

        let handle = tokio::spawn(async move {
            let mut image_index = worker_id % images.len();

            loop {
                let elapsed = start_time.elapsed().as_secs_f64();

                let worker_start_time =
                    (worker_id as f64 / max_concurrency as f64) * ramp_duration as f64;
                if elapsed < worker_start_time {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }

                if elapsed > (ramp_duration + hold_duration) as f64 {
                    break;
                }

                metrics.active_workers.fetch_add(1, Ordering::Relaxed);

                let image = &images[image_index % images.len()];
                let input = ImageInput::from_png_bytes(image.clone());

                let request_start = Instant::now();
                match client.post(&url).json(&input).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        match resp.json::<ImageOutput>().await {
                            Ok(output) => {
                                println!("class_id: {}, confidence: {:.4}", output.class_id, output.confidence);
                            }
                            Err(e) => {
                                eprintln!("Failed to parse response: {}", e);
                            }
                        }
                        let latency = request_start.elapsed().as_micros() as u64;
                        let ts = start_time.elapsed().as_secs_f64();
                        metrics.latency_us.fetch_add(latency, Ordering::Relaxed);
                        metrics.request_count.fetch_add(1, Ordering::Relaxed);
                        let _ = latency_tx.send((ts, latency));
                    }
                    _ => {}
                }

                metrics.active_workers.fetch_sub(1, Ordering::Relaxed);
                image_index += 1;
            }
        });

        worker_handles.push(handle);
    }

    for handle in worker_handles {
        let _ = handle.await;
    }

    drop(latency_tx);

    let samples = sampler_handle.await.unwrap_or_default();

    let elapsed = start_time.elapsed().as_secs_f64();
    let total_requests = metrics_for_summary.request_count.load(Ordering::Relaxed);
    let throughput = total_requests as f64 / elapsed;

    let output_dir = args.output.parent().unwrap();
    std::fs::create_dir_all(output_dir)?;

    let mut writer = Writer::from_path(&args.output)?;
    writer.write_record([
        "server",
        "timestamp_sec",
        "concurrency",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p99_ms",
        "throughput_rps",
    ])?;

    for sample in &samples {
        writer.write_record([
            &args.server,
            &format!("{:.2}", sample.timestamp_sec),
            &sample.concurrency.to_string(),
            &format!("{:.2}", sample.latency_p50_ms),
            &format!("{:.2}", sample.latency_p90_ms),
            &format!("{:.2}", sample.latency_p99_ms),
            &format!("{:.2}", sample.throughput_rps),
        ])?;
    }

    writer.flush()?;

    println!(
        "Done! {} samples, {} total requests, {:.2} req/s",
        samples.len(),
        total_requests,
        throughput
    );

    Ok(())
}

fn load_all_images(data_dir: &PathBuf) -> Result<Vec<Vec<u8>>> {
    use std::fs;

    let mut files: Vec<_> = fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "png")
                .unwrap_or(false)
        })
        .collect();

    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    files.shuffle(&mut rng);

    let images: Vec<Vec<u8>> = files
        .iter()
        .filter_map(|entry| fs::read(entry.path()).ok())
        .collect();

    Ok(images)
}
