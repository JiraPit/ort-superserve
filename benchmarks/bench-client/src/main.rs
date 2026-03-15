use anyhow::Result;
use clap::Parser;
use csv::Writer;
use shared::MnistInput;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
}

#[derive(Clone)]
struct Metrics {
    latency_us: Arc<AtomicU64>,
    request_count: Arc<AtomicU64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Server: {}", args.server);
    println!("Port: {}", args.port);
    println!("Output: {:?}", args.output);
    println!("Ramp duration: {}s", args.ramp_duration);
    println!("Hold duration: {}s", args.hold_duration);
    println!("Max concurrency: {}", args.max_concurrency);

    let base_url = format!("http://localhost:{}", args.port);
    let health_url = format!("{}/health", base_url);
    let infer_url = format!("{}/infer", base_url);

    // Wait for server to be ready
    println!("Waiting for server to be ready...");
    for i in 0..30 {
        match reqwest::get(&health_url).await {
            Ok(resp) if resp.status().is_success() => {
                println!("Server is ready!");
                break;
            }
            _ => {
                if i == 29 {
                    anyhow::bail!("Server not ready after 30 seconds");
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }

    // Warmup
    println!("Running warmup...");
    let warmup_images: Vec<Vec<u8>> = load_random_images(&args.data_dir, args.warmup_requests)?;
    for image in warmup_images {
        let input = MnistInput::from_png_bytes(image);
        let client = reqwest::Client::new();
        let _ = client.post(&infer_url).json(&input).send().await?;
    }
    println!("Warmup complete");

    // Metrics
    let metrics = Metrics {
        latency_us: Arc::new(AtomicU64::new(0)),
        request_count: Arc::new(AtomicU64::new(0)),
    };

    // Latency tracker for percentiles
    let (latency_tx, mut latency_rx) = mpsc::unbounded_channel::<u64>();

    // Spawn metrics collector
    let collector_handle = tokio::spawn(async move {
        let mut latencies: Vec<u64> = Vec::new();
        while let Some(lat) = latency_rx.recv().await {
            latencies.push(lat);
        }
        latencies
    });

    // Load images
    let total_requests_estimate = args.max_concurrency * 100; // Rough estimate
    let all_images: Vec<Vec<u8>> = load_random_images(&args.data_dir, total_requests_estimate)?;

    // Start benchmark
    println!("Starting benchmark...");
    let start_time = Instant::now();

    // Spawn workers
    let mut worker_handles = Vec::new();

    for worker_id in 0..args.max_concurrency {
        let client = reqwest::Client::new();
        let url = infer_url.clone();
        let metrics = metrics.clone();
        let latency_tx = latency_tx.clone();
        let images: Vec<Vec<u8>> = all_images
            .iter()
            .skip(worker_id % all_images.len())
            .cloned()
            .collect();
        let ramp_duration = args.ramp_duration;
        let hold_duration = args.hold_duration;
        let max_concurrency = args.max_concurrency;

        let handle = tokio::spawn(async move {
            let mut local_image_index = 0;
            let start_time = Instant::now();

            loop {
                let elapsed = start_time.elapsed().as_secs_f64();

                // Check if we should start this worker
                let worker_start_time =
                    (worker_id as f64 / max_concurrency as f64) * ramp_duration as f64;
                if elapsed < worker_start_time {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }

                // Check if benchmark is complete
                if elapsed > (ramp_duration + hold_duration) as f64 {
                    break;
                }

                // Send request
                let image = &images[local_image_index % images.len()];
                let input = MnistInput::from_png_bytes(image.clone());

                let request_start = Instant::now();
                match client.post(&url).json(&input).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        let latency = request_start.elapsed().as_micros() as u64;
                        metrics.latency_us.fetch_add(latency, Ordering::Relaxed);
                        metrics.request_count.fetch_add(1, Ordering::Relaxed);
                        let _ = latency_tx.send(latency);
                    }
                    _ => {}
                }

                local_image_index += 1;
            }
        });

        worker_handles.push(handle);
    }

    // Wait for all workers to complete
    for handle in worker_handles {
        let _ = handle.await;
    }

    // Drop the sender to close the channel
    drop(latency_tx);

    // Collect latencies
    let latencies = collector_handle.await.unwrap_or_default();

    // Calculate metrics
    let elapsed = start_time.elapsed().as_secs_f64();
    let total_requests = metrics.request_count.load(Ordering::Relaxed);
    let throughput = total_requests as f64 / elapsed;

    // Calculate percentiles
    let mut sorted_latencies: Vec<u64> = latencies.clone();
    sorted_latencies.sort();

    let p50 = percentile(&sorted_latencies, 50);
    let p90 = percentile(&sorted_latencies, 90);
    let p99 = percentile(&sorted_latencies, 99);

    println!("Benchmark complete!");
    println!("Total requests: {}", total_requests);
    println!("Duration: {:.2}s", elapsed);
    println!("Throughput: {:.2} req/s", throughput);
    println!("Latency p50: {:.2}ms", p50 as f64 / 1000.0);
    println!("Latency p90: {:.2}ms", p90 as f64 / 1000.0);
    println!("Latency p99: {:.2}ms", p99 as f64 / 1000.0);

    // Write CSV
    let output_dir = args.output.parent().unwrap();
    std::fs::create_dir_all(output_dir)?;

    let mut writer = Writer::from_path(&args.output)?;
    writer.write_record([
        "server",
        "total_requests",
        "duration_sec",
        "throughput_rps",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p99_ms",
    ])?;
    writer.write_record([
        &args.server,
        &total_requests.to_string(),
        &format!("{:.2}", elapsed),
        &format!("{:.2}", throughput),
        &format!("{:.2}", p50 as f64 / 1000.0),
        &format!("{:.2}", p90 as f64 / 1000.0),
        &format!("{:.2}", p99 as f64 / 1000.0),
    ])?;
    writer.flush()?;

    println!("Results written to {:?}", args.output);

    Ok(())
}

fn percentile(sorted: &[u64], p: u64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (sorted.len() as f64 * p as f64 / 100.0).min(sorted.len() as f64 - 1.0) as usize;
    sorted[idx]
}

fn load_random_images(data_dir: &PathBuf, count: usize) -> Result<Vec<Vec<u8>>> {
    use std::fs;

    let mut images = Vec::new();
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

    for entry in files.iter().take(count) {
        let bytes = fs::read(entry.path())?;
        images.push(bytes);
    }

    Ok(images)
}
