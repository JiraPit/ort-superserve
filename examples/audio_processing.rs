//! Audio preprocessing example showing how to process WAV files for ONNX models.
//!
//! This demonstrates:
//! - Loading WAV files (simulating server upload) in constructor
//! - Converting stereo to mono in preprocess step
//! - Resampling to target sample rate in preprocess step
//! - Normalizing to fixed length in preprocess step
//! - Wrapping all CPU-bound work in `spawn_blocking`

use anyhow::Result;
use ndarray::Array1;
use ort_superserve::{Input, Output, Outputs, Server, ServerConfig, create_env};
use std::path::PathBuf;

/// Raw audio samples loaded from WAV file.
/// In a real server, this simulates receiving raw audio bytes from an upload.
struct AudioInput {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

impl AudioInput {
    /// Load audio from a WAV file.
    /// In a real server, this simulates receiving raw audio bytes from an upload.
    fn from_wav(path: PathBuf) -> Result<Self> {
        let reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap() as f32 / max_val)
                    .collect()
            }
        };

        Ok(AudioInput {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }
}

impl Input for AudioInput {
    type Preprocessed = Array1<f32>;
    type Context = ();

    async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> {
        tokio::task::spawn_blocking(move || {
            let target_sample_rate = 16000;
            let target_length = 16000;

            // Convert stereo to mono
            let mono_samples = if self.channels == 2 {
                self.samples
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|[left, right]| (left + right) / 2.0)
                    .collect()
            } else {
                self.samples
            };

            // Resample to target rate
            let resampled = if self.sample_rate != target_sample_rate {
                resample_linear(&mono_samples, self.sample_rate, target_sample_rate)
            } else {
                mono_samples
            };

            // Normalize to fixed length
            let normalized = normalize(&resampled, target_length);

            Ok((Array1::from_vec(normalized), ()))
        })
        .await?
    }
}

/// Linear interpolation resampling.
fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio) as usize;

    (0..output_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let left = pos.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = pos - left as f64;

            if left < samples.len() {
                samples[left] * (1.0 - frac as f32) + samples[right] * frac as f32
            } else {
                0.0
            }
        })
        .collect()
}

/// Normalize to fixed length with zero-padding or truncation.
fn normalize(samples: &[f32], target_length: usize) -> Vec<f32> {
    let mut result = vec![0.0; target_length];
    let copy_len = samples.len().min(target_length);
    result[..copy_len].copy_from_slice(&samples[..copy_len]);
    result
}

#[derive(Debug)]
struct AudioOutput {
    embedding: Vec<f32>,
}

impl Output for AudioOutput {
    async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
        let raw = outputs.first::<f32>()?;
        Ok(AudioOutput {
            embedding: raw.iter().cloned().collect(),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: audio_processing <model.onnx> <audio.wav>");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let audio_path = PathBuf::from(&args[2]);

    // Create the ONNX Runtime environment explicitly. ONNX Runtime allows one per
    // process, so applications with several servers should share (clone) it.
    let env = create_env()?;

    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_threads_per_session(4)
        .with_max_batch_size(8)
        .with_min_batch_size(1)
        .with_environment(env);

    let server = Server::<AudioInput, AudioOutput>::from_file(model_path, config).await?;

    println!("Server initialized successfully!");
    println!("Loading audio from {:?}...", audio_path);

    // Load raw audio (simulating server upload receiving bytes)
    let input = AudioInput::from_wav(audio_path)?;

    println!("Running inference...");

    let result = server.infer(input).await?;

    println!("Audio embedding: {} values", result.embedding.len());
    println!(
        "First 10 values: {:?}",
        &result.embedding[..10.min(result.embedding.len())]
    );

    server.shutdown();

    Ok(())
}
