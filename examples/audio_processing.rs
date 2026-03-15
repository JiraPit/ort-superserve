//! Audio preprocessing example showing how to process WAV files for ONNX models.
//!
//! This demonstrates:
//! - Loading WAV files using the `hound` crate
//! - Converting stereo to mono
//! - Resampling to target sample rate (using linear interpolation for simplicity)
//! - Normalizing audio samples
//! - Wrapping CPU-bound work in `spawn_blocking`

use anyhow::Result;
use ndarray::{Array1, ArrayD, ArrayViewD, Axis};
use ort_superserve::{Input, Output, Server, ServerConfig};
use std::path::PathBuf;

/// Input type that holds audio samples loaded from a WAV file.
struct AudioInput {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl AudioInput {
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

        // Convert stereo to mono by averaging channels
        let mono_samples = if spec.channels == 2 {
            samples
                .chunks_exact(2)
                .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                .collect()
        } else {
            samples
        };

        Ok(AudioInput {
            samples: mono_samples,
            sample_rate: spec.sample_rate,
        })
    }
}

/// Preprocessed audio ready for batching.
struct PreprocessedAudio {
    data: Array1<f32>,
}

impl Input for AudioInput {
    type Preprocessed = PreprocessedAudio;

    async fn preprocess(self) -> Result<Self::Preprocessed> {
        tokio::task::spawn_blocking(move || {
            let target_sample_rate = 16000;
            let target_length = 16000;

            let resampled = if self.sample_rate != target_sample_rate {
                resample_linear(&self.samples, self.sample_rate, target_sample_rate)
            } else {
                self.samples
            };

            let normalized = normalize(&resampled, target_length);

            Ok(PreprocessedAudio {
                data: Array1::from_vec(normalized),
            })
        })
        .await?
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        let views: Vec<_> = items.iter().map(|a| a.data.view()).collect();
        let batched = ndarray::stack(Axis(0), &views)?;
        Ok(batched.into_dyn())
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
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
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

    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_threads_per_session(4)
        .with_max_batch_size(8)
        .with_min_batch_size(1);

    let server = Server::<AudioInput, AudioOutput>::from_file(model_path, config).await?;

    println!("Server initialized successfully!");
    println!("Loading audio from {:?}...", audio_path);

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
