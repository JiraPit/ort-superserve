//! Image preprocessing example showing real image processing with the `image` crate.
//!
//! This demonstrates how to preprocess actual images for ONNX models, including:
//! - Loading images from disk
//! - Resizing to model input dimensions
//! - Converting to RGB
//! - Normalizing pixel values to f32
//! - Wrapping CPU-bound work in `spawn_blocking`

use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::{Array3, ArrayD, ArrayViewD, Axis};
use ort_superserve::{Input, Output, Server, ServerConfig};
use std::path::PathBuf;

/// Input type that holds a dynamic image.
struct ImageInput {
    image: DynamicImage,
}

impl ImageInput {
    fn from_path(path: PathBuf) -> Result<Self> {
        let image = image::open(path)?;
        Ok(Self { image })
    }
}

/// Preprocessed image data ready for batching.
struct PreprocessedImage {
    data: Array3<f32>,
}

impl Input for ImageInput {
    type Preprocessed = PreprocessedImage;

    async fn preprocess(self) -> Result<Self::Preprocessed> {
        tokio::task::spawn_blocking(move || {
            let target_height = 224;
            let target_width = 224;

            let resized = self.image.resize_exact(
                target_width,
                target_height,
                image::imageops::FilterType::Lanczos3,
            );

            let rgb: ImageBuffer<Rgb<u8>, Vec<u8>> = resized.to_rgb8();

            let (width, height) = rgb.dimensions();
            let mut data = Array3::<f32>::zeros((3, height as usize, width as usize));

            for (x, y, pixel) in rgb.enumerate_pixels() {
                for c in 0..3 {
                    data[[c, y as usize, x as usize]] = pixel[c] as f32 / 255.0;
                }
            }

            Ok(PreprocessedImage { data })
        })
        .await?
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        let views: Vec<_> = items.iter().map(|item| item.data.view()).collect();
        let batched = ndarray::stack(Axis(0), &views)?;
        Ok(batched.into_dyn())
    }
}

#[derive(Debug)]
struct ClassificationOutput {
    class_id: usize,
    confidence: f32,
}

impl Output for ClassificationOutput {
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
        let shape = raw.shape();
        let batch_size = shape[0];

        let mut scores = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let slice = raw.index_axis(Axis(0), i);
            scores.push(slice[[0]]);
        }

        let (class_id, &confidence) = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((0, &0.0));

        Ok(ClassificationOutput {
            class_id,
            confidence,
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: image_processing <model.onnx> <image.jpg>");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let image_path = PathBuf::from(&args[2]);

    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_threads_per_session(4)
        .with_max_batch_size(8)
        .with_min_batch_size(1);

    let server = Server::<ImageInput, ClassificationOutput>::from_file(model_path, config).await?;

    println!("Server initialized successfully!");
    println!("Loading image from {:?}...", image_path);

    let input = ImageInput::from_path(image_path)?;

    println!("Running inference...");

    let result = server.infer(input).await?;

    println!(
        "Classification: class_id={}, confidence={:.4}",
        result.class_id, result.confidence
    );

    server.shutdown();

    Ok(())
}
