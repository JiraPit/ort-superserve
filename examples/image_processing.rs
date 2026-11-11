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
use ndarray::Array3;
use ort_superserve::{Input, Output, Outputs, Server, ServerConfig, create_env};
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

impl Input for ImageInput {
    type Preprocessed = Array3<f32>;
    type Context = ();

    async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> {
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

            Ok((data, ()))
        })
        .await?
    }
}

#[derive(Debug)]
struct ClassificationOutput {
    class_id: usize,
    confidence: f32,
}

impl Output for ClassificationOutput {
    async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
        let raw = outputs.first::<f32>()?;
        // `raw` holds the class scores for this one image (batch axis already removed).
        let (class_id, &confidence) = raw
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

    // Create the ONNX Runtime environment explicitly. ONNX Runtime allows one per
    // process, so applications with several servers should share (clone) it.
    let env = create_env()?;

    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_threads_per_session(4)
        .with_max_batch_size(8)
        .with_min_batch_size(1)
        .with_environment(env);

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
