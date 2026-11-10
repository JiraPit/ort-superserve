//! End-to-end test against a real model (MobileNetV2).
//!
//! Requires downloaded test assets; see `panic_with_download_instructions`.

use anyhow::Result;
use futures::future::join_all;
use ndarray::{Array3, ArrayD, ArrayViewD};
use ort_superserve::{
    Environment, Input, Output, Server, ServerConfig, create_env, helpers::batch_array,
};
use std::path::{Path, PathBuf};
use std::time::Duration;

const ASSETS_DIR: &str = "test_assets/mobilenetv2-12-int8";

fn model_path() -> PathBuf {
    Path::new(ASSETS_DIR).join("mobilenetv2-12-int8.onnx")
}

fn image_path(index: usize) -> PathBuf {
    Path::new(ASSETS_DIR)
        .join("images")
        .join(format!("{index}.png"))
}

fn panic_with_download_instructions() -> ! {
    panic!(
        "Test model not found. Please download it first.\n\
         \n\
         From the project root, run:\n\
         uv run --project download_test_assets download_test_assets/download_test_assets.py --model mobilenet\n\
         \n\
         This will download the MobileNetV2 model and test images to {ASSETS_DIR}/"
    )
}

struct ImageInput {
    png_bytes: Vec<u8>,
}

impl Input for ImageInput {
    type Preprocessed = Array3<f32>;
    type Context = ();

    async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> {
        let decoder = png::Decoder::new(std::io::Cursor::new(self.png_bytes));
        let mut reader = decoder.read_info()?;
        let mut buf = vec![0; reader.output_buffer_size()];
        reader.next_frame(&mut buf)?;

        let info = reader.info();
        let (width, height) = (info.width as usize, info.height as usize);
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        let tensor = Array3::from_shape_fn((3, height, width), |(c, y, x)| {
            let value = buf[(y * width + x) * 3 + c] as f32 / 255.0;
            (value - mean[c]) / std[c]
        });
        Ok((tensor, ()))
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        batch_array(&items)
    }
}

struct Logits(Vec<f32>);

impl Output for Logits {
    async fn postprocess(raw: ArrayViewD<'_, f32>, _context: ()) -> Result<Self> {
        Ok(Logits(raw.iter().copied().collect()))
    }
}

async fn start(config: ServerConfig, env: &Environment) -> Server<ImageInput, Logits> {
    Server::from_file(model_path(), config.with_environment(env.clone()))
        .await
        .expect("Failed to create server")
}

fn load_image(index: usize) -> ImageInput {
    let png_bytes = std::fs::read(image_path(index)).expect("Failed to read PNG image");
    ImageInput { png_bytes }
}

#[tokio::test]
async fn batched_inference_matches_single_item_inference() {
    if !model_path().exists() || !image_path(0).exists() {
        panic_with_download_instructions();
    }
    const IMAGES: usize = 4;
    let env = create_env().unwrap();

    let single = start(ServerConfig::new().with_max_batch_size(1), &env).await;
    let mut expected = Vec::new();
    for i in 0..IMAGES {
        expected.push(single.infer(load_image(i)).await.unwrap().0);
    }

    let batched = start(
        ServerConfig::new()
            .with_max_batch_size(IMAGES)
            .with_max_wait_time(Duration::from_secs(2)),
        &env,
    )
    .await;
    let actual = join_all((0..IMAGES).map(|i| batched.infer(load_image(i)))).await;

    for (i, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let actual = actual.unwrap().0;
        assert_eq!(actual.len(), 1000, "image {i}: expected ImageNet logits");
        let max_diff = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "image {i}: logits differ by {max_diff}");
    }
}
