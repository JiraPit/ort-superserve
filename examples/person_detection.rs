use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array3, ArrayD, ArrayViewD, Axis};
use ort_superserve::{Input, Output, Server, ServerConfig};
use std::path::Path;

pub struct PersonDetectionInput {
    image: DynamicImage,
    target_size: (u32, u32),
}

impl PersonDetectionInput {
    pub fn new(image: DynamicImage, target_size: (u32, u32)) -> Self {
        Self { image, target_size }
    }

    pub fn from_path<P: AsRef<Path>>(path: P, target_size: (u32, u32)) -> Result<Self> {
        let image = image::open(path)?;
        Ok(Self::new(image, target_size))
    }
}

impl Input for PersonDetectionInput {
    type Preprocessed = (Array3<f32>, (u32, u32));

    async fn preprocess(self) -> Result<Self::Preprocessed> {
        tokio::task::spawn_blocking(move || {
            let (target_w, target_h) = self.target_size;
            let (orig_w, orig_h) = self.image.dimensions();

            let resized =
                self.image
                    .resize_exact(target_w, target_h, image::imageops::FilterType::Triangle);
            let rgb = resized.to_rgb8();

            let mut array = Array3::<f32>::zeros((3, target_h as usize, target_w as usize));

            for (y, row) in rgb.rows().enumerate() {
                for (x, pixel) in row.enumerate() {
                    array[[0, y, x]] = pixel[0] as f32 / 255.0;
                    array[[1, y, x]] = pixel[1] as f32 / 255.0;
                    array[[2, y, x]] = pixel[2] as f32 / 255.0;
                }
            }

            Ok((array, (orig_w, orig_h)))
        })
        .await?
    }

    fn batch(items: Vec<Self::Preprocessed>) -> Result<ArrayD<f32>> {
        let tensors: Vec<_> = items.iter().map(|(t, _)| t.view()).collect();
        let batched = ndarray::stack(Axis(0), &tensors)?;
        Ok(batched.into_dyn())
    }
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub confidence: f32,
    pub label: String,
}

#[derive(Debug, Clone)]
pub struct PersonDetectionOutput {
    pub detections: Vec<BoundingBox>,
    pub original_dims: (u32, u32),
}

pub struct PersonDetectionBatcher;

impl Output for PersonDetectionOutput {
    async fn postprocess(raw: ArrayViewD<'_, f32>) -> Result<Self> {
        let shape = raw.shape().to_vec();
        let num_anchors = shape[1];

        let owned = raw.to_owned();

        tokio::task::spawn_blocking(move || {
            let raw_2d = owned.into_shape_with_order((4 + 1, num_anchors))?;

            let mut detections = Vec::new();

            let conf_threshold = 0.5;

            for i in 0..num_anchors {
                let score = raw_2d[[4, i]];

                if score >= conf_threshold {
                    let cx = raw_2d[[0, i]];
                    let cy = raw_2d[[1, i]];
                    let w = raw_2d[[2, i]];
                    let h = raw_2d[[3, i]];

                    detections.push(BoundingBox {
                        x1: cx - w / 2.0,
                        y1: cy - h / 2.0,
                        x2: cx + w / 2.0,
                        y2: cy + h / 2.0,
                        confidence: score,
                        label: "person".to_string(),
                    });
                }
            }

            let detections = nms(detections, 0.45);

            Ok(PersonDetectionOutput {
                detections,
                original_dims: (640, 640),
            })
        })
        .await?
    }
}

fn nms(mut detections: Vec<BoundingBox>, iou_threshold: f32) -> Vec<BoundingBox> {
    if detections.is_empty() {
        return Vec::new();
    }

    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut kept = Vec::new();
    while !detections.is_empty() {
        let current = detections.remove(0);
        kept.push(current.clone());

        detections.retain(|other| compute_iou(&current, other) <= iou_threshold);
    }
    kept
}

fn compute_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let w = (x2 - x1).max(0.0);
    let h = (y2 - y1).max(0.0);
    let inter = w * h;

    let area1 = (box1.x2 - box1.x1).max(0.0) * (box1.y2 - box1.y1).max(0.0);
    let area2 = (box2.x2 - box2.x1).max(0.0) * (box2.y2 - box2.y1).max(0.0);

    let union = area1 + area2 - inter + 1e-6;

    if union <= 0.0 { 0.0 } else { inter / union }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let config = ServerConfig::new()
        .with_num_sessions(2)
        .with_threads_per_session(4)
        .with_max_batch_size(8)
        .with_min_batch_size(1);

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: person_detection <model.onnx>");
    let image_path = std::env::args().nth(2);

    let server =
        Server::<PersonDetectionInput, PersonDetectionOutput>::from_file(&model_path, config)
            .await?;

    println!("Server initialized successfully!");

    if let Some(img_path) = image_path {
        println!("Processing image: {}", img_path);

        let input = PersonDetectionInput::from_path(&img_path, (640, 640))?;
        let result = server.infer(input).await?;

        println!("Found {} detections:", result.detections.len());
        for (i, det) in result.detections.iter().enumerate() {
            println!(
                "  {}: [{:.1}, {:.1}, {:.1}, {:.1}] confidence={:.3}",
                i, det.x1, det.y1, det.x2, det.y2, det.confidence
            );
        }
    } else {
        println!("Running batch inference demo...");

        let mut handles = Vec::new();
        for _ in 0..4 {
            let dummy_image =
                DynamicImage::ImageRgb8(ImageBuffer::from_pixel(640, 640, Rgb([0u8, 0, 0])));
            let input = PersonDetectionInput::new(dummy_image, (640, 640));
            let server_clone = server.clone();

            handles.push(tokio::spawn(async move { server_clone.infer(input).await }));
        }

        let results = futures::future::try_join_all(handles).await?;

        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(output) => println!("Request {}: {} detections", i, output.detections.len()),
                Err(e) => println!("Request {} failed: {}", i, e),
            }
        }
    }

    Ok(())
}
