//! Detector with several outputs of different element types and built-in NMS.
//!
//! Many exported detectors (e.g. with an `EfficientNMS`/`NonMaxSuppression`
//! head) return a variable number of detections for the whole batch:
//!
//! - `boxes`:       `[num_dets, 4]` f32
//! - `scores`:      `[num_dets]`    f32
//! - `labels`:      `[num_dets]`    i64
//! - `batch_index`: `[num_dets]`    i64, the image each detection belongs to
//!
//! Axis 0 of these outputs is not the batch axis, so each output is split by
//! `batch_index` instead. The same works for models exported with a fixed
//! batch size: partial batches are padded, and detections of padded images are
//! dropped before they reach any caller.
//!
//! Adjust the output names to your model.

use anyhow::Result;
use ndarray::Array3;
use ort_superserve::{Input, Output, Outputs, Server, ServerConfig, Split, create_env};

struct ImageInput {
    data: Array3<f32>,
}

impl Input for ImageInput {
    type Preprocessed = Array3<f32>;
    type Context = ();

    async fn preprocess(self) -> Result<(Self::Preprocessed, Self::Context)> {
        Ok((self.data, ()))
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct Detection {
    bbox: [f32; 4],
    score: f32,
    label: i64,
}

#[derive(Debug)]
struct Detections(#[allow(dead_code)] Vec<Detection>);

impl Output for Detections {
    async fn postprocess(outputs: Outputs, _context: ()) -> Result<Self> {
        // Only this image's detections, in model order.
        let boxes = outputs.get::<f32>("boxes")?;
        let scores = outputs.get::<f32>("scores")?;
        let labels = outputs.get::<i64>("labels")?;

        let detections = boxes
            .outer_iter()
            .zip(scores.iter())
            .zip(labels.iter())
            .map(|((b, &score), &label)| Detection {
                bbox: [b[0], b[1], b[2], b[3]],
                score,
                label,
            })
            .collect();
        Ok(Detections(detections))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: nms_detection <model.onnx>");

    let by_image = || Split::batch_index_from("batch_index", None);
    let config = ServerConfig::new()
        .with_max_batch_size(8)
        .with_output_split("boxes", by_image())
        .with_output_split("scores", by_image())
        .with_output_split("labels", by_image())
        .with_environment(create_env()?);

    let server = Server::<ImageInput, Detections>::from_file(&model_path, config).await?;

    let requests = (0..4).map(|_| {
        server.infer(ImageInput {
            data: Array3::zeros((3, 640, 640)),
        })
    });
    for (i, result) in futures::future::join_all(requests)
        .await
        .into_iter()
        .enumerate()
    {
        println!("Image {i}: {:?}", result?);
    }

    server.shutdown();
    Ok(())
}
