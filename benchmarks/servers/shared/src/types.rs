//! Shared types for MobileNetV2 image classification across benchmark servers.

use anyhow::Result;
use image::{ImageBuffer, Rgb};
use ndarray::{Array1, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

/// Input payload containing a base64-encoded PNG image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnistInput {
    /// Raw PNG image bytes, serialized as base64 in JSON.
    #[serde(with = "base64")]
    pub image_bytes: Vec<u8>,
}

/// Base64 serialization module for image bytes.
mod base64 {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(data: &Vec<u8>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = STANDARD.encode(data);
        encoded.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        STANDARD.decode(&encoded).map_err(serde::de::Error::custom)
    }
}

/// Output payload containing predicted class and confidence score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnistOutput {
    /// Predicted class index (0-999 for ImageNet).
    pub digit: usize,
    /// Confidence score derived from softmax probabilities.
    pub confidence: f32,
}

impl MnistInput {
    /// Creates an input from raw PNG bytes.
    pub fn from_png_bytes(bytes: Vec<u8>) -> Self {
        Self { image_bytes: bytes }
    }

    /// Loads an input from a PNG file on disk.
    pub fn from_png_file(path: &std::path::Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Ok(Self { image_bytes: bytes })
    }

    /// Decodes the PNG bytes into an RGB image buffer resized to 224x224.
    pub fn decode(&self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let img = image::load_from_memory(&self.image_bytes)?;
        let rgb = img.to_rgb8();

        let resized =
            image::imageops::resize(&rgb, 224, 224, image::imageops::FilterType::Triangle);

        Ok(resized)
    }

    /// Converts the image into an ONNX-compatible input tensor.
    ///
    /// Returns a tensor with shape [3, 224, 224] normalized to [0, 1].
    /// When batched, this becomes [batch, 3, 224, 224].
    pub fn to_input_array(&self) -> Result<ArrayD<f32>> {
        let rgb = self.decode()?;
        let (width, height) = rgb.dimensions();

        // MobileNetV2 expects NCHW format: [C, H, W]
        // Channel-first layout: RRR...GGG...BBB
        let mut data = Vec::with_capacity((width * height * 3) as usize);

        for channel in 0..3 {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    data.push(pixel.0[channel] as f32 / 255.0);
                }
            }
        }

        let arr = Array1::from_vec(data);
        let arr = arr.into_shape_with_order(IxDyn(&[3, height as usize, width as usize]))?;
        Ok(arr)
    }
}

impl MnistOutput {
    /// Converts raw logits to prediction output using softmax.
    ///
    /// Computes the softmax probability of the maximum logit
    /// as the confidence score.
    pub fn from_logits(logits: &[f32]) -> Self {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let max_val = logits[max_idx];
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();
        let confidence = 1.0 / exp_sum;

        Self {
            digit: max_idx,
            confidence,
        }
    }

    /// Creates output from pre-computed softmax probabilities.
    pub fn from_softmax_probs(probs: &[f32]) -> Self {
        let max_idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Self {
            digit: max_idx,
            confidence: probs[max_idx],
        }
    }
}
