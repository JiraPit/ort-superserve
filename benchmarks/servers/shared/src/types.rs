//! Shared types for ResNet50 image classification across benchmark servers.

use anyhow::Result;
use image::{ImageBuffer, Rgb};
use ndarray::{Array1, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

/// Input payload containing a base64-encoded PNG image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
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
pub struct ImageOutput {
    /// Predicted class index (0-999 for ImageNet).
    pub digit: usize,
    /// Confidence score derived from softmax probabilities.
    pub confidence: f32,
}

impl ImageInput {
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
    /// Note: ImageNet normalization (mean/std) should be applied by the server.
    pub fn to_input_array(&self) -> Result<ArrayD<f32>> {
        let rgb = self.decode()?;
        let (width, height) = rgb.dimensions();

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

/// Applies ImageNet normalization to a tensor in [C, H, W] format.
/// Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
pub fn apply_imagenet_normalization(tensor: ArrayD<f32>) -> ArrayD<f32> {
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let mut result = tensor.into_owned();
    for c in 0..3 {
        for i in 0..224 {
            for j in 0..224 {
                let val = result[[c, i, j]];
                result[[c, i, j]] = (val - mean[c]) / std[c];
            }
        }
    }
    result
}

impl ImageOutput {
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
