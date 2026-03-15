//! Utility functions for MNIST benchmark data loading.

use anyhow::{Context, Result};
use std::path::Path;

/// Loads a random image from the MNIST test images directory.
///
/// Images are expected to be named `{index}.png` where index is in range 0..10000.
pub fn load_random_image(images_dir: &Path) -> Result<Vec<u8>> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let idx: usize = rng.gen_range(0..10000);
    let path = images_dir.join(format!("{}.png", idx));

    std::fs::read(&path).with_context(|| format!("Failed to read image: {:?}", path))
}
