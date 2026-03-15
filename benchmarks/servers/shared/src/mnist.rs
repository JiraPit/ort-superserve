use anyhow::{Context, Result};
use std::path::Path;

pub fn load_random_image(images_dir: &Path) -> Result<Vec<u8>> {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let idx: usize = rng.gen_range(0..10000);
    let path = images_dir.join(format!("{}.png", idx));

    std::fs::read(&path).with_context(|| format!("Failed to read image: {:?}", path))
}
