#!/usr/bin/env python3
"""Download MNIST model and test images, convert to PNG format."""

import os
import urllib.request
import gzip
import struct
from pathlib import Path

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
TEST_IMAGES_URL = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"
)
TEST_LABELS_URL = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"
)


def download_file(url: str, dest: Path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def read_mnist_images(filepath: Path):
    """Read MNIST image file and return list of (image, label) tuples."""
    with gzip.open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST images file: magic={magic}")

        images = []
        for _ in range(num_images):
            img = f.read(rows * cols)
            images.append((rows, cols, img))
        return images


def read_mnist_labels(filepath: Path):
    """Read MNIST label file and return list of labels."""
    with gzip.open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST labels file: magic={magic}")
        return list(f.read(num_labels))


def save_as_png(images_dir: Path, images_data: list, labels: list):
    """Save MNIST images as PNG files."""
    import array

    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(images_data)} images as PNG...")

    for idx, ((rows, cols, img_data), label) in enumerate(zip(images_data, labels)):
        img_array = array.array("B", img_data)

        png = create_png(rows, cols, img_array)

        filepath = images_dir / f"{idx}.png"
        with open(filepath, "wb") as f:
            f.write(png)

        if (idx + 1) % 1000 == 0:
            print(f"  Saved {idx + 1}/{len(images_data)} images")

    print(f"Saved all {len(images_data)} images to {images_dir}")


def create_png(width: int, height: int, pixels):
    """Create a grayscale PNG image from raw pixel data."""
    import zlib

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk + crc.to_bytes(4, "big")

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    ihdr_data = (
        width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08"  # bit depth = 8
        b"\x00"  # color type = grayscale
        b"\x00"  # compression = deflate
        b"\x00"  # filter = adaptive
        b"\x00"  # interlace = none
    )
    ihdr = png_chunk(b"IHDR", ihdr_data)

    # IDAT chunk (image data)
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"  # filter type: none
        for x in range(width):
            raw_data += bytes([pixels[y * width + x]])

    compressed = zlib.compress(raw_data)
    idat = png_chunk(b"IDAT", compressed)

    # IEND chunk
    iend = png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


def main():
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    images_dir = data_dir / "images"

    model_path = data_dir / "mnist-12.onnx"
    images_gz = data_dir / "t10k-images-idx3-ubyte.gz"
    labels_gz = data_dir / "t10k-labels-idx1-ubyte.gz"

    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download model
    if not model_path.exists():
        download_file(MODEL_URL, model_path)
    else:
        print(f"Model already exists: {model_path}")

    # Download test images and labels
    if not images_gz.exists():
        download_file(TEST_IMAGES_URL, images_gz)
    else:
        print(f"Images already downloaded: {images_gz}")

    if not labels_gz.exists():
        download_file(TEST_LABELS_URL, labels_gz)
    else:
        print(f"Labels already downloaded: {labels_gz}")

    # Convert to PNG
    if not images_dir.exists() or len(list(images_dir.glob("*.png"))) < 1000:
        print("Converting MNIST to PNG...")
        images = read_mnist_images(images_gz)
        labels = read_mnist_labels(labels_gz)
        save_as_png(images_dir, images, labels)
    else:
        print(f"Images already converted: {images_dir}")

    print("Done!")
    print(f"Model: {model_path}")
    print(f"Images: {images_dir} ({len(list(images_dir.glob('*.png')))} PNG files)")


if __name__ == "__main__":
    main()
