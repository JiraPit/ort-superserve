#!/usr/bin/env python3
"""Download model and test data for testing and benchmarking."""

import argparse
import zlib
import urllib.request
import tarfile
from pathlib import Path

MODELS = {
    "mobilenet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12-int8.tar.gz",
        "filename": "mobilenetv2-12-int8.onnx",
    },
    "resnet50": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12-int8.tar.gz",
        "filename": "resnet50-v1-12-int8.onnx",
    },
}


def download_file(url: str, dest: Path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def create_sample_images(images_dir: Path, count: int = 10):
    """Create sample PNG images for testing (random noise as placeholder)."""
    import random

    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {count} sample images...")

    for idx in range(count):
        width, height = 224, 224
        pixels = bytes([random.randint(0, 255) for _ in range(width * height * 3)])

        png = create_png_rgb(width, height, pixels)

        filepath = images_dir / f"{idx}.png"
        with open(filepath, "wb") as f:
            f.write(png)

        if (idx + 1) % 10 == 0:
            print(f"  Created {idx + 1}/{count} images")

    print(f"Created all {count} images in {images_dir}")


def create_png_rgb(width: int, height: int, pixels: bytes) -> bytes:
    """Create an RGB PNG image from raw pixel data."""

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = zlib.crc32(chunk) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk + crc.to_bytes(4, "big")

    signature = b"\x89PNG\r\n\x1a\n"

    ihdr_data = (
        width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    )
    ihdr = png_chunk(b"IHDR", ihdr_data)

    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"
        for x in range(width):
            offset = (y * width + x) * 3
            raw_data += pixels[offset : offset + 3]

    compressed = zlib.compress(raw_data)
    idat = png_chunk(b"IDAT", compressed)

    iend = png_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX model and generate test images"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["mobilenet", "resnet50"],
        help="Model to download (required: mobilenet or resnet50)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to store model and images (default: test_assets/{model_filename_root})",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=10,
        help="Number of sample images to generate (default: 10)",
    )
    args = parser.parse_args()

    model_info = MODELS[args.model]
    model_filename = model_info["filename"]
    model_dir_name = model_filename.replace(".onnx", "")

    if args.data_dir is None:
        data_dir = Path("test_assets") / model_dir_name
    else:
        data_dir = args.data_dir

    images_dir = data_dir / "images"
    model_path = data_dir / model_filename
    tar_path = data_dir / f"{args.model}.tar.gz"

    data_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        if not tar_path.exists():
            download_file(model_info["url"], tar_path)

        print("Extracting model...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".onnx"):
                    member.name = model_filename
                    tar.extract(member, data_dir)
        print(f"Model extracted to {model_path}")

        tar_path.unlink()
    else:
        print(f"Model already exists: {model_path}")

    if (
        not images_dir.exists()
        or len(list(images_dir.glob("*.png"))) < args.image_count
    ):
        create_sample_images(images_dir, count=args.image_count)
    else:
        print(
            f"Images already exist: {images_dir} ({len(list(images_dir.glob('*.png')))} files)"
        )

    print("\nDone!")
    print(f"Model: {model_path}")
    print(f"Images: {images_dir}")


if __name__ == "__main__":
    main()
