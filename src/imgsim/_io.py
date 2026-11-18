"""Image loading and preprocessing.

Every image is decoded exactly once, producing both the grayscale thumbnail used
for hashing and the small RGB grid used for color verification.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# 超大图（如全景照）默认会被 Pillow 拦截，这里解除限制，由调用方自行控制内存
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".avif", ".heic"}
)

_WHITE = (255, 255, 255, 255)


def iter_images(root: Path, recursive: bool = False) -> list[Path]:
    """列出目录下的图片文件，按路径排序以保证结果可复现。"""
    it = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=lambda p: str(p).lower())
    return files


def load_features(path: Path, size: int, color_grid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode *path* once and return ``(gray, color_signature)``.

    ``gray`` is a ``size x size`` float64 array, ``color_signature`` is a
    ``color_grid * color_grid * 3`` float32 array with values in ``[0, 1]``.

    Returns ``None`` when the file cannot be decoded (corrupt, not an image,
    unreadable permission). Callers are expected to collect these paths rather
    than abort the whole batch.
    """
    try:
        with Image.open(path) as im:
            # JPEG only: let libjpeg downsample during decoding. A 4000x3000 photo
            # decodes 5-10x faster because only 1/8 of the pixels are ever produced.
            # No-op for other formats, and Pillow never scales below the requested size.
            # Draft is a hint, not a guarantee — a failure here is not fatal.
            with contextlib.suppress(Exception):
                im.draft("RGB", (max(size, color_grid), max(size, color_grid)))

            im = ImageOps.exif_transpose(im)  # honour EXIF orientation before anything else

            if im.mode in ("RGBA", "LA", "P"):
                # Composite onto white: a transparent PNG would otherwise become
                # solid black in grayscale and hash as if it were a dark photo.
                im = im.convert("RGBA")
                canvas = Image.new("RGBA", im.size, _WHITE)
                canvas.alpha_composite(im)
                im = canvas
            im = im.convert("RGB")

            # Region-average resampling is exactly the block mean we want for color signatures
            signature = np.asarray(
                im.resize((color_grid, color_grid), Image.BOX), dtype=np.float32
            ).ravel() / np.float32(255.0)

            gray_im = im.convert("L")
            if gray_im.size != (size, size):
                gray_im = gray_im.resize((size, size), Image.LANCZOS)
            gray = np.asarray(gray_im, dtype=np.float64)

        return gray, signature
    except Exception:  # - a broken file must not kill a 100k-image batch
        return None


def load_thumbnail(path: Path, max_side: int) -> Image.Image | None:
    """Load an RGB thumbnail for report rendering, preserving aspect ratio."""
    try:
        with Image.open(path) as im:
            im.draft("RGB", (max_side, max_side))
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            return im.copy()
    except Exception:
        return None
