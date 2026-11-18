"""Shared fixtures: a small synthetic corpus generated on the fly.

Tests must not depend on checked-in binaries — they would bloat the repo and make
results non-reproducible. Instead every test builds its images from a fixed seed.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance

SIDE = 256


def make_image(seed: int, palette: tuple[int, int, int], shape: str = "mixed") -> Image.Image:
    """A textured 'pseudo photo': gradient base + noise + random shapes."""
    rng = np.random.default_rng(seed)
    picker = random.Random(seed)
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    channels = np.stack(
        [
            (xx / SIDE) * palette[0] * 0.7 + rng.normal(0, 10, (SIDE, SIDE)),
            (yy / SIDE) * palette[1] * 0.7 + rng.normal(0, 10, (SIDE, SIDE)),
            ((xx + yy) / (2 * SIDE)) * palette[2] * 0.7 + rng.normal(0, 10, (SIDE, SIDE)),
        ],
        axis=-1,
    )
    image = Image.fromarray(np.clip(channels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(18):
        x, y = picker.randint(0, SIDE - 40), picker.randint(0, SIDE - 40)
        size = picker.randint(24, 110)
        color = tuple(picker.randint(10, 245) for _ in range(3))
        if shape == "circle":
            draw.ellipse([x, y, x + size, y + size], fill=color)
        elif shape == "rect":
            draw.rectangle([x, y, x + size, y + size], fill=color)
        elif shape == "poly":
            draw.polygon([(x, y), (x + size, y + size // 2), (x + size // 3, y + size)],
                         fill=color)
        else:
            choice = picker.random()
            if choice < 0.4:
                draw.rectangle([x, y, x + size, y + size], fill=color)
            elif choice < 0.75:
                draw.ellipse([x, y, x + size, y + size], fill=color)
            else:
                draw.polygon([(x, y), (x + size, y + size // 2), (x + size // 3, y + size)],
                             fill=color)
    return image


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> dict[str, object]:
    """A directory with three variant families plus four unrelated images."""
    root: Path = tmp_path_factory.mktemp("corpus")
    a = make_image(1, (200, 90, 60), "circle")
    b = make_image(7, (70, 130, 200), "rect")
    c = make_image(13, (90, 180, 120), "poly")

    family_a = [
        a,
        a.resize((128, 128), Image.LANCZOS),
        a.resize((300, 300), Image.BICUBIC),
        ImageEnhance.Brightness(a).enhance(1.12),
    ]
    names_a = ["a0.jpg", "a1.jpg", "a2.jpg", "a3.jpg"]
    for image, name in zip(family_a, names_a):
        image.convert("RGB").save(root / name, quality=88)

    b.save(root / "b0.jpg", quality=92)
    b.resize((180, 180), Image.LANCZOS).save(root / "b1.png")
    b.convert("L").convert("RGB").save(root / "b2.jpg", quality=85)  # grayscale variant
    names_b = ["b0.jpg", "b1.png", "b2.jpg"]

    c.save(root / "c0.jpg", quality=92)
    c.crop((10, 10, SIDE - 10, SIDE - 10)).save(root / "c1.jpg", quality=88)
    names_c = ["c0.jpg", "c1.jpg"]

    Image.new("RGB", (SIDE, SIDE), (255, 255, 255)).save(root / "white.jpg")
    Image.new("RGB", (SIDE, SIDE), (0, 0, 0)).save(root / "black.jpg")

    stripes = Image.new("RGB", (SIDE, SIDE))
    pen = ImageDraw.Draw(stripes)
    for x in range(0, SIDE, 16):
        pen.rectangle([x, 0, x + 8, SIDE], fill=(30, 60, 200))
    stripes.save(root / "stripes.png")

    make_image(101, (240, 200, 40), "rect").save(root / "x0.jpg", quality=88)
    make_image(202, (30, 30, 40), "circle").save(root / "x1.jpg", quality=88)

    (root / "broken.jpg").write_bytes(b"this is not an image")

    return {
        "dir": root,
        "group_a": [root / n for n in names_a],
        "group_b": [root / n for n in names_b],
        "group_c": [root / n for n in names_c],
        "singles": [root / "white.jpg", root / "black.jpg", root / "stripes.png",
                    root / "x0.jpg", root / "x1.jpg"],
        "broken": root / "broken.jpg",
    }


@pytest.fixture
def index(corpus):
    """An ImageIndex over the synthetic corpus (function scoped, cheap to rebuild)."""
    import imgsim

    instance = imgsim.ImageIndex()
    instance.add_dir(corpus["dir"])
    return instance
