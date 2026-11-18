"""Color verification on top of perceptual hashing.

Perceptual hashes are computed on grayscale pixels, which means a red apple and a
green apple of the same shape produce nearly identical fingerprints. This module
adds a cheap color signature that vetoes such matches.

The signature is a small RGB grid (8x8 by default) taken by region averaging. The
distance between two signatures is computed in HSL space rather than RGB, for one
specific reason: converting an image to grayscale destroys its saturation but not
its perceived lightness. If we compared RGB directly, "photo vs its black-and-white
version" would look as different as "red apple vs green apple" — both are large
color changes. In HSL, the hue and saturation terms are weighted by the *smaller*
saturation of the two images, so they automatically vanish for grayscale images
and only lightness is compared. Hue differences between two saturated images
still register at full strength.
"""

from __future__ import annotations

import numpy as np

# Weights were calibrated on the bundled test corpus (see tests/test_color.py):
# same-image variants score 0.002-0.06, unrelated images score 0.39-0.90.
HUE_WEIGHT = 0.55
SATURATION_WEIGHT = 0.35
LIGHTNESS_WEIGHT = 0.90

# Fraction of grid cells considered when scoring. Using the most-different cells
# instead of the mean keeps localized changes visible (recoloring one object
# affects only a few cells and would be averaged away).
DEFAULT_TOP_K = 12


def rgb_to_hsl(signature: np.ndarray) -> np.ndarray:
    """Convert ``(n, cells * 3)`` RGB in ``[0, 1]`` to ``(n, cells, 3)`` HSL.

    Lightness is the perceptual ``0.299 R + 0.587 G + 0.114 B``, not the HSV value
    (``max(r, g, b)``). This matters: pure red has value 1.0 but lightness 0.299,
    and its grayscale conversion also has lightness 0.299 — so the two agree, while
    using HSV value they would differ by 0.7.
    """
    px = np.ascontiguousarray(signature, dtype=np.float32).reshape(signature.shape[0], -1, 3)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    hi, lo = px.max(-1), px.min(-1)
    delta = hi - lo

    saturation = np.where(hi > 1e-6, delta / np.maximum(hi, 1e-6), 0.0)

    hue = np.zeros_like(hi)
    chromatic = delta > 1e-6
    idx = chromatic & (hi == r)
    hue[idx] = ((g - b)[idx] / delta[idx]) % 6.0
    idx = chromatic & (hi == g)
    hue[idx] = ((b - r)[idx] / delta[idx]) + 2.0
    idx = chromatic & (hi == b)
    hue[idx] = ((r - g)[idx] / delta[idx]) + 4.0
    hue /= 6.0

    lightness = 0.299 * r + 0.587 * g + 0.114 * b
    return np.stack([hue, saturation, lightness], axis=-1).astype(np.float32)


def color_distance(
    a: np.ndarray,
    b: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    chunk: int = 128,
) -> np.ndarray:
    """Color distance matrix between two sets of signatures.

    Args:
        a: ``(m, cells * 3)`` signatures.
        b: ``(n, cells * 3)`` signatures.
        top_k: How many of the most-different cells to average over.
        chunk: Row chunk size, bounding peak memory to ``chunk * n * cells`` floats.

    Returns:
        ``(m, n)`` float32 matrix. Values below ~0.18 indicate the two images share
        a color layout; above that they are probably different subjects.
    """
    n = b.shape[0]
    a_hsl = rgb_to_hsl(a)
    b_hsl = rgb_to_hsl(b)
    out = np.empty((a_hsl.shape[0], n), dtype=np.float32)

    for start in range(0, a_hsl.shape[0], chunk):
        stop = min(start + chunk, a_hsl.shape[0])
        left = a_hsl[start:stop, None, :, :]
        right = b_hsl[None, :, :, :]

        hue_delta = np.abs(left[..., 0] - right[..., 0])
        hue_delta = np.minimum(hue_delta, 1.0 - hue_delta) * 2.0  # hue is circular
        sat_delta = np.abs(left[..., 1] - right[..., 1])
        light_delta = np.abs(left[..., 2] - right[..., 2])
        sat_floor = np.minimum(left[..., 1], right[..., 1])

        total = (
            HUE_WEIGHT * hue_delta * sat_floor
            + SATURATION_WEIGHT * sat_delta * sat_floor
            + LIGHTNESS_WEIGHT * light_delta
        )
        k = min(top_k, total.shape[2])
        out[start:stop] = np.partition(total, -k, axis=2)[:, :, -k:].mean(axis=2)

    return out
