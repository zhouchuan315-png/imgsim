"""Color verification: the part that stops grayscale hashes from lying."""

from __future__ import annotations

import numpy as np

from imgsim._io import load_features
from imgsim.color import color_distance, rgb_to_hsl


def _sig(corpus, name):
    entry = next(p for p in corpus["dir"].iterdir() if p.name == name)
    features = load_features(entry, 32, 8)
    assert features is not None
    return features[1]


def _distance(corpus, a, b):
    left = np.array([_sig(corpus, a)], dtype=np.float32)
    right = np.array([_sig(corpus, b)], dtype=np.float32)
    return float(color_distance(left, right)[0, 0])


def test_rgb_to_hsl_shape_and_ranges():
    rgb = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)  # 2 cells
    hsl = rgb_to_hsl(rgb)
    assert hsl.shape == (1, 2, 3)
    assert hsl[..., 1].max() <= 1.0 and hsl[..., 1].min() >= 0.0
    assert hsl[..., 2].max() <= 1.0 and hsl[..., 2].min() >= 0.0


def test_lightness_matches_grayscale_conversion():
    """Pure red has HSL lightness 0.299 in both RGB and grayscale form.

    Using HSV value instead would give 1.0 vs 0.299 and break the grayscale-variant
    case below.
    """
    red = rgb_to_hsl(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    gray = rgb_to_hsl(np.array([[0.299, 0.299, 0.299]], dtype=np.float32))
    assert abs(float(red[0, 0, 2]) - float(gray[0, 0, 2])) < 1e-3


def test_identical_images_have_zero_distance(corpus):
    assert _distance(corpus, "a0.jpg", "a0.jpg") < 1e-6


def test_grayscale_variant_is_not_vetoed(corpus):
    """Regression: converting to grayscale must not make an image look 'different'.

    Without the saturation-weighted hue term this distance jumps above 0.3 because
    the gray signature is nowhere near the colorful one in RGB space.
    """
    assert _distance(corpus, "b0.jpg", "b2.jpg") < 0.05


def test_brightness_shift_is_tolerated(corpus):
    assert _distance(corpus, "a0.jpg", "a3.jpg") < 0.12


def test_flat_white_and_stripes_are_separated(corpus):
    """Regression: both hash to an all-zero fingerprint, color must tell them apart."""
    assert _distance(corpus, "white.jpg", "stripes.png") > 0.2


def test_unrelated_images_are_separated(corpus):
    assert _distance(corpus, "a0.jpg", "b0.jpg") > 0.2


def test_distance_matrix_is_symmetric(corpus):
    names = ["a0.jpg", "b0.jpg", "white.jpg", "x0.jpg"]
    signatures = np.array([_sig(corpus, n) for n in names], dtype=np.float32)
    matrix = color_distance(signatures, signatures)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-5)


def test_chunking_does_not_change_result(corpus):
    signatures = np.array(
        [_sig(corpus, n) for n in ["a0.jpg", "b0.jpg", "white.jpg", "x0.jpg"]],
        dtype=np.float32,
    )
    assert np.allclose(
        color_distance(signatures, signatures, chunk=1),
        color_distance(signatures, signatures, chunk=128),
        atol=1e-6,
    )
