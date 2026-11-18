"""Fingerprint correctness and stability."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from imgsim._io import load_features
from imgsim.fingerprint import (
    dct_matrix,
    dhash,
    hamming,
    hamming_matrix,
    pack_bits,
    phash,
)


@pytest.fixture
def gray_of(corpus):
    def _load(path, size=32):
        features = load_features(path, size, 8)
        assert features is not None
        return features[0]

    return _load


def test_dct_matrix_is_orthonormal():
    """C @ C.T == I is the property that makes C @ x @ C.T a proper 2D DCT."""
    c = dct_matrix(16)
    assert np.allclose(c @ c.T, np.eye(16), atol=1e-12)


def test_dct_matrix_is_cached():
    assert dct_matrix(8) is dct_matrix(8)


def test_phash_bit_count(gray_of, corpus):
    bits, _ = phash(gray_of(corpus["group_a"][0]))
    assert bits.shape == (64,)
    assert bits.dtype == np.uint8
    assert set(np.unique(bits)) <= {0, 1}


def test_phash_survives_resize(gray_of, corpus):
    """Resizing 256 -> 128 -> 300 must not change the fingerprint much."""
    reference = pack_bits(np.array([phash(gray_of(corpus["group_a"][0]))[0]]))
    for variant in corpus["group_a"][1:3]:
        other = pack_bits(np.array([phash(gray_of(variant))[0]]))
        assert hamming(reference, other)[0, 0] <= 6


def test_phash_survives_brightness_shift(gray_of, corpus):
    reference = pack_bits(np.array([phash(gray_of(corpus["group_a"][0]))[0]]))
    brighter = pack_bits(np.array([phash(gray_of(corpus["group_a"][3]))[0]]))
    assert hamming(reference, brighter)[0, 0] <= 6


def test_phash_separates_unrelated_images(gray_of, corpus):
    a = pack_bits(np.array([phash(gray_of(corpus["group_a"][0]))[0]]))
    x = pack_bits(np.array([phash(gray_of(corpus["singles"][3]))[0]]))
    assert hamming(a, x)[0, 0] > 12


def test_ac_ratio_flags_flat_images(corpus):
    """Solid colors must be reported as low-information so callers can distrust the hash.

    This is the guard against the classic pHash failure: a white rectangle and a
    black rectangle both produce an all-zero fingerprint.
    """
    white = load_features(corpus["singles"][0], 32, 8)
    assert white is not None
    _, ac_ratio = phash(white[0])
    assert ac_ratio < 0.02


def test_ac_ratio_high_for_photos(gray_of, corpus):
    _, ac_ratio = phash(gray_of(corpus["group_a"][0]))
    assert ac_ratio > 0.02


def test_dhash_is_a_gradient_hash(gray_of, corpus):
    bits = dhash(gray_of(corpus["group_a"][0]))
    assert bits.shape == (64,)
    assert set(np.unique(bits)) <= {0, 1}


def test_pack_bits_roundtrip_preserves_bit_order():
    """Regression: save/load reads codes back through a big-endian view.

    If the packing order and the unpacking order ever disagree, every fingerprint
    silently reverses and cached indexes stop matching freshly indexed images.
    """
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, (5, 64)).astype(np.uint8)
    codes = pack_bits(bits)
    assert codes.dtype == np.uint64
    raw = np.frombuffer(codes.astype(">u8").tobytes(), dtype=np.uint8).reshape(5, 8)
    assert np.array_equal(np.unpackbits(raw, axis=1), bits)


def test_hamming_known_values():
    a = np.array([0b1010], dtype=np.uint64)
    b = np.array([0b0011], dtype=np.uint64)
    assert hamming(a, b)[0, 0] == 2
    assert hamming(a, a)[0, 0] == 0


def test_hamming_matrix_shape_and_symmetry(gray_of, corpus):
    codes = pack_bits(np.array([phash(gray_of(p))[0] for p in corpus["group_a"]]))
    matrix = hamming_matrix(codes)
    assert matrix.shape == (4, 4)
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(4))


def test_hamming_matrix_chunking_is_equivalent(gray_of, corpus):
    codes = pack_bits(np.array([phash(gray_of(p))[0] for p in corpus["group_a"]]))
    assert np.array_equal(hamming_matrix(codes, chunk=1), hamming_matrix(codes, chunk=1024))


def test_load_features_returns_none_for_corrupt_file(corpus):
    assert load_features(corpus["broken"], 32, 8) is None


def test_load_features_composites_transparency_on_white(tmp_path):
    """A transparent PNG must read as white, not black."""
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    path = tmp_path / "transparent.png"
    image.save(path)
    features = load_features(path, 32, 8)
    assert features is not None
    gray, _ = features
    assert gray.mean() > 250
