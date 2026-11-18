"""Perceptual hashing: DCT-based pHash and gradient-based dHash.

Both produce a ``hash_size ** 2`` bit fingerprint (64 bits by default). Two images
are compared by the Hamming distance between their fingerprints — the number of
differing bits.

Why two hashes: pHash captures overall low-frequency structure and survives
resizing, JPEG recompression and mild color grading. dHash captures horizontal
gradients and is far more sensitive to real structural edits. Averaging the two
distances suppresses most false positives that either hash alone would produce.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# numpy >= 2.0 ships a hardware-backed popcount. Older versions fall back to
# unpacking bits, which is ~20x slower but functionally identical.
_HAS_BITCOUNT = hasattr(np, "bitwise_count")


@lru_cache(maxsize=8)
def dct_matrix(n: int) -> np.ndarray:
    """Orthogonal DCT-II transform matrix, so that ``C @ x @ C.T`` is a 2D DCT.

    Row index is frequency, column index is position::

        C[freq, pos] = sqrt(2/n) * cos(pi * (2*pos + 1) * freq / (2n))

    with the ``freq == 0`` row scaled by ``1/sqrt(2)``. Getting that orientation
    backwards still produces a plausible-looking hash but destroys orthogonality and
    moves the DC coefficient out of ``[0, 0]``, which breaks the AC-energy guard.

    Computed once per size and cached — building it costs O(n^2) and it is reused
    for every image in a batch.
    """
    k = np.arange(n, dtype=np.float64)
    freq = k[:, None]
    pos = k[None, :]
    matrix = np.cos(np.pi * (2.0 * pos + 1.0) * freq / (2.0 * n))
    matrix[0, :] *= 1.0 / np.sqrt(2.0)  # DC row
    return matrix * np.sqrt(2.0 / n)


def phash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> tuple[np.ndarray, float]:
    """Compute a DCT perceptual hash.

    Args:
        gray: ``size x size`` grayscale image, where ``size = hash_size * highfreq_factor``.
        hash_size: Side length of the retained low-frequency block. 8 gives 64 bits.
        highfreq_factor: How much detail to keep before the DCT. 4 is the standard choice.

    Returns:
        ``(bits, ac_ratio)`` where ``bits`` is a flat ``hash_size ** 2`` uint8 array of
        0/1 and ``ac_ratio`` is the energy of the AC (detail) coefficients relative to
        the DC (average brightness) coefficient.

        ``ac_ratio`` exists because pHash silently degrades on images with no low-frequency
        structure — solid colors, line art, regular stripes. Their AC coefficients are all
        near zero, the median comparison collapses, and every such image hashes to the same
        value. Callers should treat a ratio below ~0.02 as "this fingerprint is unreliable".
    """
    size = hash_size * highfreq_factor
    if gray.shape != (size, size):
        from PIL import Image

        gray = np.asarray(
            Image.fromarray(gray.astype(np.uint8)).resize((size, size), Image.LANCZOS),
            dtype=np.float64,
        )

    c = dct_matrix(size)
    coefficients = c @ gray @ c.T
    low = coefficients[:hash_size, :hash_size]

    dc = abs(low[0, 0]) + 1e-6
    ac_ratio = float((np.abs(low).sum() - abs(low[0, 0])) / dc)
    bits = (low > np.median(low)).ravel().astype(np.uint8)
    return bits, ac_ratio


def dhash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute a difference hash from horizontally adjacent pixel gradients.

    Resistant to brightness and contrast shifts because it only compares neighbours.
    """
    from PIL import Image

    im = Image.fromarray(gray.astype(np.uint8)).resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = np.asarray(im, dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).ravel().astype(np.uint8)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack an ``(n, k)`` array of 0/1 into ``(n,)`` uint64 codes.

    Packing enables ``a ^ b`` followed by a single popcount instead of comparing
    bit by bit.
    """
    bits = np.ascontiguousarray(bits.astype(np.uint8))
    packed = np.packbits(bits, axis=1)
    return np.ascontiguousarray(packed).view(">u8").ravel().astype(np.uint64)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two sets of packed codes.

    Returns an ``(len(a), len(b))`` uint8 matrix.
    """
    x = a[:, None] ^ b[None, :]
    if _HAS_BITCOUNT:
        return np.bitwise_count(x).astype(np.uint8)
    # Fallback for numpy < 2.0: expand bytes back to bits and sum
    x = np.ascontiguousarray(x).view(np.uint8).reshape(x.shape[0], x.shape[1], 8)
    return np.unpackbits(x, axis=2).sum(axis=2, dtype=np.uint16).astype(np.uint8)


def hamming_matrix(codes: np.ndarray, chunk: int = 1024, progress: object = None) -> np.ndarray:
    """Full ``(n, n)`` distance matrix, computed in row chunks to bound memory.

    A 100k-image corpus would need a 10 GB dense matrix; chunking keeps peak usage
    at ``chunk * n`` bytes and lets callers handle very large sets by streaming.
    """
    n = codes.shape[0]
    out = np.zeros((n, n), dtype=np.uint8)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        out[start:stop] = hamming(codes[start:stop], codes)
        if progress is not None:
            progress(stop, n)
    return out
