"""imgsim — find duplicate and near-duplicate images.

Quick start::

    import imgsim

    index = imgsim.ImageIndex()
    index.add_dir("photos/", recursive=True)
    result = index.compare(threshold=10)

    for group in result.groups:
        print(len(group), "similar images")
        for path in group:
            print("   ", path)

Why it exists: perceptual hashing alone produces two systematic false positives —
color-blind matches (a red apple and a green apple of the same shape share a
fingerprint) and collapsed fingerprints on flat images (a white rectangle and a
blue-white stripe pattern both hash to zero). imgsim adds a hue-aware color veto
and a low-information guard on top of the standard pHash + dHash pair.

Two algorithms are available:

``mode="hash"`` (default)
    DCT perceptual hash + difference hash, 64 bits each, compared by Hamming
    distance. No ML dependencies, ~3 s for 1000 images.

``mode="embed"``
    ResNet18 embeddings compared by cosine similarity. Recognises the same subject
    from different angles, at the cost of a PyTorch dependency.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._types import Group, Pair, Result
from .index import ImageIndex
from .report import render_html, save_html

# Read the version from the installed distribution rather than hardcoding it.
# A literal here is a second source of truth that silently drifts: 0.1.1 shipped
# with the new code but __version__ still reporting 0.1.0.
try:
    __version__ = _distribution_version("imgsim-py")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "Group",
    "ImageIndex",
    "Pair",
    "Result",
    "__version__",
    "render_html",
    "save_html",
]
