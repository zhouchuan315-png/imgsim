"""Optional semantic embedding backend.

Perceptual hashing answers "is this the same picture?" — it is blind to meaning.
Two photos of the same beach from different angles hash completely differently.

This backend uses a pretrained ResNet18 as a feature extractor (the classifier head
removed, so the output is a 512-d embedding) and compares images by cosine
similarity. It requires PyTorch, installed via ``pip install imgsim-py[embed]``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ._io import load_thumbnail

ProgressCallback = Callable[[int, int], None]

EMBEDDING_DIM = 512


def torch_available() -> bool:
    """Whether the optional PyTorch dependencies are importable."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


def extract_features(
    paths: Sequence[str | Path],
    batch_size: int = 16,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Compute L2-normalised ResNet18 embeddings.

    Args:
        paths: Images to embed.
        batch_size: Images per forward pass.
        progress: Called as ``progress(done, total)``.

    Returns:
        ``(len(paths), 512)`` float32 array. Rows for images that could not be
        decoded are filled with NaN so callers can filter them out.

    Raises:
        RuntimeError: If PyTorch is not installed.
    """
    try:
        import torch
        from torchvision import transforms
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "embed mode requires PyTorch. Install it with:\n"
            "    pip install imgsim-py[embed]\n"
            "or\n"
            "    pip install torch torchvision"
        ) from exc

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # strip the 1000-class head, keep the 512-d pooled features
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(weights.meta["mean"], weights.meta["std"]),
        ]
    )

    total = len(paths)
    out = np.full((total, EMBEDDING_DIM), np.nan, dtype=np.float32)
    buffer: list[np.ndarray] = []
    positions: list[int] = []

    def flush() -> None:
        if not buffer:
            return
        with torch.inference_mode():
            batch = torch.from_numpy(np.stack(buffer))
            features = model(batch).numpy()
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-12
        out[positions] = features
        buffer.clear()
        positions.clear()

    for i, path in enumerate(paths):
        image = load_thumbnail(Path(path), 256)
        if image is not None:
            buffer.append(np.asarray(preprocess(image), dtype=np.float32))
            positions.append(i)
        if len(buffer) >= batch_size:
            flush()
        if progress:
            progress(i + 1, total)
    flush()
    return out
