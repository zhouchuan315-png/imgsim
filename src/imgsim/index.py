"""The main entry point: an index of images you can query for similarity."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ._io import iter_images, load_features
from ._types import Group, Pair, Result
from .cluster import connected_groups
from .color import color_distance
from .fingerprint import dhash, hamming, hamming_matrix, pack_bits, phash

ProgressCallback = Callable[[int, int], None]

#: Below this AC-energy ratio a pHash is considered unreliable (solid colors, line art).
LOW_INFORMATION_AC = 0.02

#: For low-information images, demand near-identical color as well.
LOW_INFORMATION_COLOR_TOL = 0.06

#: Default similarity cutoffs, in raw distance units.
DEFAULT_HASH_THRESHOLD = 10.0
DEFAULT_EMBED_THRESHOLD = 0.15
DEFAULT_COLOR_THRESHOLD = 0.18


class ImageIndex:
    """An in-memory index of image fingerprints.

    Typical use::

        index = ImageIndex()
        index.add_dir("photos/")
        result = index.compare(threshold=10)
        for group in result.groups:
            print(len(group), "similar images")

    The index stores only fingerprints (64 bits + a 192-float color signature per
    image), so a million images fit in a few hundred megabytes. Decoding is the
    expensive part and it happens once, in :meth:`add_dir`.

    Args:
        hash_size: Side length of the pHash block. 8 gives 64-bit fingerprints, which
            is the sweet spot for near-duplicate detection. Raising it to 16 gives 256
            bits and finer discrimination, at the cost of needing a lower threshold.
        color_grid: Side length of the color signature grid. 8 means 64 cells.
        mode: ``"hash"`` for perceptual hashing (no ML dependencies) or ``"embed"``
            for ResNet18 semantic features (requires the ``embed`` extra).
    """

    def __init__(self, hash_size: int = 8, color_grid: int = 8, mode: str = "hash") -> None:
        if hash_size < 2:
            raise ValueError("hash_size must be at least 2")
        if color_grid < 2:
            raise ValueError("color_grid must be at least 2")
        if mode not in ("hash", "embed"):
            raise ValueError("mode must be 'hash' or 'embed'")

        self.hash_size = hash_size
        self.color_grid = color_grid
        self.mode = mode

        self._paths: list[Path] = []
        self._failed: list[Path] = []
        self._seen: set[str] = set()

        self._phash_bits: list[np.ndarray] = []
        self._dhash_bits: list[np.ndarray] = []
        self._signatures: list[np.ndarray] = []
        self._ac_ratios: list[float] = []
        self._valid: list[bool] = []
        self._features: np.ndarray | None = None

        self._cached_distances: np.ndarray | None = None
        self._cached_colors: np.ndarray | None = None

    # ------------------------------------------------------------------ indexing

    def __len__(self) -> int:
        return len(self._paths)

    def __repr__(self) -> str:
        return (
            f"ImageIndex(mode={self.mode!r}, images={len(self._paths)}, "
            f"failed={len(self._failed)})"
        )

    @property
    def paths(self) -> list[Path]:
        """All indexed paths, in insertion order."""
        return list(self._paths)

    @property
    def failed(self) -> list[Path]:
        """Paths that could not be decoded and were skipped."""
        return list(self._failed)

    def clear(self) -> None:
        """Drop everything from the index."""
        self.__init__(self.hash_size, self.color_grid, self.mode)  # type: ignore[misc]

    def add(self, path: str | Path) -> bool:
        """Index a single image.

        Returns:
            True if the image was indexed, False if it was already present or
            could not be decoded.
        """
        return self.add_many([path]) == 1

    def add_many(
        self,
        paths: Sequence[str | Path],
        workers: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> int:
        """Index a sequence of images, decoding in parallel.

        Args:
            paths: Image paths. Duplicates (by resolved path) are skipped.
            workers: Decoding threads. Defaults to ``min(8, cpu_count)``. Pillow
                releases the GIL during decode, so threads scale nearly linearly.
            progress: Called as ``progress(done, total)`` after each image.

        Returns:
            Number of newly indexed images.
        """
        candidates: list[Path] = []
        for raw in paths:
            path = Path(raw)
            key = str(path.resolve()) if path.exists() else str(path)
            if key in self._seen:
                continue
            self._seen.add(key)
            candidates.append(path)

        if not candidates:
            return 0

        if self.mode == "embed":
            return self._add_many_embed(candidates, workers, progress)

        workers = workers if workers is not None else min(8, os.cpu_count() or 1)
        size = self.hash_size * 4
        added = 0

        def decode(path: Path):
            return load_features(path, size, self.color_grid)

        def consume(path: Path, features) -> None:
            nonlocal added
            if features is None:
                self._failed.append(path)
                return
            gray, signature = features
            ph_bits, ac_ratio = phash(gray, self.hash_size)
            self._paths.append(path)
            self._phash_bits.append(ph_bits)
            self._dhash_bits.append(dhash(gray, self.hash_size))
            self._signatures.append(signature)
            self._ac_ratios.append(ac_ratio)
            self._valid.append(True)
            added += 1

        if workers <= 1 or len(candidates) == 1:
            for done, path in enumerate(candidates, 1):
                consume(path, decode(path))
                if progress:
                    progress(done, len(candidates))
        else:
            # Submit in order and collect in order, so `paths` stays deterministic
            # while decoding still happens concurrently.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(decode, p) for p in candidates]
                for done, (path, future) in enumerate(zip(candidates, futures), 1):
                    consume(path, future.result())
                    if progress:
                        progress(done, len(candidates))

        self._invalidate()
        return added

    def add_dir(
        self,
        root: str | Path,
        recursive: bool = False,
        workers: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> int:
        """Index every image in a directory.

        Args:
            root: Directory to scan.
            recursive: Descend into subdirectories.
            workers: Decoding threads, see :meth:`add_many`.
            progress: Called as ``progress(done, total)``.

        Returns:
            Number of newly indexed images.
        """
        directory = Path(root).expanduser()
        if not directory.is_dir():
            raise NotADirectoryError(f"not a directory: {directory}")
        return self.add_many(iter_images(directory, recursive), workers, progress)

    def _add_many_embed(
        self,
        paths: list[Path],
        workers: int | None,
        progress: ProgressCallback | None,
    ) -> int:
        from .embed import extract_features

        try:
            features = extract_features(paths, progress=progress)
        except RuntimeError:
            for path in paths:
                self._failed.append(path)
            return 0

        good, bad = [], []
        for row, path in enumerate(paths):
            if np.isnan(features[row]).any():
                bad.append(path)
            else:
                good.append(row)
                self._paths.append(path)

        self._failed.extend(bad)
        chunk = features[good] if good else np.zeros((0, 512), dtype=np.float32)
        self._features = chunk if self._features is None else np.vstack([self._features, chunk])
        self._valid.extend([True] * len(good))
        self._invalidate()
        return len(good)

    # ---------------------------------------------------------------- comparison

    def _invalidate(self) -> None:
        self._cached_distances = None
        self._cached_colors = None

    def _matrices(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Distance matrix and (hash mode only) color distance matrix, cached."""
        if self._cached_distances is not None:
            return self._cached_distances, self._cached_colors

        if self.mode == "embed" and self._features is not None:
            features = self._features
            distances = np.clip(1.0 - features @ features.T, 0.0, 2.0).astype(np.float32)
            self._cached_distances = distances
            self._cached_colors = None
            return distances, None

        if not self._phash_bits:
            empty = np.zeros((0, 0), dtype=np.uint8)
            self._cached_distances, self._cached_colors = empty, None
            return empty, None

        ph_codes = pack_bits(np.array(self._phash_bits))
        dh_codes = pack_bits(np.array(self._dhash_bits))
        distances = (
            (hamming_matrix(ph_codes).astype(np.float32)
             + hamming_matrix(dh_codes).astype(np.float32)) / 2.0
        )
        signatures = np.array(self._signatures, dtype=np.float32)
        colors = color_distance(signatures, signatures)

        self._cached_distances = distances
        self._cached_colors = colors
        return distances, colors

    def compare(
        self,
        threshold: float | None = None,
        color_threshold: float = DEFAULT_COLOR_THRESHOLD,
        pairs_limit: int | None = None,
    ) -> Result:
        """Group images whose fingerprints are within *threshold*.

        Args:
            threshold: Maximum distance to consider similar. Defaults to 10 in hash
                mode (out of 64 bits) and 0.15 in embed mode (cosine distance).
                Lower is stricter.
            color_threshold: Color veto cutoff, hash mode only. Pairs whose color
                distance exceeds this are never considered similar, regardless of
                their fingerprint distance. This is what stops "red apple" from
                matching "green apple".
            pairs_limit: Cap on the number of pairs returned, keeping the closest.
                Useful when a corpus has enormous duplicate clusters.

        Returns:
            A :class:`~imgsim._types.Result`.
        """
        started = time.perf_counter()
        if threshold is None:
            threshold = DEFAULT_HASH_THRESHOLD if self.mode == "hash" else DEFAULT_EMBED_THRESHOLD

        n = len(self._paths)
        if n < 2:
            return Result(
                n_images=n, failed=list(self._failed), threshold=threshold,
                kind=self.mode, elapsed=time.perf_counter() - started,
            )

        distances, colors = self._matrices()
        effective = distances.astype(np.float32, copy=True)

        if colors is not None:
            # Color veto. Perceptual hashes are grayscale, so without this a red and
            # a green apple of identical shape hash to the same fingerprint.
            low_information = np.array(self._ac_ratios) < LOW_INFORMATION_AC
            strict = low_information[:, None] | low_information[None, :]
            if strict.any():
                # Solid-color and line-art images hash unreliably; demand closer color.
                colors_ok = colors <= np.where(strict, LOW_INFORMATION_COLOR_TOL, color_threshold)
            else:
                colors_ok = colors <= color_threshold
            effective[~colors_ok] = np.inf

        if self._valid and not all(self._valid):
            invalid = np.array([not v for v in self._valid])
            effective[invalid, :] = np.inf
            effective[:, invalid] = np.inf

        groups = self._build_groups(effective, threshold)
        pairs = self._build_pairs(effective, threshold, pairs_limit)

        return Result(
            groups=groups,
            pairs=pairs,
            n_images=n,
            failed=list(self._failed),
            threshold=threshold,
            kind=self.mode,
            elapsed=time.perf_counter() - started,
        )

    def _build_groups(self, distances: np.ndarray, threshold: float) -> list[Group]:
        finite = np.where(np.isfinite(distances), distances, np.inf)
        groups: list[Group] = []
        for members in connected_groups(finite, threshold):
            head = members[0]
            ordered = sorted(members, key=lambda i: finite[head, i])
            groups.append(
                Group(
                    paths=[self._paths[i] for i in ordered],
                    distances=[float(finite[head, i]) for i in ordered],
                )
            )
        return groups

    def _build_pairs(
        self, distances: np.ndarray, threshold: float, limit: int | None
    ) -> list[Pair]:
        n = distances.shape[0]
        rows, cols = np.triu_indices(n, k=1)
        values = distances[rows, cols]
        keep = values <= threshold
        rows, cols, values = rows[keep], cols[keep], values[keep]
        if limit is not None and values.size > limit:
            order = np.argpartition(values, limit)[:limit]
            order = order[np.argsort(values[order])]
            rows, cols, values = rows[order], cols[order], values[order]
        else:
            order = np.argsort(values)
            rows, cols, values = rows[order], cols[order], values[order]
        return [
            Pair(a=self._paths[int(i)], b=self._paths[int(j)], distance=float(v), kind=self.mode)
            for i, j, v in zip(rows, cols, values)
        ]

    def nearest(self, path: str | Path, k: int = 5) -> list[tuple[Path, float]]:
        """Find the *k* indexed images closest to an arbitrary image.

        The query image does not need to be in the index; it is fingerprinted on
        the fly and compared against everything stored.

        Args:
            path: Query image.
            k: How many neighbours to return.

        Returns:
            ``[(path, distance), ...]`` sorted by increasing distance. Empty if the
            query cannot be decoded.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        query = Path(path)
        if self.mode == "embed":
            return self._nearest_embed(query, k)

        if not self._phash_bits:
            return []

        features = load_features(query, self.hash_size * 4, self.color_grid)
        if features is None:
            return []
        gray, signature = features
        ph_bits, _ = phash(gray, self.hash_size)
        dh_bits = dhash(gray, self.hash_size)

        ph_q = pack_bits(np.array([ph_bits]))
        dh_q = pack_bits(np.array([dh_bits]))
        ph_codes = pack_bits(np.array(self._phash_bits))
        dh_codes = pack_bits(np.array(self._dhash_bits))

        combined = (
            hamming(ph_q, ph_codes).astype(np.float32)
            + hamming(dh_q, dh_codes).astype(np.float32)
        ) / 2.0
        colors = color_distance(np.array([signature], dtype=np.float32),
                                np.array(self._signatures, dtype=np.float32))[0]
        # hamming() returns a (1, n) matrix for a single query; flatten before sorting
        scores = combined[0]
        scores = np.where(colors <= DEFAULT_COLOR_THRESHOLD, scores, np.inf)

        valid = np.array(self._valid)
        scores = np.where(valid, scores, np.inf)

        order = np.argsort(scores)[:k]
        return [(self._paths[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def _nearest_embed(self, query: Path, k: int) -> list[tuple[Path, float]]:
        from .embed import extract_features

        if self._features is None or not len(self._features):
            return []
        features = extract_features([query])
        if np.isnan(features).any():
            return []
        distances = 1.0 - self._features @ features[0]
        order = np.argsort(distances)[:k]
        return [(self._paths[i], float(distances[i])) for i in order]

    # -------------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Persist fingerprints so later runs can skip decoding.

        The saved file contains no image data, only signatures, so it is small
        (a few hundred bytes per image).
        """
        target = Path(path)
        payload = {
            "version": np.array([1]),
            "mode": np.array([self.mode]),
            "hash_size": np.array([self.hash_size]),
            "color_grid": np.array([self.color_grid]),
            "paths": np.array([str(p) for p in self._paths]),
            "failed": np.array([str(p) for p in self._failed]),
        }
        if self.mode == "hash" and self._phash_bits:
            payload["phash"] = pack_bits(np.array(self._phash_bits))
            payload["dhash"] = pack_bits(np.array(self._dhash_bits))
            payload["signatures"] = np.array(self._signatures, dtype=np.float32)
            payload["ac_ratios"] = np.array(self._ac_ratios, dtype=np.float32)
        if self._features is not None:
            payload["features"] = self._features
        tmp = target.with_suffix(target.suffix + ".tmp")
        # Pass a handle, not a path: np.savez_compressed appends ".npz" to any path
        # that lacks it, which would leave us renaming a file that does not exist.
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **payload)
        tmp.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> ImageIndex:
        """Restore an index written by :meth:`save`."""
        data = np.load(Path(path), allow_pickle=False)
        index = cls(
            hash_size=int(data["hash_size"][0]),
            color_grid=int(data["color_grid"][0]),
            mode=str(data["mode"][0]),
        )
        index._paths = [Path(p) for p in data["paths"].tolist()]
        index._seen = {str(p.resolve()) for p in index._paths}
        if "failed" in data:
            index._failed = [Path(p) for p in data["failed"].tolist()]

        if "phash" in data:
            ph = data["phash"].ravel()
            dh = data["dhash"].ravel()
            width = index.hash_size ** 2
            # `pack_bits` produced the codes through a big-endian view, so the bytes
            # must be read back big-endian too. Using the native (little-endian)
            # representation here would silently reverse every bit order.
            index._phash_bits = [
                np.unpackbits(np.frombuffer(c.astype(">u8").tobytes(), dtype=np.uint8))[:width]
                for c in ph
            ]
            index._dhash_bits = [
                np.unpackbits(np.frombuffer(c.astype(">u8").tobytes(), dtype=np.uint8))[:width]
                for c in dh
            ]
            index._signatures = list(data["signatures"])
            index._ac_ratios = data["ac_ratios"].tolist()
            index._valid = [True] * len(index._paths)

        if "features" in data:
            index._features = data["features"]
            index._valid = [True] * len(index._paths)

        return index

    def iter_pairs(self, threshold: float) -> Iterator[tuple[Path, Path, float]]:
        """Stream similar pairs instead of materialising them all.

        Equivalent to :meth:`compare` but yields lazily, which matters when a large
        duplicate cluster would otherwise produce millions of Pair objects.
        """
        distances, colors = self._matrices()
        n = distances.shape[0]
        for i in range(n - 1):
            row = distances[i]
            for j in range(i + 1, n):
                value = row[j]
                if value > threshold:
                    continue
                if colors is not None and colors[i, j] > DEFAULT_COLOR_THRESHOLD:
                    continue
                yield self._paths[i], self._paths[j], float(value)
