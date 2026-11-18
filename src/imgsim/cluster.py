"""Transitive grouping of similar images.

Similarity is not transitive — A looks like B, B looks like C, but A and C may be
far apart. Treating it as transitive anyway (union-find) is the standard choice for
deduplication: it produces groups where every image is reachable from every other
through a chain of similar pairs, which is what a user means by "these are the same
photo". The cost is occasional chaining on long gradient series, which callers can
bound by lowering the threshold.
"""

from __future__ import annotations

import numpy as np


class DisjointSet:
    """Union-find with path halving and union by size."""

    __slots__ = ("_parent", "_size")

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._size = [1] * n

    def find(self, x: int) -> int:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets containing *a* and *b*. Returns True if they were separate."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]
        return True


def connected_groups(distances: np.ndarray, threshold: float) -> list[list[int]]:
    """Group indices whose pairwise distance is at or below *threshold*.

    Args:
        distances: Square symmetric distance matrix.
        threshold: Inclusive cutoff.

    Returns:
        Index lists for groups of size >= 2, largest group first. Images with no
        neighbour are omitted — callers usually only care about what clustered.
    """
    n = distances.shape[0]
    if n < 2:
        return []

    rows, cols = np.triu_indices(n, k=1)
    close = distances[rows, cols] <= threshold

    forest = DisjointSet(n)
    for i, j in zip(rows[close].tolist(), cols[close].tolist()):
        forest.union(i, j)

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(forest.find(i), []).append(i)

    groups = [members for members in buckets.values() if len(members) > 1]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups
