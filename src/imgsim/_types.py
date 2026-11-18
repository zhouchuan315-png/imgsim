"""Public result types returned by :class:`imgsim.ImageIndex`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Maximum possible distance, used to normalise hash distances into a similarity score.
HASH_BITS = 64


@dataclass(frozen=True)
class Pair:
    """A pair of images that the algorithm considers similar.

    Attributes:
        a: First image path.
        b: Second image path.
        distance: Raw distance. Hamming bits (0-64) in hash mode, cosine distance
            (0-2) in embed mode. Lower is more similar.
        kind: ``"hash"`` or ``"embed"``, indicating how the distance was measured.
    """

    a: Path
    b: Path
    distance: float
    kind: str = "hash"

    @property
    def similarity(self) -> float:
        """Distance normalised to ``[0, 1]``, where 1.0 means identical."""
        if self.kind == "hash":
            return max(0.0, 1.0 - self.distance / HASH_BITS)
        return max(0.0, 1.0 - self.distance)


@dataclass
class Group:
    """A set of images that are transitively similar to each other.

    ``paths[0]`` is the group representative (the first image encountered); the
    remaining paths are ordered by increasing distance to it.

    Attributes:
        paths: Image paths in this group, representative first.
        distances: Distance of each path to ``paths[0]``. Always starts with 0.0.
    """

    paths: list[Path]
    distances: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

    def __getitem__(self, index):
        return self.paths[index]

    @property
    def representative(self) -> Path:
        """The image this group is anchored on."""
        return self.paths[0]

    @property
    def max_distance(self) -> float:
        """Largest distance within the group, i.e. how loose the group is."""
        return max(self.distances) if self.distances else 0.0


@dataclass
class Result:
    """Everything a single comparison produced.

    Attributes:
        groups: Similar groups, ordered by size (largest first).
        pairs: Every similar pair, ordered by increasing distance.
        n_images: Number of images successfully indexed.
        failed: Paths that could not be decoded, so they were excluded.
        threshold: The threshold used for this comparison.
        kind: ``"hash"`` or ``"embed"``.
        elapsed: Wall-clock seconds spent on the comparison (excluding indexing).
    """

    groups: list[Group] = field(default_factory=list)
    pairs: list[Pair] = field(default_factory=list)
    n_images: int = 0
    failed: list[Path] = field(default_factory=list)
    threshold: float = 10.0
    kind: str = "hash"
    elapsed: float = 0.0

    @property
    def duplicates(self) -> list[Group]:
        """Groups with exactly identical fingerprints (distance 0)."""
        return [g for g in self.groups if g.max_distance == 0]

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)
