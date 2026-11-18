"""Walk through the imgsim API on a folder of images.

Run with::

    python examples/quickstart.py /path/to/photos

Builds an index, reports groups, saves the HTML report, and demonstrates cache reuse
and nearest-neighbour lookup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import imgsim


def main(directory: str) -> None:
    root = Path(directory)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    index = imgsim.ImageIndex()

    def report(done: int, total: int) -> None:
        print(f"\r  decoding {done}/{total}", end="", flush=True)

    print(f"Indexing {root} ...")
    added = index.add_dir(root, recursive=True, progress=report)
    print(f"\n  {added} images indexed, {len(index.failed)} skipped")

    result = index.compare(threshold=10)
    print(
        f"\n{len(result.groups)} group(s), {len(result.pairs)} similar pair(s), "
        f"{result.elapsed:.2f}s"
    )

    for number, group in enumerate(result.groups, 1):
        print(f"\n  Group {number} ({len(group)} images, max distance {group.max_distance:g})")
        for path, distance in zip(group.paths, group.distances):
            print(f"    {distance:>6.1f}  {path.name}")

    report_path = imgsim.save_html(result, root.parent / "imgsim-report.html")
    print(f"\nReport written to {report_path}")

    # Fingerprints are ~300 bytes per image, so keeping them makes threshold tuning free.
    cache = root.parent / "imgsim-cache.npz"
    index.save(cache)
    restored = imgsim.ImageIndex.load(cache)
    relaxed = restored.compare(threshold=6)
    print(
        f"Re-loaded from cache: {len(relaxed.groups)} group(s) at threshold 6 "
        f"(took {relaxed.elapsed:.3f}s)"
    )

    if index.paths:
        query = index.paths[0]
        print(f"\nNearest neighbours of {query.name}:")
        for path, distance in index.nearest(query, k=3):
            print(f"    {distance:>6.1f}  {path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
