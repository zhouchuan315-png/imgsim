"""Command line interface.

Design notes:
  * One positional argument (the directory) covers the common case; ``--query``
    switches to nearest-neighbour lookup instead of a separate subcommand.
  * ``--format json`` makes the tool composable in pipelines.
  * ``--fail-on-duplicates`` lets CI reject a commit that introduces duplicate assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__
from ._types import Result
from .index import (
    DEFAULT_COLOR_THRESHOLD,
    DEFAULT_EMBED_THRESHOLD,
    DEFAULT_HASH_THRESHOLD,
    ImageIndex,
)
from .report import save_html


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _Progress:
    """Single-line progress reporter that stays quiet when piped."""

    def __init__(self, label: str, enabled: bool = True) -> None:
        self.label = label
        self.enabled = enabled and _is_tty()
        self._started = False

    def __call__(self, done: int, total: int) -> None:
        if not self.enabled:
            return
        self._started = True
        width = 24
        filled = int(width * done / total) if total else width
        bar = "=" * filled + " " * (width - filled)
        print(f"\r  {self.label} [{bar}] {done}/{total}", end="", flush=True)

    def close(self) -> None:
        if self._started:
            print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imgsim",
        description="Find duplicate and near-duplicate images.",
        epilog="Examples:\n"
        "  imgsim ./photos\n"
        "  imgsim ./photos -r -t 8 -o ./out\n"
        "  imgsim ./photos --query cat.jpg -k 5\n"
        "  imgsim ./photos --format json | jq '.groups | length'",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", type=Path, help="Directory containing images.")
    parser.add_argument(
        "-t", "--threshold", type=float, default=None,
        help=f"Similarity cutoff. Hamming bits in hash mode (default {DEFAULT_HASH_THRESHOLD:g}), "
             f"cosine distance in embed mode (default {DEFAULT_EMBED_THRESHOLD:g}). Lower is stricter.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Descend into subdirectories."
    )
    parser.add_argument(
        "--mode", choices=("hash", "embed"), default="hash",
        help="hash = perceptual fingerprint (default, no ML deps); embed = ResNet18 features.",
    )
    parser.add_argument(
        "--color-threshold", type=float, default=DEFAULT_COLOR_THRESHOLD,
        help=f"Color veto cutoff, hash mode only (default {DEFAULT_COLOR_THRESHOLD}). "
             "Lower rejects more 'same shape, different color' matches.",
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("imgsim-out"),
                        help="Output directory (default ./imgsim-out).")
    parser.add_argument(
        "-f", "--format", choices=("text", "json", "csv"), default="text",
        help="Stdout format (default text).",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation.")
    parser.add_argument("--topk", type=int, default=0,
                        help="Also write each image's top-K nearest neighbours to topk.csv.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Decoding threads (default: min(8, cpu_count)).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output.")
    parser.add_argument("--cache", type=Path, default=None,
                        help="Fingerprint cache file (default <out>/features.npz).")
    parser.add_argument("--reuse", action="store_true",
                        help="Load the cache if present instead of decoding, so threshold "
                             "changes are instant.")
    parser.add_argument("--query", type=Path, default=None,
                        help="Find images similar to this one instead of grouping a directory.")
    parser.add_argument("-k", "--neighbours", type=int, default=5,
                        help="Number of results for --query (default 5).")
    parser.add_argument("--fail-on-duplicates", action="store_true",
                        help="Exit with status 1 if any similar group is found.")
    parser.add_argument("--version", action="version", version=f"imgsim {__version__}")
    return parser


def _print_text(result: Result, out_dir: Path, report: Path | None) -> None:
    if not result.groups:
        print("\nNo similar images found.")
    else:
        print(f"\n{len(result.groups)} group(s), {len(result.pairs)} similar pair(s):\n")
        for number, group in enumerate(result.groups[:20], 1):
            head = f"  [{number}] {len(group)} images"
            span = f"  (max distance {group.max_distance:g})"
            print(head + span)
            for path, distance in zip(group.paths[:6], group.distances[:6]):
                print(f"       {distance:>6g}  {path}")
            if len(group) > 6:
                print(f"       ... and {len(group) - 6} more")
        if len(result.groups) > 20:
            print(f"\n  ... and {len(result.groups) - 20} more groups")
    if report:
        print(f"\nReport: {report}")


def _print_json(result: Result) -> None:
    payload = {
        "images": result.n_images,
        "threshold": result.threshold,
        "mode": result.kind,
        "elapsed": round(result.elapsed, 3),
        "failed": [str(p) for p in result.failed],
        "groups": [
            {
                "size": len(group),
                "max_distance": group.max_distance,
                "images": [{"path": str(p), "distance": d}
                           for p, d in zip(group.paths, group.distances)],
            }
            for group in result.groups
        ],
        "pairs": [
            {"a": str(pair.a), "b": str(pair.b), "distance": pair.distance}
            for pair in result.pairs
        ],
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _print_csv(result: Result) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(["group", "size", "distance", "path"])
    for number, group in enumerate(result.groups, 1):
        for path, distance in zip(group.paths, group.distances):
            writer.writerow([number, len(group), round(distance, 4), str(path)])


def _write_topk(result: Result, index: ImageIndex, path: Path, k: int) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "rank", "neighbour", "distance"])
        for source in index.paths:
            for rank, (neighbour, distance) in enumerate(index.nearest(source, k), 1):
                if neighbour == source:
                    continue
                writer.writerow([source.name, rank, neighbour.name, round(distance, 4)])


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    directory: Path = args.directory.expanduser()
    if not directory.is_dir():
        parser.error(f"not a directory: {directory}")

    out_dir: Path = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: Path = args.cache or (out_dir / "features.npz")

    index = ImageIndex(mode=args.mode)

    reused = False
    if args.reuse and cache.exists():
        try:
            index = ImageIndex.load(cache)
            if index.mode != args.mode:
                index = ImageIndex(mode=args.mode)
            else:
                reused = True
        except Exception as exc:  # - a stale cache should never be fatal
            print(f"warning: could not reuse cache ({exc}); re-indexing", file=sys.stderr)
            index = ImageIndex(mode=args.mode)

    if not reused:
        progress = _Progress("indexing", enabled=not args.quiet)
        try:
            index.add_dir(directory, recursive=args.recursive,
                          workers=args.workers, progress=progress)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        finally:
            progress.close()
        index.save(cache)

    if args.query is not None:
        neighbours = index.nearest(args.query, args.neighbours)
        if args.format == "json":
            json.dump(
                [{"path": str(p), "distance": d} for p, d in neighbours],
                sys.stdout, ensure_ascii=False, indent=2,
            )
            sys.stdout.write("\n")
        else:
            print(f"\n{len(neighbours)} nearest to {args.query.name}:")
            for path, distance in neighbours:
                print(f"  {distance:>8.3f}  {path}")
        if index.failed:
            print(f"\n{len(index.failed)} file(s) could not be decoded.", file=sys.stderr)
        return 0

    result = index.compare(threshold=args.threshold, color_threshold=args.color_threshold)

    report: Path | None = None
    if not args.no_report:
        report = save_html(result, out_dir / "report.html")

    if result.groups:
        with open(out_dir / "groups.csv", "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group", "size", "distance", "path"])
            for number, group in enumerate(result.groups, 1):
                for path, distance in zip(group.paths, group.distances):
                    writer.writerow([number, len(group), round(distance, 4), str(path)])
        with open(out_dir / "pairs.csv", "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["a", "b", "distance", "similarity"])
            for pair in result.pairs:
                writer.writerow([str(pair.a), str(pair.b),
                                 round(pair.distance, 4), round(pair.similarity, 4)])

    if args.topk > 0:
        _write_topk(result, index, out_dir / "topk.csv", args.topk)

    if args.format == "json":
        _print_json(result)
    elif args.format == "csv":
        _print_csv(result)
    else:
        _print_text(result, out_dir, report)

    if result.failed:
        print(f"{len(result.failed)} file(s) could not be decoded.", file=sys.stderr)

    if args.fail_on_duplicates and result.groups:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
