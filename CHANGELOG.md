# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-11-19

### Fixed

- `imgsim.__version__` and `imgsim --version` reported `0.1.0` throughout the
  0.1.1 release even though the code was at 0.1.1. The version lived as a
  hardcoded literal in `__init__.py` as well as in `pyproject.toml`, and only
  the latter was bumped during release. It now reads from the installed
  distribution metadata, leaving pyproject.toml as the single source of truth,
  and a test fails if the two ever disagree again.

## [0.1.1] - 2026-11-19

### Fixed

- `iter_pairs()` applied the flat default color cutoff and skipped the
  low-information guard, so on corpora containing flat images it returned pairs
  that `compare()` had already rejected — 91,558 against 91,783 on a 2000-image
  test. It now shares `_effective_distances()` with `compare()` and `nearest()`,
  which removes the possibility of the three drifting apart.
- `nearest()` ignored low-information images and offered no way to adjust the
  color cutoff. It now accepts `color_threshold` and applies the same rules as
  `compare()`.

### Changed

- `iter_pairs()` scans one row at a time with a vectorised comparison instead of
  a Python double loop: 470 ms to 52 ms for 2000 images (9x).
- README accuracy claims are now measured on 992 real photographs and a
  1200-image variant set, replacing the synthetic demo. The cropping limitation
  is documented with its measured numbers rather than left implicit.

## [0.1.0] - 2026-11-18

First public release.

### Added

- `ImageIndex` with `add`, `add_many`, `add_dir`, `compare`, `nearest`, `save`, `load`,
  `iter_pairs` and `clear`.
- Perceptual hashing: DCT `pHash` plus gradient `dHash`, averaged into one distance.
- Color veto in HSL space, weighted by the smaller of the two saturations so that
  grayscale conversions are not rejected while differently-colored subjects are.
- Low-information guard: fingerprints on flat images (solid colors, line art, stripes)
  are flagged by AC/DC energy ratio and require near-identical color to match.
- Union-find clustering, so similarity chains group transitively.
- CLI (`imgsim`) with text, JSON and CSV output, cache reuse, query mode,
  top-k export and `--fail-on-duplicates` for CI.
- Self-contained HTML report with embedded thumbnails.
- Optional `embed` mode using ResNet18 features and cosine distance.

### Performance

- 1000 images (400×400) indexed in 1.7 s and compared in 2.3 s on 4 cores.
- `Image.draft()` downsampling during JPEG decode, 5–10× on large photos.
- Hamming distance via `numpy.bitwise_count` hardware popcount.

[Unreleased]: https://github.com/zhouchuan315-png/imgsim/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/zhouchuan315-png/imgsim/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/zhouchuan315-png/imgsim/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/zhouchuan315-png/imgsim/releases/tag/v0.1.0
