# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/zhouchuan315-png/imgsim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zhouchuan315-png/imgsim/releases/tag/v0.1.0
