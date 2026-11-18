# imgsim

**Find duplicate and near-duplicate images in a folder — and actually get the right answer.**

[![CI](https://github.com/zhouchuan315-png/imgsim/actions/workflows/ci.yml/badge.svg)](https://github.com/zhouchuan315-png/imgsim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/badge/pypi-imgsim--py-orange.svg)](https://pypi.org/project/imgsim-py/)

`imgsim` indexes a directory of images, computes a 64-bit perceptual fingerprint for
each one, and groups the ones that are the same picture — resized, recompressed,
cropped, brightened, or converted to black and white.

It exists because plain perceptual hashing has two failure modes that show up in
every real photo library:

| Failure | What you see without a fix |
|---|---|
| **Color blindness.** pHash works on grayscale. | A red apple and a green apple of the same shape are reported identical. |
| **Fingerprint collapse.** Flat images have no low-frequency structure. | A white rectangle and a blue-white stripe pattern hash to the same all-zero fingerprint. |

`imgsim` adds a hue-aware color veto and a low-information guard on top of the
standard pHash + dHash pair. On the bundled benchmark (1000 images, 20 families of
50 variants each) it recovers **20/20 groups with zero cross-family contamination**.

---

## Install

```bash
pip install imgsim-py
```

The distribution is named `imgsim-py` (the plain `imgsim` name on PyPI belongs to an
unrelated, unmaintained project), but the import name is `imgsim`:

```python
import imgsim
```

Optional semantic mode:

```bash
pip install imgsim-py[embed]   # pulls PyTorch + torchvision (~2 GB)
```

## Quick start

### Command line

```bash
imgsim ./photos                      # group duplicates, write ./imgsim-out/report.html
imgsim ./photos -r -t 8              # recurse, stricter threshold
imgsim ./photos --query cat.jpg -k 5 # what else looks like cat.jpg?
imgsim ./photos -f json | jq '.groups | length'
```

`imgsim --help` for everything.

### Python

```python
import imgsim

index = imgsim.ImageIndex()
index.add_dir("photos/", recursive=True)      # decodes once, in parallel

result = index.compare(threshold=10)          # Hamming bits, out of 64
for group in result.groups:
    print(len(group), "similar images")
    for path, distance in zip(group.paths, group.distances):
        print(f"   {distance:>6.1f}  {path}")

imgsim.save_html(result, "report.html")
```

Fingerprints are cheap to keep around, so threshold tuning is instant:

```python
index.save("photos.imgsim")                   # ~300 bytes per image
index = imgsim.ImageIndex.load("photos.imgsim")
result = index.compare(threshold=6)           # no re-decoding
```

Nearest-neighbour lookup for an image that is not in the index:

```python
for path, distance in index.nearest("new-photo.jpg", k=5):
    print(distance, path)
```

## Choosing a threshold

`threshold` is the maximum Hamming distance between two 64-bit fingerprints —
literally *how many of the 64 bits are allowed to differ*.

| Value | What it finds |
|---|---|
| 0–4 | Byte-identical copies, re-encodes, metadata-only edits |
| **10 (default)** | Resized, recompressed, brightness-shifted, grayscale versions |
| 14–18 | The same asset cropped or slightly recomposed |
| >20 | Starts merging genuinely different photos |

Screenshots and digital art usually want 6–8. Phone photo libraries need 10–14
because of sensor noise and HDR processing. **Calibrate on a 200-image sample
first** — the right value varies by roughly 2x between sources.

## How it works

Four steps per image, all numpy, no OpenCV:

1. **Resize to 32×32 grayscale.** Kills resolution and compression differences.
2. **2D DCT, keep the top-left 8×8.** Low frequencies describe layout, not pixels.
3. **Compare each coefficient to the block median.** Above → 1, below → 0. That's 64 bits.
4. **Distance = popcount of the XOR.** Two fingerprints, one XOR, count the 1s.

Two hashes are computed and their distances averaged: `pHash` (DCT, survives resize
and recompression) and `dHash` (horizontal gradients, sensitive to real structural
edits). Averaging suppresses most false positives that either would produce alone.

### The color veto

A 192-float signature is taken by averaging an 8×8 RGB grid, then compared in HSL:

```
distance = 0.55·Δhue·minSat + 0.35·Δsat·minSat + 0.90·Δlightness
```

The `minSat` factor is the important part. Grayscale conversion destroys saturation
but not perceived lightness, so weighting hue and saturation by the *smaller* of the
two saturations makes those terms vanish for black-and-white images — only lightness
is compared. "Photo vs its B&W version" passes; "red apple vs green apple" (both
saturated) is rejected at full strength.

Only the most-different 12 of the 64 cells are averaged, so recoloring one object in
a corner is not diluted away by 52 unchanged cells.

Measured separation: same-image variants score **0.002–0.06**, unrelated images score
**0.39–0.90**. The default cutoff of 0.18 sits in the gap.

### The low-information guard

A pHash is only meaningful if the image has low-frequency structure. For solid
colors, line art, and regular stripes, the AC coefficients are ~0, the median
comparison degenerates, and every such image hashes identically. `imgsim` measures
the AC/DC energy ratio and, below 0.02, demands near-identical color (0.06) before
calling two images similar.

## Performance

Measured on a 4-core Windows laptop, 1000 JPEGs at 400×400:

| Stage | Time |
|---|---|
| Decode + fingerprint | 1.7 s |
| Similarity matrix + clustering | 2.3 s |
| **Total** | **4.0 s** |

Two things make this fast: `Image.draft()` lets libjpeg downsample during decoding
(5–10× on multi-megapixel photos), and Hamming distance runs through
`numpy.bitwise_count`, a hardware popcount, over uint64 codes.

Memory is a fingerprint per image — 8 bytes of hash plus a 192-float color signature
— so a million images fit comfortably in RAM.

Beyond ~50k images the dense `N×N` matrix becomes the bottleneck (50k → 2.5 GB).
At that point use `index.iter_pairs()` to stream instead, or bucket by hash prefix
before comparing. The API leaves room for both.

## Semantic mode

Perceptual hashing answers *"is this the same picture?"*. It is blind to meaning: two
photos of the same beach from different angles hash completely differently.

```bash
imgsim ./photos --mode embed --threshold 0.15
```

`embed` mode extracts 512-d ResNet18 features and compares cosine distance. It
recognises the same subject across viewpoints, at the cost of a PyTorch dependency
and much weaker fine-grained discrimination — it will happily merge two different
sunsets. Use it when "similar subject" matters more than "same file".

## API at a glance

| Member | Purpose |
|---|---|
| `ImageIndex(mode="hash")` | Build an index |
| `.add(path)` / `.add_many(paths)` / `.add_dir(dir)` | Index images (parallel decode) |
| `.compare(threshold, color_threshold, pairs_limit)` | Group into `Result` |
| `.nearest(path, k)` | Top-k neighbours of any image |
| `.save(path)` / `ImageIndex.load(path)` | Persist / restore fingerprints |
| `.iter_pairs(threshold)` | Stream pairs without materialising them |
| `.failed` | Paths that could not be decoded |
| `Result.groups` / `.pairs` / `.duplicates` | Results |
| `render_html(result)` / `save_html(result, path)` | Report |

## Development

```bash
git clone https://github.com/zhouchuan315-png/imgsim
cd imgsim
pip install -e ".[dev]"
pytest
```

Tests build their images from a fixed seed — no binary fixtures in the repo.

## License

MIT. See [LICENSE](LICENSE).

中文文档：[README.zh-CN.md](README.zh-CN.md)
