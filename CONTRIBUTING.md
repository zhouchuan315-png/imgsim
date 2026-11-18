# Contributing to imgsim

Thanks for taking the time to contribute. This is a small library, so the process is
short.

## Setup

```bash
git clone https://github.com/zhouchuan315-png/imgsim
cd imgsim
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before opening a PR

```bash
pytest          # all tests must pass
ruff check .    # lint must be clean
```

CI runs both on Linux, macOS and Windows for Python 3.9–3.13. If a test only fails on
one platform, say so in the PR — it is usually a path or encoding issue, not logic.

## Guidelines

**Tests come with the change.** Every bug fix needs a regression test that fails
before the fix. Several tests in this repo are named after the failure they guard
against (`test_flat_white_and_stripes_are_separated`, `test_pack_bits_roundtrip_preserves_bit_order`)
— keep that convention.

**No binary fixtures.** Tests generate images from a fixed seed via
`tests/conftest.py::make_image`. The repo stays small and results stay reproducible.

**Changing the algorithm changes everyone's thresholds.** If you touch the hashing or
the color distance, include before/after numbers on a corpus of at least 100 images in
the PR description. A change that improves one case by 10% while degrading another by
5% is usually not worth it.

**Keep the dependency footprint at two.** `numpy` and `Pillow`, nothing else in the
core path. Anything heavier goes behind an optional extra. This is why Hamming distance
has a numpy-1.x fallback instead of just requiring scipy.

## Reporting bugs

Include:

- `imgsim` version, Python version, OS
- The command or code snippet you ran
- What you expected vs what happened
- If it is a misclassification: two example images (or a description that lets us
  reproduce them) and the threshold you used

## Requesting features

Open an issue describing the problem, not the solution. "Our photo library has 40k
images and clustering takes 8 minutes" is more useful than "add FAISS support" — the
right fix might be bucketing, a coarser hash, or something we haven't thought of.

## License

By contributing you agree that your contribution is licensed under the MIT License.
