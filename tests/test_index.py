"""End-to-end behaviour of ImageIndex."""

from __future__ import annotations

import numpy as np
import pytest

import imgsim
from imgsim import ImageIndex


def test_add_dir_counts_successes(corpus, index):
    total_files = len(list(corpus["dir"].iterdir()))
    assert len(index) == total_files - 1  # the corrupt file must be excluded
    assert len(index.failed) == 1
    assert index.failed[0].name == "broken.jpg"


def test_add_is_idempotent(corpus, index):
    before = len(index)
    assert index.add(corpus["group_a"][0]) is False
    assert len(index) == before


def test_add_returns_false_for_missing_file(tmp_path):
    instance = ImageIndex()
    assert instance.add(tmp_path / "nope.jpg") is False
    assert len(instance) == 0


def test_compare_recovers_every_family(corpus, index):
    result = index.compare(threshold=12)
    payloads = [{p.name for p in group} for group in result.groups]

    assert {"a0.jpg", "a1.jpg", "a2.jpg", "a3.jpg"} <= set().union(*payloads)
    assert {"b0.jpg", "b1.png", "b2.jpg"} in payloads
    assert {"c0.jpg", "c1.jpg"} in payloads


def test_compare_keeps_unrelated_images_apart(corpus, index):
    result = index.compare(threshold=12)
    grouped = {p.name for group in result.groups for p in group}
    assert "x0.jpg" not in grouped
    assert "x1.jpg" not in grouped


def test_low_information_images_are_not_merged(corpus, index):
    """Regression: white and stripes both hash to zero; color must keep them apart."""
    result = index.compare(threshold=12)
    for group in result.groups:
        names = {p.name for p in group}
        assert not {"white.jpg", "stripes.png"} <= names
        assert not {"white.jpg", "black.jpg"} <= names


def test_threshold_monotonicity(index):
    loose = index.compare(threshold=20)
    tight = index.compare(threshold=4)
    assert len(loose.groups) >= len(tight.groups)
    assert len(loose.pairs) >= len(tight.pairs)


def test_pairs_are_sorted_by_distance(index):
    result = index.compare(threshold=16)
    distances = [pair.distance for pair in result.pairs]
    assert distances == sorted(distances)


def test_pairs_limit_keeps_the_closest(index):
    full = index.compare(threshold=16)
    limited = index.compare(threshold=16, pairs_limit=2)
    assert len(limited.pairs) == min(2, len(full.pairs))
    assert [p.distance for p in limited.pairs] == [p.distance for p in full.pairs[:2]]


def test_group_is_ordered_by_distance_to_representative(index):
    result = index.compare(threshold=16)
    for group in result.groups:
        assert group.distances[0] == 0.0
        assert group.distances == sorted(group.distances)


def test_similarity_property_is_normalised(index):
    result = index.compare(threshold=16)
    for pair in result.pairs[:5]:
        assert 0.0 <= pair.similarity <= 1.0


def test_save_load_roundtrip_is_exact(index, tmp_path):
    """Cached fingerprints must reproduce the in-memory distance matrix bit for bit."""
    cache = tmp_path / "cache.npz"
    index.save(cache)
    assert cache.exists()

    restored = ImageIndex.load(cache)
    assert restored.paths == index.paths
    assert restored.mode == index.mode
    assert np.array_equal(restored._matrices()[0], index._matrices()[0])

    before = index.compare(threshold=12)
    after = restored.compare(threshold=12)
    assert len(after.groups) == len(before.groups)
    assert [len(g) for g in after.groups] == [len(g) for g in before.groups]


def test_save_does_not_leave_temp_files(index, tmp_path):
    index.save(tmp_path / "cache.npz")
    assert [p.name for p in tmp_path.iterdir()] == ["cache.npz"]


def test_save_is_atomic_on_overwrite(index, tmp_path):
    cache = tmp_path / "cache.npz"
    index.save(cache)
    first = cache.stat().st_size
    index.save(cache)
    assert cache.stat().st_size == first


def test_nearest_returns_closest_first(index, corpus):
    results = index.nearest(corpus["group_a"][0], k=3)
    assert len(results) == 3
    assert results[0][0] == corpus["group_a"][0]
    assert results[0][1] == 0.0
    assert [d for _, d in results] == sorted(d for _, d in results)


def test_nearest_rejects_bad_k(index, corpus):
    with pytest.raises(ValueError):
        index.nearest(corpus["group_a"][0], k=0)


def test_nearest_on_corrupt_query_returns_empty(index, corpus):
    assert index.nearest(corpus["broken"], k=3) == []


def test_empty_index_compare_is_safe():
    result = ImageIndex().compare()
    assert result.groups == []
    assert result.n_images == 0


def test_single_image_compare_is_safe(corpus):
    instance = ImageIndex()
    instance.add(corpus["group_a"][0])
    result = instance.compare()
    assert result.groups == []


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        ImageIndex(hash_size=1)
    with pytest.raises(ValueError):
        ImageIndex(color_grid=1)
    with pytest.raises(ValueError):
        ImageIndex(mode="quantum")


def test_add_dir_rejects_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        ImageIndex().add_dir(tmp_path / "ghost")


def test_repr_is_informative(index):
    text = repr(index)
    assert "images=" in text and "failed=1" in text


def test_clear_resets_everything(index):
    index.clear()
    assert len(index) == 0
    assert index.failed == []


def test_progress_callback_is_called(corpus):
    seen: list[tuple[int, int]] = []
    instance = ImageIndex()
    instance.add_many([corpus["group_a"][0], corpus["group_b"][0]],
                      progress=lambda done, total: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]


def test_report_contains_every_group(index, tmp_path):
    result = index.compare(threshold=12)
    html = imgsim.render_html(result)
    assert html.startswith("<!DOCTYPE html>")
    assert "similar groups" in html
    assert "data:image/jpeg;base64," in html
    written = imgsim.save_html(result, tmp_path / "nested" / "report.html")
    assert written.exists()
    assert written.stat().st_size > 1000


def test_report_escapes_filenames(tmp_path):
    """Filenames with HTML metacharacters must not become markup."""
    # Windows forbids <>:"/\|?* in filenames, so use & and quotes, which are legal
    # everywhere and are exactly what an unescaped template gets wrong.
    tricky = tmp_path / "a&b'c.jpg"
    tricky.write_bytes(b"not really an image")
    group = imgsim.Group(paths=[tricky, tricky], distances=[0.0, 0.0])
    result = imgsim.Result(n_images=2, groups=[group], pairs=[], threshold=10)

    html = imgsim.render_html(result)
    assert "a&b'c.jpg" not in html
    assert "a&amp;b" in html


def test_iter_pairs_matches_compare(index):
    result = index.compare(threshold=12)
    streamed = list(index.iter_pairs(12))
    assert len(streamed) == len(result.pairs)
