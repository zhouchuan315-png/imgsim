"""Union-find grouping."""

from __future__ import annotations

import numpy as np

from imgsim.cluster import DisjointSet, connected_groups


def test_disjoint_set_merges_transitively():
    forest = DisjointSet(5)
    assert forest.union(0, 1) is True
    assert forest.union(1, 2) is True
    assert forest.union(3, 4) is True
    assert forest.find(0) == forest.find(2)
    assert forest.find(0) != forest.find(4)


def test_disjoint_set_reports_redundant_union():
    forest = DisjointSet(3)
    forest.union(0, 1)
    assert forest.union(0, 1) is False


def test_connected_groups_finds_pairs():
    distances = np.array(
        [
            [0, 2, 40, 40],
            [2, 0, 40, 40],
            [40, 40, 0, 3],
            [40, 40, 3, 0],
        ],
        dtype=float,
    )
    groups = connected_groups(distances, threshold=10)
    assert groups == [[0, 1], [2, 3]]


def test_connected_groups_chains_through_intermediates():
    """A-B and B-C are similar even when A-C is not; dedup wants them grouped."""
    distances = np.array(
        [[0, 5, 30], [5, 0, 5], [30, 5, 0]],
        dtype=float,
    )
    assert connected_groups(distances, threshold=6) == [[0, 1, 2]]


def test_connected_groups_ignores_singletons():
    distances = np.array([[0, 50], [50, 0]], dtype=float)
    assert connected_groups(distances, threshold=10) == []


def test_connected_groups_respects_threshold_boundary():
    distances = np.array([[0, 10], [10, 0]], dtype=float)
    assert connected_groups(distances, threshold=10) == [[0, 1]]
    assert connected_groups(distances, threshold=9.99) == []


def test_connected_groups_orders_by_size():
    distances = np.full((5, 5), 50.0, dtype=float)
    np.fill_diagonal(distances, 0)
    distances[0, 1] = distances[1, 0] = 1
    distances[2, 3] = distances[3, 2] = 1
    distances[2, 4] = distances[4, 2] = 1
    distances[3, 4] = distances[4, 3] = 1
    groups = connected_groups(distances, threshold=5)
    assert [len(g) for g in groups] == [3, 2]


def test_connected_groups_handles_tiny_inputs():
    assert connected_groups(np.zeros((0, 0)), 10) == []
    assert connected_groups(np.zeros((1, 1)), 10) == []


def test_connected_groups_skips_infinite_distances():
    distances = np.array([[0, np.inf], [np.inf, 0]], dtype=float)
    assert connected_groups(distances, threshold=10) == []
