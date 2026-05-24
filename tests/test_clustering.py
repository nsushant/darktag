import numpy as np
from darktag.tagging.clustering import (
    _extract_features,
    _select_best_label,
    cluster_tagged_particles,
)


class TestExtractFeatures:
    def test_2d(self):
        particles = {'x': np.array([1.0, 2.0, 3.0]), 'y': np.array([4.0, 5.0, 6.0])}
        features, used = _extract_features(particles, ['x', 'y'])
        assert features.shape == (3, 2)
        assert used == ['x', 'y']

    def test_3d(self):
        particles = {
            'x': np.array([1.0, 2.0]),
            'y': np.array([3.0, 4.0]),
            'z': np.array([5.0, 6.0]),
        }
        features, used = _extract_features(particles, ['x', 'y', 'z'])
        assert features.shape == (2, 3)
        assert used == ['x', 'y', 'z']

    def test_with_missing_column(self):
        particles = {'x': np.array([1.0, 2.0]), 'y': np.array([3.0, 4.0])}
        features, used = _extract_features(particles, ['x', 'y', 'vx'])
        assert features.shape == (2, 2)
        assert used == ['x', 'y']

    def test_not_enough_features(self):
        import pytest
        particles = {'x': np.array([1.0, 2.0])}
        with pytest.raises(ValueError, match='at least 2'):
            _extract_features(particles, ['x'])

    def test_empty_particles(self):
        particles = {'x': np.array([]), 'y': np.array([])}
        import pytest
        with pytest.raises(ValueError, match='at least 2'):
            _extract_features(particles, ['x', 'y'])

    def test_6d(self):
        particles = {
            'x': np.array([1.0]), 'y': np.array([2.0]), 'z': np.array([3.0]),
            'vx': np.array([4.0]), 'vy': np.array([5.0]), 'vz': np.array([6.0]),
        }
        features, used = _extract_features(particles, ['x', 'y', 'z', 'vx', 'vy', 'vz'])
        assert features.shape == (1, 6)
        assert used == ['x', 'y', 'z', 'vx', 'vy', 'vz']


class TestSelectBestLabel:
    def test_all_noise(self):
        label = _select_best_label(np.array([-1, -1, -1]), None, np.array([0, 1, 2]))
        assert label == -1

    def test_largest_cluster(self):
        labels = np.array([0, 0, 0, 1, 1, -1])
        label = _select_best_label(labels, None, np.array([0, 1, 2, 3, 4, 5]))
        assert label == 0

    def test_tie_returns_first_largest(self):
        labels = np.array([1, 1, 0, 0, -1])
        label = _select_best_label(labels, None, np.array([0, 1, 2, 3, 4]))
        assert label in (0, 1)

    def test_prev_iords_overlap(self):
        labels = np.array([0, 0, 0, 1, 1, -1])
        iords = np.array([10, 20, 30, 40, 50, 60])
        label = _select_best_label(labels, np.array([10, 40]), iords)
        assert label == 0

    def test_prev_iords_all_noise(self):
        labels = np.array([0, 1, -1])
        iords = np.array([10, 20, 30])
        label = _select_best_label(labels, np.array([30]), iords)
        assert label == 0

    def test_no_prev_overlap_falls_to_largest(self):
        labels = np.array([0, 0, 1, 1, 1, -1])
        iords = np.array([10, 20, 30, 40, 50, 60])
        label = _select_best_label(labels, np.array([99]), iords)
        assert label == 1

    def test_prev_iords_empty(self):
        labels = np.array([0, 0, 1, 1])
        iords = np.array([10, 20, 30, 40])
        label = _select_best_label(labels, np.array([]), iords)
        assert label == 0


class TestClusterTaggedParticles:
    def _make_blob(self, n, center, spread=0.1):
        """Helper: create a 2D gaussian blob."""
        rng = np.random.RandomState(42)
        x = rng.normal(center[0], spread, n)
        y = rng.normal(center[1], spread, n)
        return x, y

    def test_dbscan_2d(self):
        x1, y1 = self._make_blob(50, (0.0, 0.0))
        x2, y2 = self._make_blob(20, (5.0, 5.0))
        particles = {
            'x': np.concatenate([x1, x2]),
            'y': np.concatenate([y1, y2]),
            'iord': np.arange(70),
        }
        labels, best_label, features = cluster_tagged_particles(
            particles, method='dbscan', eps=0.5, dbscan_min_samples=5
        )
        assert best_label != -1
        assert len(set(labels) - {-1}) >= 1
        cluster_size = np.sum(labels == best_label)
        assert cluster_size >= 50

    def test_hdbscan_2d(self):
        x1, y1 = self._make_blob(50, (0.0, 0.0))
        x2, y2 = self._make_blob(20, (5.0, 5.0))
        particles = {
            'x': np.concatenate([x1, x2]),
            'y': np.concatenate([y1, y2]),
            'iord': np.arange(70),
        }
        labels, best_label, features = cluster_tagged_particles(
            particles, method='hdbscan', min_cluster_size=5
        )
        assert best_label != -1
        cluster_size = np.sum(labels == best_label)
        assert cluster_size >= 50

    def test_dbscan_3d(self):
        x1, y1 = self._make_blob(30, (0.0, 0.0))
        z1 = np.random.RandomState(42).normal(0, 0.1, 30)
        particles = {
            'x': x1, 'y': y1, 'z': z1,
            'iord': np.arange(30),
        }
        labels, best_label, features = cluster_tagged_particles(
            particles, method='dbscan', feature_cols=['x', 'y', 'z'],
            eps=0.5, dbscan_min_samples=3
        )
        assert best_label != -1
        assert features.shape[1] == 3

    def test_dbscan_with_prev_iords(self):
        x = np.concatenate([
            np.random.RandomState(42).normal(0, 0.1, 30),
            np.random.RandomState(42).normal(5, 0.1, 10),
        ])
        y = np.concatenate([
            np.random.RandomState(42).normal(0, 0.1, 30),
            np.random.RandomState(42).normal(5, 0.1, 10),
        ])
        iords = np.arange(40)
        particles = {'x': x, 'y': y, 'iord': iords}
        labels, best_label, _ = cluster_tagged_particles(
            particles, prev_iords=np.array([0, 1, 2]),
            method='dbscan', eps=0.5, dbscan_min_samples=3
        )
        assert best_label != -1

    def test_too_few_particles(self):
        particles = {'x': np.array([1.0, 2.0]), 'y': np.array([3.0, 4.0]), 'iord': np.array([0, 1])}
        labels, best_label, _ = cluster_tagged_particles(particles, method='dbscan')
        assert best_label == -1
        assert np.all(labels == -1)

    def test_with_scaling(self):
        x = np.random.RandomState(42).normal(0, 0.1, 30)
        y = np.random.RandomState(42).normal(0, 0.1, 30)
        particles = {'x': x, 'y': y, 'iord': np.arange(30)}
        labels, best_label, _ = cluster_tagged_particles(
            particles, method='dbscan', scale=True,
            eps=0.5, dbscan_min_samples=3
        )
        assert best_label != -1

    def test_hdbscan_with_scaling(self):
        x = np.random.RandomState(42).normal(0, 0.1, 30)
        y = np.random.RandomState(42).normal(0, 0.1, 30)
        particles = {'x': x, 'y': y, 'iord': np.arange(30)}
        labels, best_label, _ = cluster_tagged_particles(
            particles, method='hdbscan', scale=True,
            min_cluster_size=5
        )
        assert best_label != -1

    def test_unknown_method(self):
        import pytest
        particles = {'x': np.array([1.0, 2.0, 3.0]), 'y': np.array([4.0, 5.0, 6.0]), 'iord': np.arange(3)}
        with pytest.raises(ValueError, match='Unknown'):
            cluster_tagged_particles(particles, method='kmeans')

    def test_sample_weight_dbscan(self):
        x = np.random.RandomState(42).normal(0, 0.1, 30)
        y = np.random.RandomState(42).normal(0, 0.1, 30)
        iords = np.arange(30)
        particles = {'x': x, 'y': y, 'iord': iords}
        labels1, best1, _ = cluster_tagged_particles(
            particles, method='dbscan', eps=0.5, dbscan_min_samples=3
        )
        labels2, best2, _ = cluster_tagged_particles(
            particles, method='dbscan', eps=0.5, dbscan_min_samples=3,
            sample_weight=np.ones(30)
        )
        if best1 != -1 and best2 != -1:
            assert np.all(labels1 == labels2)
