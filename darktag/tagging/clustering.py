"""
Clustering utilities for particle tagging.

Provides DBSCAN and HDBSCAN wrappers for identifying main-galaxy
particle clusters in phase space.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

_HDBSCAN_AVAILABLE = False
try:
    import hdbscan
    _HDBSCAN_AVAILABLE = True
except ImportError:
    pass


def _extract_features(particles, feature_cols):
    features = []
    available = []
    for col in feature_cols:
        try:
            arr = np.asarray(particles[col])
            if arr.ndim == 1 and arr.size > 0:
                features.append(arr.astype(np.float64))
                available.append(col)
        except (KeyError, TypeError, ValueError):
            pass
    if len(features) < 2:
        raise ValueError(
            f"Need at least 2 feature dimensions, got {len(features)} "
            f"from requested {feature_cols}. Available: {available}"
        )
    return np.column_stack(features), available


def _select_best_label(labels, prev_iords, particles_iords):
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        return -1

    if prev_iords is not None and len(prev_iords) > 0:
        prev_mask = np.isin(particles_iords, prev_iords)
        if np.any(prev_mask):
            prev_labels = labels[prev_mask]
            prev_no_noise = prev_labels[prev_labels != -1]
            if len(prev_no_noise) > 0:
                counter = Counter(prev_no_noise)
                return counter.most_common(1)[0][0]

    labels_no_noise = labels[labels != -1]
    if len(labels_no_noise) == 0:
        return -1
    counter = Counter(labels_no_noise)
    return counter.most_common(1)[0][0]


def cluster_tagged_particles(
    particles,
    prev_iords=None,
    method='dbscan',
    feature_cols=None,
    scale=False,
    sample_weight=None,
    eps=0.05,
    dbscan_min_samples=2,
    min_cluster_size=10,
    hdbscan_min_samples=None,
    cluster_selection_epsilon=0.0,
    cluster_selection_method='eom',
):
    if feature_cols is None:
        feature_cols = ['x', 'y']

    features, used_cols = _extract_features(particles, feature_cols)
    n = features.shape[0]

    if n <= 2:
        return np.full(n, -1, dtype=int), -1, features

    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = features

    method = method.lower()
    if method == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
        labels = clusterer.fit_predict(features_scaled, sample_weight=sample_weight)
    elif method == 'hdbscan':
        if not _HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan is not installed. Install with: pip install hdbscan"
            )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(features_scaled)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    iords = np.asarray(particles['iord'])
    best_label = _select_best_label(labels, prev_iords, iords)

    return labels, best_label, features_scaled
