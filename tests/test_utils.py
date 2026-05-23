import numpy as np


def initialize_arrays(n):
    x = []
    for i in range(n):
        x.append(np.array([]))
    return x


def get_dist(pos):
    return np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)


def group_mergers(z_merges, h_merges):
    merging_halos_grouped_by_z = []
    z_unique_values = sorted(list(set(z_merges)))
    for i in z_unique_values:
        lists_of_halos_merging_at_current_z = np.where(z_merges == i)
        all_halos_merging_at_current_z = []
        for list_of_halos in lists_of_halos_merging_at_current_z:
            halos_merging_at_current_z = np.array([])
            for merging_halo_object in list_of_halos:
                halos_merging_at_current_z = np.append(halos_merging_at_current_z, h_merges[merging_halo_object][1:])
            all_halos_merging_at_current_z.append(halos_merging_at_current_z)
        merging_halos_grouped_by_z.append(all_halos_merging_at_current_z)
    return merging_halos_grouped_by_z, z_unique_values


class TestInitializeArrays:
    def test_basic(self):
        result = initialize_arrays(3)
        assert len(result) == 3
        for arr in result:
            assert isinstance(arr, np.ndarray)
            assert arr.size == 0

    def test_zero(self):
        result = initialize_arrays(0)
        assert len(result) == 0

    def test_negative(self):
        result = initialize_arrays(-1)
        assert len(result) == 0

    def test_large(self):
        result = initialize_arrays(100)
        assert len(result) == 100


class TestGetDist:
    def test_basic(self):
        pos = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 0.0]])
        dists = get_dist(pos)
        expected = np.array([1.0, 5.0, 0.0])
        np.testing.assert_array_almost_equal(dists, expected)

    def test_empty(self):
        pos = np.empty((0, 3))
        dists = get_dist(pos)
        assert dists.size == 0

    def test_negative_coords(self):
        pos = np.array([[-3.0, -4.0, 0.0]])
        dists = get_dist(pos)
        np.testing.assert_array_almost_equal(dists, [5.0])

    def test_origin(self):
        pos = np.array([[0.0, 0.0, 0.0]])
        dists = get_dist(pos)
        assert dists[0] == 0.0


class TestGroupMergers:
    def test_empty(self):
        result, z_vals = group_mergers(np.array([]), np.array([]))
        assert len(result) == 0
        assert len(z_vals) == 0

    def test_single_z(self):
        h_merges = np.array(["halo_A", "halo_B", "halo_C"], dtype=object)
        result, z_vals = group_mergers(np.array([1.0, 1.0, 1.0]), h_merges)
        assert len(z_vals) == 1
        assert z_vals[0] == 1.0
        assert len(result[0]) > 0

    def test_multiple_z(self):
        h_merges = np.array(["halo_A", "halo_B", "halo_C", "halo_D"], dtype=object)
        z_merges = np.array([1.0, 1.0, 2.0, 2.0])
        result, z_vals = group_mergers(z_merges, h_merges)
        assert len(z_vals) == 2
        assert z_vals[0] == 1.0
        assert z_vals[1] == 2.0

    def test_z_order(self):
        h_merges = np.array(["halo_A", "halo_B"], dtype=object)
        z_merges = np.array([2.0, 1.0])
        result, z_vals = group_mergers(z_merges, h_merges)
        assert z_vals == [1.0, 2.0]
