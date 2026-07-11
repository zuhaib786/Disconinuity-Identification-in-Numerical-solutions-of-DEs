import unittest

import numpy as np

from tci.data import generate_piecewise_fourier


class SyntheticDataTest(unittest.TestCase):
    def test_seed_reproduces_all_arrays(self) -> None:
        first = generate_piecewise_fourier(rng=np.random.default_rng(42))
        second = generate_piecewise_fourier(rng=np.random.default_rng(42))

        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.jump_locations, second.jump_locations)

    def test_jump_labels_match_containing_cells(self) -> None:
        example = generate_piecewise_fourier(
            n_cells=32, max_jumps=12, rng=np.random.default_rng(7)
        )
        expected = np.zeros(32, dtype=np.uint8)
        cells = np.searchsorted(example.vertices, example.jump_locations) - 1
        expected[cells] = 1

        np.testing.assert_array_equal(example.labels, expected)
        self.assertTrue(np.all(example.jump_locations > example.vertices[cells]))
        self.assertTrue(np.all(example.jump_locations < example.vertices[cells + 1]))

    def test_zero_jump_examples_are_supported(self) -> None:
        example = generate_piecewise_fourier(
            n_cells=8, max_jumps=0, rng=np.random.default_rng(1)
        )
        self.assertEqual(int(example.labels.sum()), 0)
        self.assertEqual(example.jump_locations.size, 0)


if __name__ == "__main__":
    unittest.main()

