import unittest

import numpy as np

from tci.graphs import cell_features, line_graph_edges


class LineGraphTest(unittest.TestCase):
    def test_bidirectional_line_edges(self) -> None:
        edges = line_graph_edges(3)
        observed = {tuple(edge) for edge in edges.T.tolist()}
        self.assertEqual(observed, {(0, 1), (1, 0), (1, 2), (2, 1)})

    def test_periodic_line_adds_wraparound_edges(self) -> None:
        edges = line_graph_edges(4, periodic=True)
        observed = {tuple(edge) for edge in edges.T.tolist()}
        self.assertIn((3, 0), observed)
        self.assertIn((0, 3), observed)

    def test_cell_features_include_relative_geometry(self) -> None:
        dofs = np.array([[1.0, 2.0], [3.0, 5.0]])
        features = cell_features(dofs, np.array([0.0, 1.0, 3.0]))
        self.assertEqual(features.shape, (2, 4))
        np.testing.assert_allclose(features[:, -2], [2 / 3, 4 / 3])


if __name__ == "__main__":
    unittest.main()

