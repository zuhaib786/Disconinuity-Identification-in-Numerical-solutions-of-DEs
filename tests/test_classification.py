import unittest

import numpy as np

from tci.evaluation import BinaryConfusionMatrix, evaluate_binary


class BinaryClassificationTest(unittest.TestCase):
    def test_confusion_matrix_uses_standard_fp_and_fn_definitions(self) -> None:
        truth = np.array([1, 1, 1, 0, 0, 0])
        prediction = np.array([1, 0, 0, 1, 0, 0])

        matrix = BinaryConfusionMatrix.from_predictions(truth, prediction)

        self.assertEqual(matrix.true_positive, 1)
        self.assertEqual(matrix.false_negative, 2)
        self.assertEqual(matrix.false_positive, 1)
        self.assertEqual(matrix.true_negative, 2)
        self.assertAlmostEqual(matrix.precision, 1 / 2)
        self.assertAlmostEqual(matrix.recall, 1 / 3)

    def test_scores_are_thresholded_consistently(self) -> None:
        result = evaluate_binary([0, 1, 1, 0], [0.1, 0.5, 0.49, 0.9], 0.5)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["true_negative"], 1)

    def test_invalid_binary_labels_fail_loudly(self) -> None:
        with self.assertRaises(ValueError):
            BinaryConfusionMatrix.from_predictions([0, 2], [0, 1])


if __name__ == "__main__":
    unittest.main()

