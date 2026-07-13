from __future__ import annotations

import unittest

import torch

from Evaluation.metrics import background_leakage_rate, boundary_fscore


class DogsMetricTests(unittest.TestCase):
    def test_boundary_fscore_is_one_for_identical_masks(self) -> None:
        mask = torch.zeros((1, 7, 7))
        mask[:, 2:5, 2:5] = 1.0
        self.assertTrue(torch.isclose(boundary_fscore(mask, mask, tolerance=0), torch.tensor(1.0)))

    def test_background_leakage_rate_matches_equation(self) -> None:
        alpha = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
        mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        expected = torch.tensor((0.2 + 0.4) / (0.8 + 0.2 + 0.6 + 0.4))
        self.assertTrue(torch.isclose(background_leakage_rate(alpha, mask), expected))


if __name__ == "__main__":
    unittest.main()
