from __future__ import annotations

import unittest

import torch

from Evaluation.metrics import (
    background_leakage_rate,
    boundary_fscore,
    masked_psnr,
    masked_ssim,
)


class DogsMetricTests(unittest.TestCase):
    def test_psnr_is_infinite_for_identical_images(self) -> None:
        image = torch.full((3, 16, 16), 0.4)
        self.assertTrue(torch.isinf(masked_psnr(image, image)))

    def test_masked_psnr_excludes_outside_error(self) -> None:
        target = torch.zeros((3, 16, 16))
        prediction = target.clone()
        prediction[:, 8:, :] = 1.0
        mask = torch.zeros((16, 16))
        mask[:8, :] = 1.0
        self.assertTrue(torch.isinf(masked_psnr(prediction, target, mask)))

    def test_ssim_is_one_for_identical_images(self) -> None:
        image = torch.rand((3, 16, 16))
        self.assertTrue(torch.isclose(masked_ssim(image, image), torch.tensor(1.0), atol=1e-5))

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
