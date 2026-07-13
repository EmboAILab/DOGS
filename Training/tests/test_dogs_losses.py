from __future__ import annotations

import unittest

import torch

from Training.dogs_losses import (
    background_rgb_loss,
    local_plane_residual_loss,
    mask_outside_alpha_loss,
    object_foreground_loss,
)


class DogsLossTests(unittest.TestCase):
    def test_object_foreground_loss_matches_equation(self) -> None:
        prediction = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]], [[1.0, 4.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[1.0, 0.0]])
        self.assertTrue(torch.isclose(object_foreground_loss(prediction, target, mask), torch.tensor(2.0)))

    def test_mask_outside_alpha_loss_matches_equation(self) -> None:
        alpha = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
        outside = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertTrue(torch.isclose(mask_outside_alpha_loss(alpha, outside), torch.tensor(0.3)))

    def test_background_rgb_loss_uses_separate_region_normalization(self) -> None:
        prediction = torch.zeros((3, 1, 2))
        observed = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]])
        inpainted = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]], [[0.0, 2.0]]])
        visible = torch.tensor([[1.0, 0.0]])
        occluded = torch.tensor([[0.0, 1.0]])
        expected = torch.tensor(3.0 + 0.1 * 6.0)
        actual = background_rgb_loss(
            prediction,
            observed,
            visible,
            inpainted=inpainted,
            object_union_mask=occluded,
            rho=0.1,
        )
        self.assertTrue(torch.isclose(actual, expected))

    def test_local_plane_residual_is_zero_for_planar_points(self) -> None:
        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        )
        actual = local_plane_residual_loss(xyz, k_nn=3)
        self.assertTrue(torch.isclose(actual, torch.tensor(0.0), atol=1e-7))


if __name__ == "__main__":
    unittest.main()
