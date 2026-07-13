"""DOGS boundary and opacity-leakage metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    x = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    eroded = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    return mask & ~(eroded.squeeze(0).squeeze(0) > 0.5)


def _dilate(mask: torch.Tensor, tolerance: int) -> torch.Tensor:
    if tolerance <= 0:
        return mask.bool()
    x = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = 2 * tolerance + 1
    return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=tolerance).squeeze(0).squeeze(0).bool()


def boundary_fscore(
    alpha_maps: torch.Tensor,
    reference_masks: torch.Tensor,
    alpha_threshold: float = 0.5,
    tolerance: int = 2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Global boundary F-score over views using a pixel matching tolerance."""
    if alpha_maps.shape != reference_masks.shape:
        raise ValueError("alpha_maps and reference_masks must have the same [V,H,W] shape")
    predicted_count = alpha_maps.new_zeros(())
    reference_count = alpha_maps.new_zeros(())
    predicted_match = alpha_maps.new_zeros(())
    reference_match = alpha_maps.new_zeros(())

    for alpha, reference in zip(alpha_maps, reference_masks):
        predicted_boundary = _binary_boundary(alpha >= alpha_threshold)
        reference_boundary = _binary_boundary(reference > 0.5)
        predicted_count += predicted_boundary.sum()
        reference_count += reference_boundary.sum()
        predicted_match += (predicted_boundary & _dilate(reference_boundary, tolerance)).sum()
        reference_match += (reference_boundary & _dilate(predicted_boundary, tolerance)).sum()

    precision = predicted_match / predicted_count.clamp_min(eps)
    recall = reference_match / reference_count.clamp_min(eps)
    return 2.0 * precision * recall / (precision + recall).clamp_min(eps)


def background_leakage_rate(
    alpha_maps: torch.Tensor,
    reference_masks: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fraction of accumulated object opacity assigned outside target masks."""
    if alpha_maps.shape != reference_masks.shape:
        raise ValueError("alpha_maps and reference_masks must have the same [V,H,W] shape")
    outside = 1.0 - reference_masks.to(device=alpha_maps.device, dtype=alpha_maps.dtype)
    return (outside * alpha_maps).sum() / alpha_maps.sum().clamp_min(eps)
