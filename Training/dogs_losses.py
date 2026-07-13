"""Losses used by the DOGS object and background training branches."""

from __future__ import annotations

import torch


def _pixel_weight(weight: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)
    return weight.to(device=reference.device, dtype=reference.dtype)


def object_foreground_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equation (1): RGB L1 norm per valid foreground pixel."""
    weight = _pixel_weight(foreground_mask, prediction)
    pixel_l1 = (prediction - target).abs().sum(dim=0, keepdim=True)
    return (pixel_l1 * weight).sum() / weight.sum().clamp_min(eps)


def mask_outside_alpha_loss(
    alpha: torch.Tensor,
    outside_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equation (2): absolute alpha response per non-foreground pixel."""
    if alpha.dim() == 3 and alpha.shape[0] == 1:
        alpha = alpha.squeeze(0)
    weight = outside_mask.to(device=alpha.device, dtype=alpha.dtype)
    return (alpha.abs() * weight).sum() / weight.sum().clamp_min(eps)


def background_rgb_loss(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    visible_background_mask: torch.Tensor,
    inpainted: torch.Tensor | None = None,
    object_union_mask: torch.Tensor | None = None,
    rho: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equation (6): observed-background loss plus a weak inpainting prior."""
    visible = object_foreground_loss(
        prediction,
        observed,
        visible_background_mask,
        eps=eps,
    )
    if inpainted is None or object_union_mask is None or rho == 0.0:
        return visible
    prior = object_foreground_loss(
        prediction,
        inpainted,
        object_union_mask,
        eps=eps,
    )
    return visible + float(rho) * prior


def local_plane_residual_loss(
    xyz: torch.Tensor,
    k_nn: int = 8,
    max_points: int = 2048,
) -> torch.Tensor:
    """Equation (7): mean squared point-to-local-plane residual."""
    if xyz.shape[0] <= k_nn:
        return xyz.new_zeros(())

    if xyz.shape[0] > max_points:
        sample_idx = torch.randperm(xyz.shape[0], device=xyz.device)[:max_points]
        xyz_sample = xyz[sample_idx]
    else:
        xyz_sample = xyz

    if xyz_sample.shape[0] <= k_nn:
        return xyz.new_zeros(())

    with torch.no_grad():
        detached = xyz_sample.detach()
        distances = torch.cdist(detached, detached)
        neighbor_idx = distances.topk(k_nn + 1, largest=False).indices[:, 1:]
        neighborhoods = torch.cat(
            (detached[:, None, :], detached[neighbor_idx]),
            dim=1,
        )
        centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
        covariance = centered.transpose(1, 2) @ centered / float(k_nn + 1)
        normals = torch.linalg.eigh(covariance).eigenvectors[:, :, 0]

    offsets = xyz_sample[neighbor_idx] - xyz_sample[:, None, :]
    residual = torch.einsum("ni,nki->nk", normals, offsets)
    return residual.square().mean()
