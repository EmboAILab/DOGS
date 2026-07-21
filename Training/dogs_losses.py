"""Losses used by the DOGS object and background training branches."""

from __future__ import annotations

import torch


def _pixel_weight(weight: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if weight.dim() == 2:
        weight = weight.unsqueeze(0)
    return weight.to(device=reference.device, dtype=reference.dtype)


def _fixed_normalizer(
    value: float | torch.Tensor | None,
    fallback: torch.Tensor,
    reference: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if value is None:
        normalizer = fallback.sum()
    else:
        normalizer = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    return normalizer.clamp_min(eps)


def supervision_pixel_counts(
    label_maps,
    object_ids: list[int],
    background_label: int,
) -> dict:
    """Count the fixed all-view denominators in Eqs. (1), (2), and (6)."""
    foreground = {int(object_id): 0 for object_id in object_ids}
    outside = {int(object_id): 0 for object_id in object_ids}
    background_visible = 0
    object_union = 0
    view_count = 0

    for label_map in label_maps:
        labels = torch.as_tensor(label_map).long().squeeze()
        valid = labels >= 0
        view_count += 1
        background = valid & (labels == int(background_label))
        background_visible += int(background.sum().item())
        object_union += int((valid & ~background).sum().item())
        for object_id in object_ids:
            object_mask = valid & (labels == int(object_id))
            foreground[int(object_id)] += int(object_mask.sum().item())
            outside[int(object_id)] += int((valid & ~object_mask).sum().item())

    if view_count == 0:
        raise ValueError("at least one training label map is required")
    missing = [object_id for object_id, count in foreground.items() if count == 0]
    if missing:
        raise ValueError(f"object IDs have no foreground pixels: {missing}")
    if background_visible == 0:
        raise ValueError("the training views contain no visible background pixels")

    return {
        "view_count": view_count,
        "object_foreground": foreground,
        "object_outside": outside,
        "background_visible": background_visible,
        "object_union": object_union,
    }


def object_foreground_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
    normalizer: float | torch.Tensor | None = None,
    sample_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equation (1), optionally as an unbiased one-view numerator estimate."""
    weight = _pixel_weight(foreground_mask, prediction)
    pixel_l1 = (prediction - target).abs().sum(dim=0, keepdim=True)
    denominator = _fixed_normalizer(normalizer, weight, prediction, eps)
    return float(sample_scale) * (pixel_l1 * weight).sum() / denominator


def mask_outside_alpha_loss(
    alpha: torch.Tensor,
    outside_mask: torch.Tensor,
    normalizer: float | torch.Tensor | None = None,
    sample_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equation (2), optionally as an unbiased one-view numerator estimate."""
    if alpha.dim() == 3 and alpha.shape[0] == 1:
        alpha = alpha.squeeze(0)
    weight = outside_mask.to(device=alpha.device, dtype=alpha.dtype)
    denominator = _fixed_normalizer(normalizer, weight, alpha, eps)
    return float(sample_scale) * (alpha.abs() * weight).sum() / denominator


def background_rgb_loss(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    visible_background_mask: torch.Tensor,
    inpainted: torch.Tensor | None = None,
    object_union_mask: torch.Tensor | None = None,
    rho: float = 0.0,
    visible_normalizer: float | torch.Tensor | None = None,
    prior_normalizer: float | torch.Tensor | None = None,
    sample_scale: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equation (6), using fixed all-view denominators when provided."""
    visible = object_foreground_loss(
        prediction,
        observed,
        visible_background_mask,
        normalizer=visible_normalizer,
        sample_scale=sample_scale,
        eps=eps,
    )
    if inpainted is None or object_union_mask is None or rho == 0.0:
        return visible
    prior = object_foreground_loss(
        prediction,
        inpainted,
        object_union_mask,
        normalizer=prior_normalizer,
        sample_scale=sample_scale,
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
