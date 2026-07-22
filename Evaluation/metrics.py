"""DOGS reconstruction, boundary, and opacity-leakage metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batched_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError("images must have [C,H,W] or [N,C,H,W] shape")
    return image


def _as_batched_mask(mask: torch.Tensor | None, image: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.shape[0] != image.shape[0] or mask.shape[-2:] != image.shape[-2:]:
        raise ValueError("mask must match the image batch and spatial dimensions")
    return mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """PSNR over all pixels or the valid mask region."""
    prediction = _as_batched_image(prediction)
    target = _as_batched_image(target).to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = _as_batched_mask(mask, prediction)
    squared_error = (prediction - target).square()
    if valid is None:
        mse = squared_error.mean()
    else:
        mse = (squared_error * valid).sum() / (valid.sum() * prediction.shape[1]).clamp_min(eps)
    if mse <= eps:
        return prediction.new_tensor(float("inf"))
    return 10.0 * torch.log10(prediction.new_tensor(float(data_range) ** 2) / mse)


def _gaussian_window(window_size: int, sigma: float, channels: int, image: torch.Tensor) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=image.device, dtype=image.dtype)
    coordinates = coordinates - (window_size - 1) / 2.0
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def masked_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Single-scale SSIM averaged over all pixels or the valid mask region."""
    prediction = _as_batched_image(prediction)
    target = _as_batched_image(target).to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = _as_batched_mask(mask, prediction)
    if valid is not None:
        prediction = prediction * valid
        target = target * valid

    channels = prediction.shape[1]
    window = _gaussian_window(window_size, sigma, channels, prediction)
    padding = window_size // 2
    mu_prediction = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_prediction * mu_target
    variance_prediction = F.conv2d(prediction.square(), window, padding=padding, groups=channels) - mu_prediction_sq
    variance_target = F.conv2d(target.square(), window, padding=padding, groups=channels) - mu_target_sq
    covariance = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_cross
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2.0 * mu_cross + c1) * (2.0 * covariance + c2)) / (
        (mu_prediction_sq + mu_target_sq + c1)
        * (variance_prediction + variance_target + c2)
    ).clamp_min(eps)
    if valid is None:
        return ssim_map.mean()
    return (ssim_map * valid).sum() / (valid.sum() * channels).clamp_min(eps)


def masked_lpips(
    prediction: torch.Tensor,
    target: torch.Tensor,
    model,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """LPIPS on full images or images zeroed outside the valid region."""
    prediction = _as_batched_image(prediction)
    target = _as_batched_image(target).to(device=prediction.device, dtype=prediction.dtype)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = _as_batched_mask(mask, prediction)
    if valid is not None:
        prediction = prediction * valid
        target = target * valid
    return model(prediction * 2.0 - 1.0, target * 2.0 - 1.0).mean()


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
