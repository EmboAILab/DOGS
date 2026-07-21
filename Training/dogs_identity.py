"""Identity grouping and extraction helpers for DOGS object Gaussians."""

from __future__ import annotations

from pathlib import Path

import torch


def build_identity_classifier(gaussians, num_classes: int) -> torch.nn.Module:
    """Build the lightweight classifier used for Gaussian identity features."""
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    return torch.nn.Conv2d(
        int(gaussians.num_objects),
        int(num_classes),
        kernel_size=1,
    )


def gaussian_identity_probabilities(classifier, gaussians_or_features) -> torch.Tensor:
    """Return per-Gaussian softmax probabilities with shape ``[N, C]``."""
    features = (
        gaussians_or_features.get_objects
        if hasattr(gaussians_or_features, "get_objects")
        else gaussians_or_features
    )
    if features.dim() != 3:
        raise ValueError(
            "Gaussian identity features must have shape [N, 1, D] or [N, D, 1], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[1] == 1:
        classifier_input = features.permute(2, 0, 1)
    elif features.shape[2] == 1:
        classifier_input = features.permute(1, 0, 2)
    else:
        raise ValueError(
            "one non-Gaussian identity-feature dimension must be singleton, "
            f"got {tuple(features.shape)}"
        )
    logits = classifier(classifier_input)
    if logits.dim() != 3 or logits.shape[-1] != 1:
        raise RuntimeError(f"Unexpected Gaussian identity-logit shape: {tuple(logits.shape)}")
    return torch.softmax(logits, dim=0).squeeze(-1).transpose(0, 1)


def visible_object_ids(labels, background_label: int, num_classes: int) -> list[int]:
    """Return foreground identities visible in one labeled training view."""
    ids = torch.unique(labels[labels >= 0]).tolist()
    return [
        int(value)
        for value in ids
        if int(value) != int(background_label) and int(value) < int(num_classes)
    ]


def dataset_object_ids(label_maps, background_label: int, num_classes: int) -> list[int]:
    """Return all foreground identities present across the training views."""
    ids = set()
    for label_map in label_maps:
        labels = torch.as_tensor(label_map).long()
        ids.update(int(value) for value in torch.unique(labels[labels >= 0]).tolist())
    return sorted(
        value
        for value in ids
        if value != int(background_label) and value < int(num_classes)
    )


def hard_identity_assignment(
    probabilities: torch.Tensor,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Assign each Gaussian to at most one identity using argmax plus confidence gating."""
    if probabilities.dim() != 2:
        raise ValueError("probabilities must have shape [N, C]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must satisfy 0 <= threshold <= 1")
    confidence, assignment = probabilities.max(dim=1)
    return assignment.masked_fill(confidence <= float(threshold), -1)


def hard_object_mask(
    probabilities: torch.Tensor,
    object_id: int,
    threshold: float = 0.3,
) -> torch.Tensor:
    """Select one mutually exclusive post-training Gaussian identity group."""
    if probabilities.dim() != 2:
        raise ValueError("probabilities must have shape [N, C]")
    if not 0 <= object_id < probabilities.shape[1]:
        raise IndexError(f"object_id {object_id} is outside [0, {probabilities.shape[1]})")
    assignment = hard_identity_assignment(probabilities, threshold)
    return assignment == int(object_id)


def save_gaussian_subset(gaussians, mask: torch.Tensor, output_path: str | Path) -> int:
    """Save a selected Gaussian subset without mutating the source model permanently."""
    mask = mask.to(device=gaussians.get_xyz.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != gaussians.get_xyz.shape[0]:
        raise ValueError("mask length does not match the number of Gaussians")
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("the selected object group contains no Gaussians")

    parameter_names = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
        "_objects_dc",
    )
    originals = {name: getattr(gaussians, name) for name in parameter_names}
    original_radii = getattr(gaussians, "max_radii2D", None)
    try:
        for name, value in originals.items():
            subset = value[mask].detach().clone()
            setattr(gaussians, name, torch.nn.Parameter(subset, requires_grad=False))
        if original_radii is not None and original_radii.numel() == mask.numel():
            gaussians.max_radii2D = original_radii[mask].detach().clone()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gaussians.save_ply(str(output_path))
    finally:
        for name, value in originals.items():
            setattr(gaussians, name, value)
        if original_radii is not None:
            gaussians.max_radii2D = original_radii
    return count
