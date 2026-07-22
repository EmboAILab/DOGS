"""Evaluate DOGS renders with PSNR, SSIM, and LPIPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from metrics import masked_lpips, masked_psnr, masked_ssim


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def image_files(directory: Path) -> dict[str, Path]:
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not files:
        raise ValueError(f"no supported images found in {directory}")
    return files


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device=device)


def load_mask(path: Path, device: torch.device, invert: bool) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    mask = torch.from_numpy(array).to(device=device).clamp(0.0, 1.0)
    return 1.0 - mask if invert else mask


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def evaluate(args: argparse.Namespace) -> dict:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("install Evaluation/requirements.txt before running LPIPS") from exc

    device = resolve_device(args.device)
    predictions = image_files(args.prediction_dir)
    targets = image_files(args.target_dir)
    names = sorted(set(predictions) & set(targets))
    if not names:
        raise ValueError("prediction and target directories have no matching filenames")
    missing_predictions = sorted(set(targets) - set(predictions))
    missing_targets = sorted(set(predictions) - set(targets))
    if missing_predictions or missing_targets:
        raise ValueError(
            f"unmatched files: missing predictions={missing_predictions}, missing targets={missing_targets}"
        )

    masks = image_files(args.mask_dir) if args.mask_dir else None
    if masks is not None and set(names) - set(masks):
        raise ValueError(f"missing masks for: {sorted(set(names) - set(masks))}")

    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
    per_view = {}
    with torch.no_grad():
        for name in names:
            prediction = load_rgb(predictions[name], device)
            target = load_rgb(targets[name], device)
            if prediction.shape != target.shape:
                raise ValueError(f"shape mismatch for {name}: {prediction.shape} vs {target.shape}")
            mask = load_mask(masks[name], device, args.invert_mask) if masks else None
            per_view[name] = {
                "PSNR": float(masked_psnr(prediction, target, mask).item()),
                "SSIM": float(masked_ssim(prediction, target, mask).item()),
                "LPIPS": float(masked_lpips(prediction, target, lpips_model, mask).item()),
            }

    means = {
        metric: float(sum(record[metric] for record in per_view.values()) / len(per_view))
        for metric in ("PSNR", "SSIM", "LPIPS")
    }
    return {
        "protocol": {
            "prediction_dir": str(args.prediction_dir),
            "target_dir": str(args.target_dir),
            "mask_dir": str(args.mask_dir) if args.mask_dir else None,
            "invert_mask": bool(args.invert_mask),
            "lpips_network": args.lpips_net,
            "aggregation": "arithmetic mean over matched test views",
        },
        "view_count": len(per_view),
        "mean": means,
        "per_view": per_view,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--invert-mask", action="store_true")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="vgg")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
