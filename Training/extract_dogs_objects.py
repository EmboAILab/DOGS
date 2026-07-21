"""Repeatedly extract DOGS object Gaussian subspaces from one grouped model."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import torch

from arguments import ModelParams, get_combined_args
from scene import GaussianModel, Scene

from dogs_identity import (
    build_identity_classifier,
    gaussian_identity_probabilities,
    hard_object_mask,
    save_gaussian_subset,
)


def main() -> None:
    parser = ArgumentParser(description="Extract DOGS object groups from one trained model")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--object-ids", nargs="+", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path)
    args = get_combined_args(parser)

    gaussians = GaussianModel(model.extract(args).sh_degree)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    classifier = build_identity_classifier(gaussians, model.extract(args).num_classes).cuda()
    classifier_path = os.path.join(
        model.extract(args).model_path,
        "point_cloud",
        f"iteration_{scene.loaded_iter}",
        "classifier.pth",
    )
    classifier.load_state_dict(torch.load(classifier_path, map_location="cuda"))
    classifier.eval()

    output_dir = args.output_dir or Path(model.extract(args).model_path) / "objects"
    records = []
    with torch.no_grad():
        probabilities = gaussian_identity_probabilities(classifier, gaussians)
        for object_id in args.object_ids:
            mask = hard_object_mask(probabilities, object_id, args.threshold)
            output_path = output_dir / f"object_{object_id}" / "point_cloud.ply"
            count = save_gaussian_subset(gaussians, mask, output_path)
            records.append(
                {
                    "object_id": int(object_id),
                    "probability_threshold": float(args.threshold),
                    "gaussian_count": count,
                    "path": str(output_path),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "extraction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_iteration": int(scene.loaded_iter),
                "classifier": classifier_path,
                "objects": records,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
