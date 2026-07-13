"""DOGS training entry point for one independently stored Gaussian parameter set.

This file is intended to run inside the Gaussian-Grouping/3DGS codebase used by
the project.  Each invocation trains either one object Gaussian subspace or the
background Gaussian model.  The two modes never share a Gaussian parameter set.
"""

from __future__ import annotations

import json
import os
import random
import sys
import uuid
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randint

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state

from dogs_losses import (
    background_rgb_loss,
    local_plane_residual_loss,
    mask_outside_alpha_loss,
    object_foreground_loss,
)

try:
    import wandb
except ImportError:  # Optional unless --use-wandb is selected.
    wandb = None


def renderer_alpha(render_pkg):
    """Return an explicit accumulated-alpha map when the renderer exposes one."""
    for key in (
        "render_alpha",
        "rendered_alpha",
        "alpha",
        "render_object_alpha",
        "object_alpha",
        "alpha_object",
        "render_alpha_object",
    ):
        if key in render_pkg:
            return render_pkg[key]
    return None


def dual_background_alpha(viewpoint, gaussians, pipe):
    """Recover accumulated alpha from black/white background renders."""
    device = gaussians.get_xyz.device
    black = torch.zeros(3, dtype=torch.float32, device=device)
    white = torch.ones(3, dtype=torch.float32, device=device)
    black_image = render(viewpoint, gaussians, pipe, black)["render"]
    white_image = render(viewpoint, gaussians, pipe, white)["render"]
    transmittance = (white_image - black_image).mean(dim=0)
    return (1.0 - transmittance).clamp(0.0, 1.0)


def resolve_alpha(args, viewpoint, gaussians, pipe, render_pkg):
    if args.alpha_source == "dual-render":
        return dual_background_alpha(viewpoint, gaussians, pipe)
    alpha = renderer_alpha(render_pkg)
    if alpha is not None:
        return alpha
    if args.alpha_source == "renderer":
        raise RuntimeError(
            "The selected renderer does not expose accumulated alpha. "
            "Use --alpha-source dual-render or add an alpha output to the renderer."
        )
    return dual_background_alpha(viewpoint, gaussians, pipe)


def camera_mask(viewpoint, args):
    labels = viewpoint.objects.to(device="cuda").long()
    valid = labels >= 0
    if args.mode == "object":
        return valid & (labels == args.object_id), valid
    return valid & (labels == args.background_label), valid


def find_inpaint_path(root, image_name):
    if root is None:
        return None
    base = Path(str(image_name))
    candidates = [root / base]
    if not base.suffix:
        candidates.extend(root / f"{base.name}{suffix}" for suffix in (".png", ".jpg", ".jpeg"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No inpainted target found for view '{image_name}' in {root}")


def load_rgb_image(path, reference):
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).to(device=reference.device, dtype=reference.dtype)
    if tensor.shape != reference.shape:
        raise ValueError(
            f"Inpainted target shape {tuple(tensor.shape)} does not match RGB input "
            f"shape {tuple(reference.shape)} for {path}"
        )
    return tensor


def background_prior(viewpoint, args, gt_image, bg_mask, valid_mask, cache):
    if args.inpaint_dir is None:
        return None, None

    image_name = str(viewpoint.image_name)
    if image_name not in cache:
        path = find_inpaint_path(args.inpaint_dir, image_name)
        cache[image_name] = load_rgb_image(path, gt_image)
    inpainted = cache[image_name]

    object_union_mask = valid_mask & ~bg_mask
    return inpainted, object_union_mask


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    args,
):
    first_iter = 0
    prepare_output_and_logger(dataset, args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if args.mode == "object":
        object_is_present = any(
            bool((camera.objects.long() == args.object_id).any()) for camera in scene.getTrainCameras()
        )
        if not object_is_present:
            raise ValueError(f"Object ID {args.object_id} is absent from every training view")

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    inpaint_cache = {}
    ema_loss = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc=f"DOGS {args.mode} training")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()

        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifier,
                ) = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, 0, 1) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (iteration < int(opt.iterations) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint.original_image.to(device="cuda")
        domain_mask, valid_mask = camera_mask(viewpoint, args)

        zero = image.new_zeros(())
        loss_obj_fg = zero
        loss_obj_alpha = zero
        loss_bg_rgb = zero
        loss_bg_reg = zero

        if args.mode == "object":
            loss_obj_fg = object_foreground_loss(image, gt_image, domain_mask)
            alpha = resolve_alpha(args, viewpoint, gaussians, pipe, render_pkg)
            outside_mask = valid_mask & ~domain_mask
            loss_obj_alpha = mask_outside_alpha_loss(alpha, outside_mask)
            loss = loss_obj_fg + args.lambda_alpha * loss_obj_alpha
        else:
            inpainted, object_union_mask = background_prior(
                viewpoint, args, gt_image, domain_mask, valid_mask, inpaint_cache
            )
            loss_bg_rgb = background_rgb_loss(
                image,
                gt_image,
                domain_mask,
                inpainted=inpainted,
                object_union_mask=object_union_mask,
                rho=args.rho,
            )
            if iteration % args.reg_interval == 0:
                loss_bg_reg = local_plane_residual_loss(
                    gaussians.get_xyz,
                    k_nn=args.knn_k,
                    max_points=args.reg_max_points,
                )
            loss = loss_bg_rgb + args.lambda_reg * loss_bg_reg

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"loss": f"{ema_loss:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                iteration,
                loss,
                iter_start.elapsed_time(iter_end),
                {
                    "loss_obj_fg": loss_obj_fg,
                    "loss_obj_alpha": loss_obj_alpha,
                    "loss_bg_rgb": loss_bg_rgb,
                    "loss_bg_reg": loss_bg_reg,
                },
                args,
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving {args.mode} Gaussian parameter set")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                visibility_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                viewspace_points = render_pkg["viewspace_points"]
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_points, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                checkpoint_path = os.path.join(scene.model_path, f"chkpnt{iteration}.pth")
                torch.save((gaussians.capture(), iteration), checkpoint_path)


def prepare_output_and_logger(dataset, args):
    if not dataset.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        dataset.model_path = os.path.join("./output", unique_str[:10])

    os.makedirs(dataset.model_path, exist_ok=True)
    print(f"Output folder: {dataset.model_path}")
    with open(os.path.join(dataset.model_path, "cfg_args"), "w", encoding="utf-8") as handle:
        handle.write(str(Namespace(**vars(dataset))))

    run_record = {
        "mode": args.mode,
        "object_id": args.object_id,
        "background_label": args.background_label,
        "iterations": args.dogs_iterations,
        "seed": args.seed,
        "lambda_alpha": args.lambda_alpha,
        "lambda_reg": args.lambda_reg,
        "rho": args.rho if args.inpaint_dir else 0.0,
        "inpaint_dir_provided": args.inpaint_dir is not None,
        "knn_k": args.knn_k,
        "reg_interval": args.reg_interval,
        "reg_max_points": args.reg_max_points,
        "alpha_source": args.alpha_source,
    }
    with open(os.path.join(dataset.model_path, "dogs_run.json"), "w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2)


def training_report(iteration, total_loss, elapsed, components, args):
    if not args.use_wandb:
        return
    log = {
        "train/total_loss": total_loss.item(),
        "train/iteration_ms": elapsed,
        "train/iteration": iteration,
    }
    for name, value in components.items():
        log[f"train/{name}"] = value.item()
    wandb.log(log)


def apply_config(args):
    if args.config_file is None:
        return args
    with open(args.config_file, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    allowed = {
        "mode",
        "object_id",
        "background_label",
        "dogs_iterations",
        "seed",
        "lambda_alpha",
        "lambda_reg",
        "inpaint_dir",
        "rho",
        "knn_k",
        "reg_interval",
        "reg_max_points",
        "alpha_source",
        "densify_until_iter",
        "num_classes",
        "reg3d_interval",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unsupported DOGS config keys: {', '.join(unknown)}")
    for key, value in config.items():
        setattr(args, key, value)
    if args.inpaint_dir is not None:
        args.inpaint_dir = Path(args.inpaint_dir)
    return args


if __name__ == "__main__":
    parser = ArgumentParser(description="Train one DOGS object subspace or the DOGS background model")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--mode", choices=("object", "background"), required=True)
    parser.add_argument("--object-id", type=int)
    parser.add_argument("--background-label", type=int, default=0)
    parser.add_argument("--dogs-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-alpha", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=0.5)
    parser.add_argument("--inpaint-dir", type=Path)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--reg-interval", type=int, default=5)
    parser.add_argument("--reg-max-points", type=int, default=2048)
    parser.add_argument("--alpha-source", choices=("auto", "renderer", "dual-render"), default="auto")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug-from", type=int, default=-1)
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--test-iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save-iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint-iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start-checkpoint", type=str)
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--use-wandb", action="store_true")

    args = apply_config(parser.parse_args(sys.argv[1:]))
    if args.mode == "object" and args.object_id is None:
        parser.error("--object-id is required when --mode object is selected")
    if not 0.0 <= args.rho < 1.0:
        parser.error("--rho must satisfy 0 <= rho < 1")
    if args.use_wandb and wandb is None:
        parser.error("wandb is not installed; remove --use-wandb or install wandb")

    args.iterations = args.dogs_iterations
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    if args.inpaint_dir is None:
        args.rho = 0.0

    safe_state(args.quiet)
    set_random_seed(args.seed)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.use_wandb:
        wandb.init(project="dogs-object-gaussian")
        wandb.config.update(vars(args))
        wandb.run.name = args.model_path

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args,
    )
    print("\nDOGS training complete.")
