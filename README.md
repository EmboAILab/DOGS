# DOGS: Decoupled Object Gaussian Splatting

This repository is the paper-aligned minimal code release for:

**DOGS: Decoupled Object Gaussian Splatting for Composable Real-Scene Reconstruction and Reuse**

DOGS targets object-level decoupling of real scenes, independent background recovery, parameter-level scene composition, and visual asset-interface checks. The code release is scoped to those representation and conversion components; it does not claim dynamics modeling, collision fidelity, closed-loop control, or robot task-performance validation.

Project page: https://emboailab.github.io/DOGS/

The project page and source release omit author biographies, personal contact details, and local machine paths. This author-neutral release policy is independent of the journal's single-anonymized manuscript review model.

The release is intentionally source-focused. It does not include full upstream repositories, datasets, model weights, checkpoints, rendered images, videos, PLY point clouds, USD/USDZ/FBX/GLB assets, or experiment outputs. Paper-related large files are listed by category in `Assets/ASSET_MANIFEST.csv` and should be distributed separately.

## Repository Contents

- `Training/train_dogs.py`
  - DOGS training entry point adapted from the Gaussian-Grouping / 3DGS workflow.
  - Implements normalized object foreground supervision, mask-outside alpha suppression, observed/inpainted background RGB supervision, and local plane regularization.
  - Trains one independently stored object subspace or the background model per invocation; it does not optimize a shared semantic Gaussian field.
  - Uses the surrounding Gaussian-Grouping/3DGS renderer, scene classes, camera loader, and mask fields. Place it into the compatible base training codebase before running.

- `Evaluation/`
  - Implements the boundary F-score and background leakage rate used by the decoupling ablation.
  - Aggregates metric counts over test views using the manuscript threshold and boundary tolerance.

- `Translate/`
  - Minimal Gaussian PLY to simulation asset conversion workflow.
  - Filters Gaussian PLY files, generates a triangle `mesh.ply` proxy/collision carrier, invokes upstream 3DGRUT for visual USDZ export, and injects the mesh into USDZ with collision metadata.
  - Only the conversion scripts and their direct local dependencies are included. The full 3DGRUT project is not vendored.

- `Simulation/`
  - Interface notes for checking converted DOGS assets in Isaac Sim / Isaac Lab-style scenes.
  - The full simulation platform, large robot assets, and generated USD/USDZ files are not vendored.

- `Assets/`
  - Manifest and restore instructions for paper-related large files that are intentionally excluded from GitHub.

- `RELEASE_POLICY.md`
  - Release policy for keeping the repository minimal and reviewable.

## Paper Scope And Evidence

DOGS focuses on object-level decoupled modeling of real captured scenes, independent background recovery, parameter-level scene composition, and conversion of Gaussian assets into simulation-readable visual assets. The code release therefore covers:

1. object Gaussian subspace training with foreground supervision and mask-outside alpha suppression;
2. background Gaussian modeling with RGB supervision and local geometry regularization;
3. parameter-level scene composition in a shared world coordinate frame;
4. Gaussian-to-asset conversion through mesh/collision-proxy and USD/USDZ carriers;
5. selected Isaac Sim / Isaac Lab interface checks for visibility, placement, and robot-scene coexistence.

The simulation-facing evidence is an asset-interface check. Collision accuracy, mass/inertia parameters, contact stability, and task-level robot behavior require simulator-side calibration outside DOGS.

## External Dependencies

The training script depends on a compatible Gaussian-Grouping/3DGS-style codebase.

The conversion workflow depends on an installed upstream 3DGRUT environment for:

```bash
python -m threedgrut.export.scripts.ply_to_usd
```

Install upstream 3DGRUT separately from:

https://github.com/nv-tlabs/3dgrut

Python packages used by the local conversion scripts include `numpy`, `scipy`, `scikit-image`, `open3d`, `trimesh`, `plyfile`, and `usd-core`.

## Large Assets and Demo Media

Large files are excluded from GitHub. Restore paper-related assets according to:

- `Assets/README.md`
- `Assets/ASSET_MANIFEST.csv`

The external package should preserve only paper-relevant paths, including:

```text
results/fallBox/
results/desk/
results/fallAstr/
results/fallCattle/
code/GaussianModel/Tracking-Anything-with-DEVA/saves/
code/GaussianModel/lama/big-lama/
code/GaussianModel/lama/hub/checkpoints/
code/GaussianModel/checkpoint/
code/leisaac/assets/dogs/
```

Do not add unrelated local demos, full upstream repositories, private documents, Word manuscript files, rendered paper drafts, checkpoints, generated Gaussian outputs, USD/USDZ/FBX/GLB files, or general experiment dumps to this Git repository.

## Evaluation Scope

The manuscript evaluates object and background reconstruction, composed-scene
rendering, boundary purity, opacity leakage, and selected asset-interface checks.
The simulation-facing evidence remains limited to visual assets, coordinate and
scale consistency, hierarchy, and geometric-proxy availability.

## 3DGRUT Attribution

The conversion workflow in `Translate/` includes code adapted from NVIDIA's 3DGRUT project, especially the USDZ mesh-injection utility.

Original project:

https://github.com/nv-tlabs/3dgrut

The adapted 3DGRUT-derived code is distributed under the Apache License 2.0. See:

```text
Translate/third_party/3dgrut/LICENSE
```

If this conversion workflow is used in a publication, cite the upstream 3DGRUT-related works listed in `Translate/THIRD_PARTY_NOTICES.md` in addition to citing DOGS.
