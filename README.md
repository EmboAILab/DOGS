# DOGS: Decoupled Object Gaussian Splatting

This repository is the paper-aligned minimal code release for:

**DOGS: Decoupled Object Gaussian Splatting for Composable Real-Scene Reconstruction and Reuse**

DOGS addresses object-level decoupled reconstruction of real scenes, independent background recovery, and direct parameter-level recomposition of the resulting assets for constructing editable robot-simulation scenes. Selected USDZ visual imports and placement checks in Isaac Sim provide interface-level validation. The code release is scoped to those representation and conversion components; it does not claim dynamics modeling, collision fidelity, closed-loop control, or robot task-performance validation.

Project page: https://emboailab.github.io/DOGS/

The project page and source release omit author biographies, personal contact details, and local machine paths. This author-neutral release policy is independent of the journal's single-anonymized manuscript review model.

The source history is intentionally code-focused. It does not include full upstream repositories, source-resolution datasets, original recordings, model weights, checkpoints, PLY point clouds, USD/USDZ/FBX/GLB assets, or experiment outputs. The project page uses only downsampled, metadata-stripped presentation derivatives. A selected, metadata-sanitized subset of the self-collected RGB data is published separately as a [GitHub Release](https://github.com/EmboAILab/DOGS/releases/tag/data-v1.0.0), without adding the source images to Git history. `Assets/DATA_AVAILABILITY.md` defines the released scope, while `Assets/ASSET_MANIFEST.csv` remains a category-level placement reference.

## Repository Contents

- `Training/train_dogs.py`
  - DOGS training entry point adapted from the Gaussian-Grouping / 3DGS workflow.
  - Implements normalized object foreground supervision, mask-outside alpha suppression, observed/inpainted background RGB supervision, and local plane regularization.
  - Learns all object-aware Gaussian parameter groups in one multi-object run and trains the aligned background as a separate model.
  - Uses fixed all-view pixel normalizers with uniformly sampled-view numerators, matching Eqs. (1), (2), and (6) while averaging object terms over all scene object IDs.
  - Uses the surrounding Gaussian-Grouping/3DGS renderer, scene classes, camera loader, and mask fields. Apply `Training/patches/gaussian_renderer_opacity_modifier.patch` in the compatible base codebase before running.

- `Training/extract_dogs_objects.py`
  - Reuses the grouped checkpoint and classifier to extract multiple object Gaussian subspaces without retraining.
  - Applies argmax identity assignment followed by the paper's confidence threshold of 0.3, producing mutually exclusive exported groups and recording each object's Gaussian count.

- `Evaluation/`
  - Provides the PSNR, SSIM, and LPIPS reconstruction entry used for the paper's main image-quality protocol.
  - Implements the reported background leakage rate and the auxiliary, non-reported boundary F-score protocol.

- `Composition/`
  - Directly concatenates compatible background and object Gaussian PLY parameter records in their shared world coordinate system to construct editable robot-simulation scenes.
  - Applies no translation, rotation, or scale transform and writes a SHA-256/count manifest for the composed scene.

- `Translate/`
  - Minimal Gaussian PLY to simulation asset conversion workflow.
  - Filters Gaussian PLY files, generates a triangle `mesh.ply` proxy/collision carrier, and invokes an installed upstream 3DGRUT package for visual USDZ export and mesh injection.
  - Only DOGS-owned conversion scripts and their direct local dependencies are included. The full 3DGRUT project and its mesh-injection source are not vendored.

- `Simulation/`
  - Documentation for checking converted DOGS assets in constructed robot-simulation scenes; this folder does not contain simulator-side executable code.
  - The full simulation platform, large robot assets, and generated USD/USDZ files are not vendored.

- `Assets/`
  - Data-release scope, category-level restore policy, and figure-level provenance for files that are intentionally excluded from Git history.

- `RELEASE_POLICY.md`
  - Release policy for keeping the repository minimal and reviewable.

## Paper Scope And Evidence

DOGS focuses on object-level decoupled reconstruction of real captured scenes, independent background recovery, and direct recomposition of object Gaussian subspaces with the aligned background model to construct editable robot-simulation scenes. The code release therefore covers:

1. object Gaussian subspace training with foreground supervision and mask-outside alpha suppression;
2. background Gaussian modeling with RGB supervision and local geometry regularization;
3. executable direct parameter-level recomposition for editable robot-simulation scene construction in a shared world coordinate frame;
4. Gaussian-to-asset conversion through mesh/collision-proxy and USD/USDZ carriers;
5. documentation corresponding to the selected USDZ visual-import evidence shown in Isaac Sim.

The resulting object and background Gaussian assets are assembled into robot-simulation scenes as reusable visual assets. The project page presents source-derived virtual-scene, robot-approach, and interaction frames for four object categories, together with a recorded robot-arm interaction sequence in one reconstructed scene. The displayed interface-level evidence covers USDZ visual import, object placement, and robot-scene coexistence in Isaac Sim. It does not establish URDF or Isaac Lab compatibility, hierarchy validation, collision-proxy loading, numerical scale accuracy, dynamics, or task-level behavior.

## External Dependencies

The training script depends on a compatible Gaussian-Grouping/3DGS-style codebase.

The conversion workflow depends on an installed upstream 3DGRUT environment for:

```bash
python -m threedgrut.export.scripts.ply_to_usd
python -m threedgrut.export.scripts.add_mesh_to_usdz --help
```

Install upstream 3DGRUT separately from:

https://github.com/nv-tlabs/3dgrut

Python packages used by the local conversion scripts include `numpy`, `scipy`, `scikit-image`, `open3d`, `trimesh`, `plyfile`, and `usd-core`.

## Large Assets and Demo Media

Source-resolution media and large binary assets are not committed to the public source history. The following files document the separate data release, category-level local paths, and figure provenance:

- `Assets/README.md`
- `Assets/ASSET_MANIFEST.csv`
- `Assets/DATA_AVAILABILITY.md`

The [selected partial dataset release](https://github.com/EmboAILab/DOGS/releases/tag/data-v1.0.0) contains 627 metadata-sanitized RGB images in nine groups. It covers one multi-object scene, four object-reference captures, empty-background views, and the additional `combination`, `guo`, and `hu` sequences. This release is only a selected subset of the self-collected data. Most benchmark experiments use public [Mip-NeRF360](https://jonbarron.info/mipnerf360/), [LLFF](https://github.com/Fyusion/LLFF), and [LERF-Mask](https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md) data, which are not mirrored here.

The public manifests deliberately omit raw-media filenames, experiment-directory names, episode identifiers, file sizes, and source hashes. Do not add unrelated local demos, full upstream repositories, private documents, Word manuscript files, source-resolution media, checkpoints, generated Gaussian outputs, USD/USDZ/FBX/GLB files, or general experiment dumps to this Git repository.

## Evaluation Scope

The manuscript evaluates object and background reconstruction, composed-scene
rendering, opacity leakage, an auxiliary boundary protocol, and selected visual
asset-interface evidence. The extracted object subspaces and aligned background
are directly recomposed as reusable visual assets for constructing editable
robot-simulation scenes. The displayed interface-level evidence is limited to
USDZ visual import, object placement, and robot-scene coexistence in Isaac Sim.

## 3DGRUT Attribution

The conversion workflow in `Translate/` calls an installed NVIDIA 3DGRUT
package for USDZ export and mesh injection. The corresponding upstream source
is not redistributed in this repository.

Original project:

https://github.com/nv-tlabs/3dgrut

The upstream 3DGRUT project is distributed under the Apache License 2.0. See:

```text
Translate/third_party/3dgrut/LICENSE
```

If this conversion workflow is used in a publication, cite the upstream 3DGRUT-related works listed in `Translate/THIRD_PARTY_NOTICES.md` in addition to citing DOGS.
