# Translate

This folder contains the minimal conversion workflow used to turn Gaussian PLY outputs into simulation-readable USDZ assets with a generated triangle `mesh.ply` geometric proxy.

The current workflow targets USD/USDZ assets for Isaac Sim or Omniverse-style simulation pipelines. It is not a direct PLY-to-URDF converter; URDF wrapping and robot-scene registration are handled by the downstream simulation project. The output supports asset-interface inspection, not automatic physical-parameter calibration.

## Main Entry Point

```bash
python Translate/run_3dgrut_usdz_collision_pipeline.py path/to/point_cloud.ply --out-dir path/to/export_dir
```

The pipeline performs four steps:

1. Filter/crop the Gaussian PLY while preserving Gaussian attributes.
2. Convert the visual Gaussian PLY to USDZ by invoking upstream 3DGRUT:
   `python -m threedgrut.export.scripts.ply_to_usd`
3. Generate a triangle `mesh.ply` from the Gaussian PLY for geometric-proxy or collision-proxy inspection.
4. Invoke the installed upstream
   `threedgrut.export.scripts.add_mesh_to_usdz` module to add `mesh.ply`
   to the USDZ and enable collision metadata when the target simulator needs
   a proxy carrier.

After conversion, check asset loading, scene placement, scale consistency, package validity, and collision-proxy availability in the target simulator. Contact behavior, mass/inertia parameters, and controller behavior remain downstream simulator settings.

## Outputs

By default, outputs are written next to the input PLY:

```text
point_cloud_filtered.ply
point_cloud.usdz
point_cloud_mesh.ply
point_cloudoutput.usdz
```

Use `--preserve-visual` when the original PLY should be used for appearance while the filtered PLY is used only for collision mesh generation.

## Required Environment

The local scripts require:

```text
numpy
scipy
scikit-image
open3d
trimesh
plyfile
usd-core
```

The visual USDZ export and mesh-injection steps require upstream 3DGRUT to be
installed and importable as `threedgrut`. The upstream mesh-injection source is
not redistributed in this repository.

## Included Files

- `run_3dgrut_usdz_collision_pipeline.py`: one-shot conversion pipeline.
- `crop_gaussian_ply.py`: Gaussian PLY filtering/cropping.
- `gaussian_ply_to_3dgrut_mesh_ply.py`: generated triangle `mesh.ply` writer.
- `gaussian_ply_to_solid_textured_mesh.py`: point-to-volume and marching-cubes mesh generation.
- `gaussian_ply_to_textured_mesh.py`: point filtering and mesh post-processing helpers.
- `gaussian_ply_to_splat_mesh.py`: Gaussian PLY bounds and point utilities.
- `ply_to_mesh.py`: Gaussian PLY parser and spherical-harmonic color helpers.
The paper-facing path is `run_3dgrut_usdz_collision_pipeline.py`. Other mesh helpers are retained because the geometric-proxy generator depends on their parsing, filtering, and mesh-processing utilities.
