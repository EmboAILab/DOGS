# Large Asset Package

Large files and demo media are not stored in GitHub. They should be distributed through an external package and restored locally when needed.

Recommended package name:

```text
DOGS-paper-assets
```

No cloud-drive URL is committed in this source release. Keep access links in the project release notes or repository release page.

The external package should preserve the restore paths listed in `ASSET_MANIFEST.csv`. Restore assets by copying the package contents into the local working tree that contains the full upstream dependencies and local experiment folders.

Important path groups:

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

`ASSET_MANIFEST.csv` records restore categories. `PAPER_MEDIA_MANIFEST.csv` records the paper-facing still/video files individually with relative paths, byte counts, and SHA-256 checksums. `FIGURE_SOURCE_MANIFEST.csv` maps every submitted figure or panel to its manual diagram source or real experimental record. The `results/` groups correspond to real capture, object-level decoupling, background recovery, object repositioning, and the selected USDZ visual-import records displayed in Isaac Sim. These records do not by themselves verify URDF or Isaac Lab compatibility, scene hierarchy, collision-proxy loading, or numerical scale accuracy.

Do not upload manuscript files, full upstream repositories, unrelated local demos, generated experiment folders, model weights, point clouds, USD/USDZ/FBX/GLB files, videos, or rendered images to GitHub.
