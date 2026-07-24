# Third-Party Notices

## NVIDIA 3DGRUT

The conversion workflow invokes NVIDIA's separately installed 3DGRUT package:

https://github.com/nv-tlabs/3dgrut

The repository does not redistribute 3DGRUT's USDZ export or mesh-injection
source. The local pipeline calls
`threedgrut.export.scripts.ply_to_usd` and
`threedgrut.export.scripts.add_mesh_to_usdz` from the user's upstream
installation. The full upstream project, training code, tests, datasets, and
third-party dependencies are not vendored here.

The upstream project is distributed under the Apache License 2.0. A copy is
provided at `third_party/3dgrut/LICENSE` for dependency attribution. Individual
upstream files remain subject to any file-specific notices in the installed
3DGRUT distribution.

## Upstream Citations

```bibtex
@article{loccoz20243dgrt,
    author = {Nicolas Moenne-Loccoz and Ashkan Mirzaei and Or Perel and Riccardo de Lutio and Janick Martinez Esturo and Gavriel State and Sanja Fidler and Nicholas Sharp and Zan Gojcic},
    title = {3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes},
    journal = {ACM Transactions on Graphics and SIGGRAPH Asia},
    year = {2024},
}
```
```bibtex
@article{wu20253dgut,
    title = {3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting},
    author = {Wu, Qi and Martinez Esturo, Janick and Mirzaei, Ashkan and Moenne-Loccoz, Nicolas and Gojcic, Zan},
    journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2025}
}
```
