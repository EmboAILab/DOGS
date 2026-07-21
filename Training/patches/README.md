# Gaussian renderer integration

DOGS renders one object identity by multiplying each Gaussian opacity by the
corresponding softmax identity probability. Apply the included patch from the
root of the compatible Gaussian-Grouping/3DGS codebase before running
`train_dogs.py`:

```bash
git apply path/to/DOGS/Training/patches/gaussian_renderer_opacity_modifier.patch
```

The patch adds the optional `opacity_modifier` argument without changing the
default whole-scene rendering behavior.
