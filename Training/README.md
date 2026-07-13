# Training

`train_dogs.py` contains the paper-aligned DOGS training entry point. The loss
definitions are implemented in `dogs_losses.py` and verified by tensor-level
tests in `tests/test_dogs_losses.py`.

It is designed to be placed into the compatible local Gaussian-Grouping / 3DGS training codebase. It is not standalone because it imports the renderer, scene, argument parser, loss utilities, and Gaussian model classes from that base project.

Implemented loss components:

- `L_obj-fg`: object foreground RGB supervision.
- `L_obj-alpha`: mask-outside alpha suppression.
- `L_object = L_obj-fg + lambda_alpha * L_obj-alpha`.
- `L_bg-rgb`: normalized visible-background RGB supervision plus the
  lower-weight LaMa appearance prior.
- `L_bg-reg`: mean squared local point-to-plane residual.
- `L_bg = L_bg-rgb + lambda_reg * L_bg-reg`.

Each invocation trains exactly one independently stored parameter set:

```bash
python train_dogs.py -s path/to/scene -m output/object_34 \
  --mode object --object-id 34 --dogs-iterations 10000

python train_dogs.py -s path/to/scene -m output/background \
  --mode background --inpaint-dir path/to/inpainted_frames \
  --rho 0.1 --reg-interval 5 --dogs-iterations 10000
```

Object mode obtains accumulated alpha from the renderer when available. Otherwise, `--alpha-source auto` recovers the same compositing quantity from differentiable black/white background renders. The implementation never substitutes a semantic-class probability for alpha. Background mode normalizes the observed-background and inpainted-region terms separately, then weights the latter by `rho`; if no inpainting directory is supplied, `rho` is set to zero. The local geometry term is the mean squared point-to-local-plane residual with stop-gradient neighborhood indices and normals.

The surrounding training setup provides the camera split and integer instance-label maps through `viewpoint.objects`. All invocations that will later be composed must use the same calibrated world coordinate system. The script writes `dogs_run.json` beside each output parameter set; retain it with the scene ID, camera split, mask revision, and source checksum.

Training-budget note:

- The default DOGS release configuration uses 10K iterations per independently stored parameter set.
- Upstream-default 30K runs for original 3DGS or Gaussian Grouping must be labeled as different-budget auxiliary settings, not as same-budget superiority evidence.
- All methods compared in a single table should share the same camera split, image resolution, mask/evaluation domain, and metric script.
