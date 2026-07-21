# Training

`train_dogs.py` contains the paper-aligned DOGS training entry point. The loss
definitions are implemented in `dogs_losses.py`, and the Gaussian identity
grouping and extraction rules are implemented in `dogs_identity.py`. Tensor-level
tests are provided in `tests/`.

It is designed to be placed into a compatible Gaussian-Grouping / 3DGS training
codebase. It imports the renderer, scene, argument parser, camera loader, and
Gaussian model classes from that base project. Apply the renderer integration
patch in `patches/` before training so object identity probabilities can modulate
per-Gaussian opacity.

Implemented loss components:

- `L_obj-fg`: object foreground RGB supervision.
- `L_obj-alpha`: mask-outside alpha suppression.
- `L_object = L_obj-fg + lambda_alpha * L_obj-alpha`.
- `L_bg-rgb`: normalized visible-background RGB supervision plus the
  lower-weight LaMa appearance prior.
- `L_bg-reg`: mean squared local point-to-plane residual.
- `L_bg = L_bg-rgb + lambda_reg * L_bg-reg`.

Object-aware parameter groups are learned together in one run. The background is
optimized as a separate aligned Gaussian model:

```bash
python train_dogs.py -s path/to/scene -m output/grouped_objects \
  --mode objects --num_classes 256 --dogs-iterations 10000

python train_dogs.py -s path/to/scene -m output/background \
  --mode background --inpaint-dir path/to/inpainted_frames \
  --rho 0.1 --reg-interval 5 --dogs-iterations 10000
```

In `objects` mode, a lightweight classifier maps each Gaussian identity feature
to a softmax distribution. Every scene object is rendered in each uniformly
sampled view, including views where that object is absent. The implementation
precomputes the fixed all-view pixel counts in Eqs. (1), (2), and (6), then scales
each sampled-view numerator by the number of training views. Averaging these
sampled estimates recovers the all-view normalized objectives, and object terms
are averaged over the complete scene-level object-ID set. The grouped checkpoint
and `classifier.pth` are saved together.

After training, any number of object assets can be extracted from that one model
without retraining:

```bash
python extract_dogs_objects.py -m output/grouped_objects --iteration 10000 \
  --object-ids 34 47 52 --threshold 0.3
```

The extraction command first assigns each Gaussian to its maximum-probability
identity and then rejects assignments whose confidence does not exceed the
threshold. This argmax-plus-threshold rule makes exported object groups mutually
exclusive. It writes one PLY per selected identity plus an
`extraction_manifest.json` containing the grouping rule, threshold, and Gaussian
counts.

Object mode obtains accumulated alpha from the renderer when available.
Otherwise, `--alpha-source auto` recovers the same compositing quantity from
differentiable black/white background renders. The softmax identity probability
modulates Gaussian opacity but is never substituted for accumulated alpha.
Background mode normalizes observed-background and inpainted-region terms
separately, then weights the latter by `rho`. If no inpainting directory is
supplied, `rho` is zero. The local geometry term is the mean squared
point-to-local-plane residual with stop-gradient neighborhood indices and normals.

The surrounding training setup provides the camera split and integer
instance-label maps through `viewpoint.objects`. The grouped object model and
background model must use the same calibrated world coordinate system. The
script writes `dogs_run.json` beside each output parameter set.

Training-budget note:

- The default DOGS release configuration uses one 10K multi-object grouping run
  and one 10K aligned-background run.
- Upstream-default 30K runs for original 3DGS or Gaussian Grouping must be labeled as different-budget auxiliary settings, not as same-budget superiority evidence.
- All methods compared in a single table should share the same camera split, image resolution, mask/evaluation domain, and metric script.
