# Evaluation

`evaluate_reconstruction.py` is the main reconstruction-quality entry point. It
matches prediction and reference images by filename and reports per-view and
mean PSNR, SSIM, and LPIPS. An optional mask directory restricts the evaluation
to the target object or background region; `--invert-mask` switches the region.

Example:

```bash
python Evaluation/evaluate_reconstruction.py \
  --prediction-dir path/to/renders \
  --target-dir path/to/gt \
  --mask-dir path/to/object_masks \
  --output-json metrics.json
```

Install the evaluation dependencies with:

```bash
pip install -r Evaluation/requirements.txt
```

`metrics.py` also implements two DOGS-specific protocols:

- background leakage rate, which is reported by the decoupling ablation;
- boundary F-score, which is retained as an auxiliary analysis protocol and is
  not one of the reported main quantitative results.

Boundary counts and leakage numerators are accumulated across views before the
final ratio is computed. The manuscript configuration uses an alpha threshold
of `0.5` and a boundary matching tolerance of `2` pixels.
