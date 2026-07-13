# Evaluation

`metrics.py` implements the two DOGS-specific metrics used by the ablation:

- boundary F-score from thresholded object alpha and reference-mask boundaries;
- background leakage rate from object opacity outside the target mask.

Boundary counts and leakage numerators are accumulated across views before the
final ratio is computed. The manuscript configuration uses an alpha threshold
of `0.5` and a boundary matching tolerance of `2` pixels.
