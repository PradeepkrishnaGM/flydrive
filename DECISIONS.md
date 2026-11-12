# Decisions

## Milestone 2: optic-flow looming (2026-11-12)
- Flag `--looming {blob,flow}`, default `blob`. Defaults are unchanged.
- **Option A (hybrid):** flow is evaluated only inside the existing "in the way" blob regions
  (dilated by `FLOW_MARGIN`). Mask-free ground-flow subtraction (option B) is on the later list.
- Flow is an affine Lucas-Kanade fit per object region rather than a dense flow field: a dense
  field is too noisy on blobs a few pixels wide. Expansion rate e = divergence / 2.
- Output stays in rad/s (theta * e), so existing readout/RL models need no retraining.
- Added `requirements-dev.txt` (pytest) and `eval_looming.py`.
