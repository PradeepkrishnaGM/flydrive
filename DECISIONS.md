# Decisions

## Milestone 2: optic-flow looming (2026-11-12)
- Flag `--looming {blob,flow}`, default `blob`. Defaults are unchanged.
- **Option A (hybrid):** flow is evaluated only inside the existing "in the way" blob regions
  (dilated by `FLOW_MARGIN`). Mask-free ground-flow subtraction (option B) is on the later list.
- Flow is an affine Lucas-Kanade fit per object region rather than a dense flow field: a dense
  field is too noisy on blobs a few pixels wide. Expansion rate e = divergence / 2.
- Output stays in rad/s (theta * e), so existing readout/RL models need no retraining.
- Added `requirements-dev.txt` (pytest) and `eval_looming.py`.

## Milestone 4: lesion experiment (2026-11-12)
- A lesion silences neurons (reset to rest every step, so they never spike, even if driven)
  rather than editing the synapse matrix: trivially reversible, no copy of 15M synapses.
  `reset()` keeps the lesion; `unlesion()` removes it.
- The reflex is built on the intact brain and the brain is lesioned afterwards, so the
  hand-tuned reflex's start-up calibration can't adapt to the damage.
- Cache version 3 stores each neuron's cell type and side (rebuilds once, ~1 minute).
- Random controls (10 and 162 neurons, 5 draws) are never taken from the looming/escape groups.
- The brain's noise is reseeded per drive, so results don't depend on process scheduling.
- Design had 17 lesions (my plan said 12: it didn't count the 10 random draws): 102 drives, ~30 min.
- Added matplotlib to requirements-dev.txt (only for the chart).
