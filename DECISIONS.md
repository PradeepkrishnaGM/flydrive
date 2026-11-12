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

## Plan for milestones 5-7 (approved 2026-11-12)
Order 5 -> 6 -> 7; milestone 6's long runs go in the background while 7 is built.
- **M5 interneuron screen:** 50 most active non-input/non-output neurons under 60 Hz looming;
  single-neuron lesions scored on (a) escape outputs (GF + 8 escape DNs) and (b) the whole
  1,299-neuron descending population, 3 noise seeds each; cumulative top-1/5/10/20; behaviour
  test top-5/top-10 vs size-matched random sets from the active-but-low-effect pool (3 draws),
  both reflexes, 3 layouts x 60 s. `run_screen.py`, `docs/screen_results.csv`, chart, README.
- **M6 camera RL retraining:** train on **flow** looming (not blob). Fix first: `--camera`
  training must not overwrite the shipped `rl_*.npz` -> save as `rl_*_cam.npz`, load with
  `run.py --rl-cam`. Normal + hard world, fly + eyes-only, 8 drives each. Compare geometry- vs
  camera-trained policies on the camera benchmark (3 seeds x 60 s). Report even if no gain.
- **M7 showcase:** `docs/demo.gif` (README hero, < 4 MB), images in `docs/` (LinkedIn
  1200x627 and 1080x1080, GitHub social preview 1280x640 - uploaded by hand); intact-vs-lesioned
  side-by-side MP4 in git-ignored `showcase/`.
- Run all three in one go; stop to ask only if a run's measured time is > 1.5x its estimate
  (M5 ~20 min, M6 ~55 min) or a result contradicts the plan.

## Milestone 5 result notes (2026-11-12)
- Screen probes use 6 noise seeds (baseline replicate spread about +-1.5%); cumulative lesions
  are re-measured on 6 fresh seeds (winner's-curse guard). Whole screen takes ~2 min.
- Finding: no single interneuron matters; top 10 remove 25% of escape spikes with no behavioural
  cost (hand-tuned 4/29 vs 6/29 intact vs 7-8/29 random; RL 1/29 vs 0/29 vs 0-1/29).
- `flydrive/lesion.py` accepts `ids:1;2;3` (explicit neuron indices).
- M6 code done: `train_rl.py --camera --looming {blob,flow}` saves `rl_*_cam_<looming>.npz`;
  `run.py --rl-cam` loads them; threshold tuning uses the same camera eyes.

## Milestone 7 notes (2026-11-12)
- `make_showcase.py find|build` records frames headlessly (SDL dummy driver) and builds
  `docs/demo.gif`, three social images, and `showcase/lesion-vs-intact.mp4` (git-ignored).
- Fedora's ffmpeg has no libx264: the MP4 uses libopenh264 (H.264, accepted by LinkedIn).
- Clip: seed 2, hand-tuned fly brain, intact vs the 10 escape neurons silenced (the lesioned car
  crashes at 5.0 s, the intact one dodges). The square card says "hand-tuned car crashes" and
  notes that the RL car doesn't, so it doesn't overclaim.
- GitHub social preview (`docs/social-preview.png`, 1280x640) must be uploaded by hand
  (repo Settings -> Social preview); there is no API for it.

## Milestone 6 result (2026-11-12)
- Camera-trained (flow) RL policies for normal and hard worlds, fly + eyes-only: 8 drives each,
  recording ~21 min, whole training incl. threshold tuning ~32 min (both worlds in parallel).
- Camera benchmark (flow looming, seeds 1-3, 60 s): normal 0/29 for all; hard fly 19/94 (geometry-
  trained: 14/94), eyes 19/94 (18/94). Camera-trained cars are ~25% faster (42.4/39.7 vs
  33.1/32.6 km/h). Conclusion: no safety gain; reported as such in the README.
- Camera-trained models (3-31 KB each) are tracked in git; recorded drives (146 MB) stay in the
  git-ignored `models/rl_data/`.

## Repo tidy-up (2026-11-12)
- Experiment scripts moved to `experiments/` (each adds the project root to `sys.path`, so they
  run from any directory); result CSVs moved from `docs/` to `results/`; `docs/` holds only
  README media. Entry points (`run.py`, `train*.py`, `download_data.py`, `setup_fedora.sh`)
  stay at the root. Paths in the entries above refer to the old layout.
- README rewritten around the results (same numbers); stale notes fixed (Godot now tested with
  4.7.2, camera retraining reported as done).
