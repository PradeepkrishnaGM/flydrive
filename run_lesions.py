"""Lesion experiment: silence parts of the fly brain and see what the car loses.

    python run_lesions.py --quick            # baseline + Giant Fiber, 1 layout (timing check)
    python run_lesions.py                    # the full experiment -> docs/lesion_results.csv
    python run_lesions.py --plot             # draw docs/lesion.png from the saved CSV

Every condition drives the same 3 obstacle layouts (60 s each, world geometry as eyes, normal
world) with two reflexes: the hand-tuned fly brain ('fly', reads the escape neurons) and the
reinforcement-learning fly brain ('rl', reads all 1,299 descending neurons). The reflex is
built on the intact brain (so it can't "re-learn" its wiring), then the lesion is applied.
"""
import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np

OUT_CSV = Path(__file__).parent / "docs" / "lesion_results.csv"
OUT_PNG = Path(__file__).parent / "docs" / "lesion.png"

# (label, group, lesion spec).  Sizes: GF 2, escape DNs 8, all outputs 10, left eye 162 neurons.
CONDITIONS = [
    ("no lesion", "baseline", ""),
    ("Giant Fiber (2)", "output", "gf_left,gf_right"),
    ("other escape DNs (8)", "output", "esc_left,esc_right"),
    ("all escape outputs (10)", "output", "gf_left,gf_right,esc_left,esc_right"),
    ("LPLC2 only", "input", "LPLC2"),
    ("LC4 only", "input", "LC4"),
    ("left-eye looming (162)", "input", "loom_left"),
] + [(f"random 10 (draw {k})", "control10", f"random:10:{k}") for k in range(1, 6)] \
  + [(f"random 162 (draw {k})", "control162", f"random:162:{k}") for k in range(1, 6)]
REFLEXES = ["fly", "rl"]
SEEDS = [1, 2, 3]
SECONDS = 60

_args = {}


def _task(job):
    label, group, spec, kind, seed = job
    import run
    if "a" not in _args:
        _args["a"] = run.build_parser().parse_args(["--benchmark", str(SECONDS)])
    a = _args["a"]
    a.lesion = spec
    run.get_brain(a).rng = np.random.default_rng(seed)       # same noise for every condition
    s = run.drive(kind, a, seed, seconds=SECONDS)
    return dict(condition=label, group=group, spec=spec, reflex=kind, seed=seed,
                collisions=s["collisions"], obstacles=s["obstacles"], distance=round(s["distance"], 1))


def probe_table(conds):
    """Direct look inside the brain: spikes in 300 ms with one eye driven at 60 Hz."""
    import run
    from flydrive.lesion import resolve
    a = run.build_parser().parse_args([])
    b = run.get_brain(a)
    print("\nInside the brain (spikes in 300 ms, eye driven at 60 Hz)")
    print(f"{'lesion':26s} {'eye':5s} {'GF L+R':>7s} {'escDN L':>8s} {'escDN R':>8s}")
    for label, group, spec in conds:
        if group.startswith("control"):
            continue
        b.unlesion()
        b.lesion(resolve(spec, b.groups, b.n))
        for eye in ("loom_left", "loom_right"):
            b.reset()
            b.set_drive({eye: 60.0})
            c = b.run(300, ["gf_left", "gf_right", "esc_left", "esc_right"])
            print(f"{label:26s} {eye[5:]:5s} {c['gf_left'] + c['gf_right']:7d} "
                  f"{c['esc_left']:8d} {c['esc_right']:8d}")
    b.unlesion()


def summarise(rows):
    print(f"\n{'condition':26s} {'reflex':6s} {'collisions':>11s} {'avoided':>8s} {'km/h':>6s}")
    for kind in REFLEXES:
        for group in ("baseline", "output", "input", "control10", "control162"):
            rs = [r for r in rows if r["reflex"] == kind and r["group"] == group]
            for label in dict.fromkeys(r["condition"] for r in rs):
                one = [r for r in rs if r["condition"] == label]
                col, obs = sum(r["collisions"] for r in one), sum(r["obstacles"] for r in one)
                if group.startswith("control") and not label.endswith("(draw 1)"):
                    continue
                if group.startswith("control"):
                    draws = [[r for r in rs if r["condition"] == c] for c in dict.fromkeys(r["condition"] for r in rs)]
                    per = [sum(r["collisions"] for r in d) for d in draws]
                    km = np.mean([r["distance"] for d in draws for r in d]) / (SECONDS * len(SEEDS)) * 3.6
                    print(f"{'random ' + group[7:] + ' (5 draws)':26s} {kind:6s} "
                          f"{np.mean(per):6.1f} ({min(per)}-{max(per)}) "
                          f"{100 * (1 - np.mean(per) / obs):6.0f}% {km:6.1f}")
                    continue
                km = np.mean([r["distance"] for r in one]) / SECONDS * 3.6
                print(f"{label:26s} {kind:6s} {col:5d}/{obs:<5d} {100 * (obs - col) / obs:7.0f}% {km:6.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.plot:
        from plot_lesions import plot
        plot(OUT_CSV, OUT_PNG)
        return
    conds = CONDITIONS[:2] if a.quick else CONDITIONS
    seeds = SEEDS[:1] if a.quick else SEEDS
    jobs = [(l, g, s, k, sd) for (l, g, s) in conds for k in REFLEXES for sd in seeds]
    print(f"{len(jobs)} drives of {SECONDS} s on {a.jobs} processes...", flush=True)
    with mp.get_context("spawn").Pool(a.jobs) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_task, jobs), 1):
            rows.append(r)
            print(f"  {i}/{len(jobs)} {r['condition']} / {r['reflex']} / layout {r['seed']}: "
                  f"{r['collisions']}/{r['obstacles']}", flush=True)
    if not a.quick:
        OUT_CSV.parent.mkdir(exist_ok=True)
        rows.sort(key=lambda r: (r["reflex"], r["condition"], r["seed"]))
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("saved", OUT_CSV)
    summarise(rows)
    probe_table(conds)


if __name__ == "__main__":
    main()
