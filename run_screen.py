"""Interneuron screen: which single cells between the eyes and the escape neurons matter?

    python run_screen.py              # the whole screen (~20 min) -> docs/screen_*.csv
    python run_screen.py --plot       # draw docs/screen.png from the saved CSV

1. Drive each eye at 60 Hz for 300 ms; the 50 most active neurons that are neither looming
   inputs nor descending outputs are the candidates.
2. Silence each candidate alone and measure how much the escape outputs (Giant Fiber + 8 escape
   DNs, what the hand-tuned car reads) and the whole descending population (1,299 neurons, what
   the RL car reads) lose. 6 noise seeds x 2 eyes per probe
   (baseline replicates differ by about +-1.5%, so only larger drops mean anything).
3. Silence the top 1, 5, 10, 20 together (ranked by escape-output drop), re-measured on 6
   fresh noise seeds so lucky rankings don't inflate the result.
4. Drive with the top 5 / top 10 silenced vs random sets of the same size drawn from the active
   candidates with below-median effect (3 draws), both reflexes, 3 layouts x 60 s.
"""
import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np

DOCS = Path(__file__).parent / "docs"
SINGLE_CSV, BEHAV_CSV, PNG = DOCS / "screen_results.csv", DOCS / "screen_behaviour.csv", DOCS / "screen.png"
N_CAND, EYES = 50, ("loom_left", "loom_right")
SEEDS_PROBE = (1, 2, 3, 4, 5, 6)           # noise seeds for the screen itself
SEEDS_CHECK = (7, 8, 9, 10, 11, 12)        # fresh seeds to confirm the cumulative lesions
RATE, MS = 60.0, 300
KS = (1, 5, 10, 20)


def labels(n_idx):
    from flydrive import connectome
    z = np.load(connectome.CACHE)
    ct, hb, sd = z["ct_cell_type"], z["ct_hemibrain_type"], z["ct_side"]
    return [f"{(ct[i] or hb[i] or '?')}_{sd[i] or '?'}" for i in n_idx]


class Prober:
    def __init__(self):
        from flydrive.brain import FlyBrain
        self.b = FlyBrain(dt=0.5, seed=0)
        g = self.b.groups
        self.out = np.concatenate([g["gf_left"], g["gf_right"], g["esc_left"], g["esc_right"]])
        self.idx = np.union1d(g["dn_all"], self.out).astype(np.int32)
        self.out_mask = np.isin(self.idx, self.out)

    def measure(self, lesion=()):
        """(escape-output spikes, descending-population spikes), summed over eyes and seeds."""
        b = self.b
        b.unlesion()
        if len(lesion):
            b.lesion(np.asarray(lesion, dtype=np.int32))
        o = d = 0.0
        for eye in EYES:
            for seed in SEEDS_PROBE:
                b.reset()
                b.rng = np.random.default_rng(seed)
                b.set_drive({eye: RATE})
                c = b.run_counts(MS, self.idx)
                o += c[self.out_mask].sum()
                d += c.sum()
        b.unlesion()
        return o, d


def activity_candidates(p):
    b, g = p.b, p.b.groups
    total = np.zeros(b.n)
    allidx = np.arange(b.n, dtype=np.int32)
    for eye in EYES:
        b.reset()
        b.rng = np.random.default_rng(0)
        b.set_drive({eye: RATE})
        total += b.run_counts(MS, allidx)
    skip = np.concatenate([g[k] for k in ("loom_left", "loom_right", "gf_left", "gf_right",
                                          "esc_left", "esc_right", "dn_all")])
    total[skip] = -1
    order = np.argsort(-total, kind="stable")[:N_CAND]
    return order.astype(np.int32), total[order]


def screen():
    p = Prober()
    cand, act = activity_candidates(p)
    names = labels(cand)
    base_o, base_d = p.measure()
    print(f"baseline: escape outputs {base_o:.0f} spikes, descending population {base_d:.0f}")
    rows = []
    for i, (n, nm, a) in enumerate(zip(cand, names, act), 1):
        o, d = p.measure([n])
        rows.append(dict(neuron=int(n), name=nm, activity=int(a),
                         out_drop=round(1 - o / base_o, 4), dn_drop=round(1 - d / base_d, 4)))
        print(f"  {i:2d}/{N_CAND} {nm:18s} activity {a:5.0f}  escape-output drop {rows[-1]['out_drop']:+.2f}"
              f"  population drop {rows[-1]['dn_drop']:+.2f}", flush=True)
    rows.sort(key=lambda r: -r["out_drop"])
    with open(SINGLE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ranked = [r["neuron"] for r in rows]
    # Ranking by a noisy measurement favours lucky neurons, so confirm on fresh noise seeds.
    global SEEDS_PROBE
    SEEDS_PROBE = SEEDS_CHECK
    base_o, base_d = p.measure()
    print("\ncumulative lesions (top k by escape-output drop), re-measured on fresh noise seeds")
    for k in KS:
        o, d = p.measure(ranked[:k])
        print(f"  top {k:2d}: escape outputs {100 * (1 - o / base_o):+5.0f}%   "
              f"descending population {100 * (1 - d / base_d):+5.0f}%")
    return rows


def behaviour(rows, jobs):
    import run_lesions as rl
    ranked = [r["neuron"] for r in rows]
    low = ranked[len(ranked) // 2:]                       # active, but below-median effect
    spec = lambda ids: "ids:" + ";".join(str(int(i)) for i in ids)
    conds = [("no lesion", "baseline", "")]
    for k in (5, 10):
        conds.append((f"top {k} interneurons", f"top{k}", spec(ranked[:k])))
        for d in range(1, 4):
            ids = np.random.default_rng(100 * k + d).choice(low, size=k, replace=False)
            conds.append((f"random {k} (draw {d})", f"control{k}", spec(ids)))
    work = [(l, g, s, kind, sd) for (l, g, s) in conds for kind in rl.REFLEXES for sd in rl.SEEDS]
    print(f"\n{len(work)} drives of {rl.SECONDS} s on {jobs} processes...", flush=True)
    out = []
    with mp.get_context("spawn").Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(rl._task, work), 1):
            out.append(r)
            print(f"  {i}/{len(work)} {r['condition']} / {r['reflex']} / layout {r['seed']}: "
                  f"{r['collisions']}/{r['obstacles']}", flush=True)
    out.sort(key=lambda r: (r["reflex"], r["condition"], r["seed"]))
    with open(BEHAV_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summarise(out)


def summarise(rows):
    print(f"\n{'lesion':22s} {'reflex':6s} {'collisions':>11s}")
    for kind in ("fly", "rl"):
        for g in ("baseline", "top5", "control5", "top10", "control10"):
            rs = [r for r in rows if r["reflex"] == kind and r["group"] == g]
            conds = list(dict.fromkeys(r["condition"] for r in rs))
            per = [sum(r["collisions"] for r in rs if r["condition"] == c) for c in conds]
            obs = sum(r["obstacles"] for r in rs if r["condition"] == conds[0])
            name = conds[0] if len(conds) == 1 else f"random {g[7:]} ({len(conds)} draws)"
            rng = f"{np.mean(per):.1f} ({min(per)}-{max(per)})" if len(per) > 1 else f"{per[0]}"
            print(f"{name:22s} {kind:6s} {rng:>9s}/{obs}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(SINGLE_CSV)))[:15]
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    y = np.arange(len(rows))[::-1]
    h = 0.36
    ax.barh(y + h / 2, [100 * float(r["out_drop"]) for r in rows], h, color="#2a78d6",
            edgecolor=SURF, linewidth=2, label="escape outputs (GF + 8 escape DNs)")
    ax.barh(y - h / 2, [100 * float(r["dn_drop"]) for r in rows], h, color="#eb6834",
            edgecolor=SURF, linewidth=2, label="all 1,299 descending neurons")
    ax.axvspan(-1.5, 1.5, color=GRID, alpha=0.6, zorder=0)
    ax.text(0, len(rows) - 0.35, "noise", color=INK2, fontsize=8.5, ha="center", va="bottom")
    ax.set_yticks(y, [r["name"] for r in rows], color=INK, fontsize=9.5)
    ax.set_xlabel("spikes lost when this one neuron is silenced (%)", color=INK2, fontsize=9)
    ax.set_title("Which single interneurons carry the looming signal?", color=INK, loc="left", fontsize=11.5)
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(PNG, dpi=130, facecolor=SURF)
    print("saved", PNG)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--screen-only", action="store_true", help="skip the driving test")
    a = ap.parse_args()
    if a.plot:
        plot()
        return
    DOCS.mkdir(exist_ok=True)
    rows = screen()
    plot()
    if not a.screen_only:
        behaviour(rows, a.jobs)


if __name__ == "__main__":
    main()
