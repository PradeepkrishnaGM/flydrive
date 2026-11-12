"""Bar chart of the lesion experiment (results/lesion_results.csv -> docs/lesion.png)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root: run.py, flydrive/
import csv
from collections import defaultdict

import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HUE = {"baseline": "#52514e", "output": "#2a78d6", "input": "#eb6834",
       "control10": "#b9b8b2", "control162": "#b9b8b2"}
TITLES = {"fly": "Hand-tuned fly brain\n(reads the escape neurons)",
          "rl": "Learned fly brain (RL)\n(reads all 1,299 descending neurons)"}
ORDER = ["no lesion", "Giant Fiber (2)", "other escape DNs (8)", "all escape outputs (10)",
         "LPLC2 neurons", "LC4 neurons", "left-eye looming (162)", "random 10 (5 draws)",
         "random 162 (5 draws)"]


def collect(path):
    """{reflex: {label: (mean collisions, min, max, group)}} and the obstacle total."""
    by = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    group, obstacles = {}, defaultdict(int)
    for r in csv.DictReader(open(path)):
        by[r["reflex"]][r["condition"]][r["seed"]] += int(r["collisions"])
        obstacles[(r["reflex"], r["condition"])] += int(r["obstacles"])
        group[r["condition"]] = r["group"]
    out, total = {}, 0
    for kind, conds in by.items():
        rows, ctl = {}, defaultdict(list)
        for label, seeds in conds.items():
            n = sum(seeds.values())
            if group[label].startswith("control"):
                ctl[group[label]].append(n)
            else:
                rows[label] = (n, n, n, group[label])
                total = obstacles[(kind, label)]
        for g, vals in ctl.items():
            rows[f"random {g[7:]} (5 draws)"] = (float(np.mean(vals)), min(vals), max(vals), g)
        out[kind] = rows
    return out, total


def plot(csv_path, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, total = collect(csv_path)
    kinds = [k for k in ("fly", "rl") if k in data]
    xmax = max(v[2] for k in kinds for v in data[k].values()) * 1.18 + 1
    fig, axes = plt.subplots(1, len(kinds), figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        rows = [(l, data[kind][l]) for l in ORDER if l in data[kind]]
        y = np.arange(len(rows))[::-1]
        ax.set_facecolor(SURFACE)
        for yi, (label, (mean, lo, hi, g)) in zip(y, rows):
            ax.barh(yi, mean, height=0.55, color=HUE[g], edgecolor=SURFACE, linewidth=2,
                    hatch="////" if g.startswith("control") else None)
            if hi > lo:
                ax.plot([lo, hi], [yi, yi], color=INK2, lw=1.5, solid_capstyle="butt")
                for x in (lo, hi):
                    ax.plot([x, x], [yi - 0.14, yi + 0.14], color=INK2, lw=1.5)
            ax.text(hi + xmax * 0.015, yi, f"{mean:.0f}" if mean == int(mean) else f"{mean:.1f}",
                    va="center", ha="left", color=INK, fontsize=9.5)
        ax.set_yticks(y, [l for l, _ in rows], color=INK, fontsize=9.5)
        ax.set_xlim(0, xmax)
        ax.set_title(TITLES[kind], color=INK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel(f"collisions (of {total} obstacles; 3 layouts x 60 s)", color=INK2, fontsize=9)
        ax.tick_params(axis="x", colors=INK2, labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=HUE["baseline"], label="no lesion"),
                        Patch(color=HUE["output"], label="escape neurons silenced"),
                        Patch(color=HUE["input"], label="looming detectors silenced"),
                        Patch(facecolor=HUE["control10"], hatch="////", edgecolor=SURFACE,
                              label="random neurons (bar: mean, whisker: min-max)")],
               loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(png_path, dpi=130, facecolor=SURFACE)
    print("saved", png_path)
