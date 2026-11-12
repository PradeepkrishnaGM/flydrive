"""Compare camera looming (blob growth vs optic flow) with the world's true looming.

Drives with no reflex (so obstacles are approached the whole way) and, every step, records
the true looming from the world's geometry and both pixel estimates from the same frame.
Prints, per distance band: correlation with the truth and the share of frames noticed.

    python experiments/eval_looming.py --seeds 1,2,3 --seconds 60
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root: run.py, flydrive/
import argparse
import math

import numpy as np

from flydrive.controller import NoReflex, drive_command
from flydrive.vision import CameraSenses, Vision
from flydrive.world import World

DT = 0.02
BANDS = [("6-12 m", 6, 12), ("12-40 m", 12, 40)]


def nearest_ahead(world):
    c, best = world.car, 1e9
    for ob in world.obstacles:
        dx, dy = ob.x - c.x, ob.y - c.y
        fwd = dx * math.cos(c.heading) + dy * math.sin(c.heading)
        if fwd > 0:
            best = min(best, fwd)
    return best


def run(seed, seconds, rows):
    world = World(seed=seed)
    world.car.speed = 12.0
    senses = CameraSenses(world, looming="blob")      # also steers the car (lane following)
    from flydrive.vision import VisionLane
    lane, reflex = VisionLane(senses), NoReflex()
    flow = Vision(looming_mode="flow")
    for _ in range(int(seconds / DT)):
        truth = sum(world.looming(DT))
        blob = sum(senses.looming(DT))
        fl = sum(flow.looming(flow.classify(senses.frame), world.t, senses.frame))
        rows.append((nearest_ahead(world), truth, blob, fl))
        steer, accel = drive_command(lane, reflex, world, DT, senses)
        world.step(steer, accel, DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    rows = []
    for s in a.seeds.split(","):
        run(int(s), a.seconds, rows)
    R = np.array(rows)
    print(f"{'band':9s} {'frames':>6s} | {'blob corr':>9s} {'noticed':>8s} | {'flow corr':>9s} {'noticed':>8s}")
    for name, lo, hi in BANDS:
        m = (R[:, 0] >= lo) & (R[:, 0] < hi) & (R[:, 1] > 0)
        t = R[m]
        if len(t) < 5:
            print(f"{name:9s} {len(t):6d} | too few frames")
            continue
        out = [f"{name:9s} {len(t):6d} |"]
        for col in (2, 3):
            c = np.corrcoef(t[:, 1], t[:, col])[0, 1] if t[:, col].std() > 0 else float("nan")
            out.append(f"{c:9.2f} {(t[:, col] > 0).mean():8.0%} |")
        print(" ".join(out))


if __name__ == "__main__":
    main()
