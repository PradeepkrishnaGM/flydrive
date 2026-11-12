"""Teach the fly brain to escape with reinforcement learning: no teacher, only crashes.

The car records practice drives, replays every possible escape from every moment of
every encounter, and learns only from which escapes crashed (plus small costs for
swerving and slowing down). See flydrive/rl.py for how this works.

Examples:
  python train_rl.py                 # fly brain + eyes-only control (about 15-25 min)
  python train_rl.py --jobs 4        # record 4 practice drives at once (more RAM, faster)
  python train_rl.py --kind eyes     # eyes-only control, no brain needed (about 2 min)
  python train_rl.py --drives 12     # more practice = better policy, slower

Recorded drives are saved in models/rl_data/, so re-running only trains the policy
(seconds) unless you ask for more drives. Practice uses obstacle layouts from seeds
2000+, so `python run.py --benchmark` (seeds 1-5) tests on unseen layouts.
"""
import argparse
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

from flydrive import rl

REWARD = (rl.COLLISION_PENALTY, rl.ESCAPE_COST, rl.DIST_COST, rl.HORIZON_S)


def drive_file(i, brain, difficulty="normal", camera=False):
    tag = "" if difficulty == "normal" else f"{difficulty}_"
    return rl.RL_DATA_DIR / f"{tag}drive_{i:02d}{'' if brain else '_nobrain'}{'_cam' if camera else ''}.pkl"


def _worker(job):
    i, seconds, use_brain, loom_gain, difficulty, camera = job
    path = drive_file(i, use_brain, difficulty, camera)
    t0 = time.time()
    rec = rl.record_drive(seed=2000 + i, seconds=seconds, use_brain=use_brain,
                          brain_seed=i, loom_gain=loom_gain, difficulty=difficulty, camera=camera)
    t1 = time.time()
    scores = rl.score_drive(rec, difficulty)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"rec": rec, "scores": scores, "reward": REWARD, "seconds": seconds}, f)
    os.replace(tmp, path)
    return (f"drive {i}: {len(rec['spawn_steps'])} encounters, recorded in {t1 - t0:.0f} s, "
            f"practised {4 * len(scores['idx'])} escapes in {time.time() - t1:.0f} s")


def load_drive(i, brain, difficulty="normal", camera=False):
    with open(drive_file(i, brain, difficulty, camera), "rb") as f:
        d = pickle.load(f)
    if tuple(d["reward"]) != REWARD:        # scoring rules changed: re-score, no new recording
        print(f"  re-scoring drive {i} with the current reward settings...")
        d["scores"] = rl.score_drive(d["rec"], difficulty)
        d["reward"] = REWARD
        with open(drive_file(i, brain, difficulty, camera), "wb") as f:
            pickle.dump(d, f)
    return d


def ensure_drives(n, seconds, use_brain, jobs, loom_gain, difficulty="normal", camera=False):
    todo = [i for i in range(n) if not drive_file(i, use_brain, difficulty, camera).exists()]
    if not todo:
        print(f"All {n} practice drives already recorded.")
        return
    what = ("fly brain" if use_brain else "eyes only") + (", camera" if camera else "")
    what += f", {difficulty} world"
    print(f"Recording {len(todo)} practice drive(s) of {seconds:.0f} s ({what}), "
          f"{min(jobs, len(todo))} at a time...", flush=True)
    jobs_list = [(i, seconds, use_brain, loom_gain, difficulty, camera) for i in todo]
    if jobs > 1 and len(todo) > 1:
        with mp.get_context().Pool(min(jobs, len(todo))) as pool:
            for msg in pool.imap_unordered(_worker, jobs_list):
                print("  " + msg, flush=True)
    else:
        for job in jobs_list:
            print("  " + _worker(job), flush=True)


def assemble(datas, kind):
    X, R, C, windows, C_none = [], [], [], [], []
    offset = 0
    for d in datas:
        sc = d["scores"]
        feats = d["rec"]["X_fly" if kind == "fly" else "X_eyes"][sc["idx"]]
        for w in range(len(sc["R_none"])):
            pts = np.flatnonzero(sc["window"] == w)
            if len(pts):
                windows.append((offset + pts[0], offset + pts[-1] + 1, float(sc["R_none"][w])))
                C_none.append(int(sc["C_none"][w]))
        X.append(feats.astype(np.float32))
        R.append(sc["R"])
        C.append(sc["C"])
        offset += len(feats)
    return np.vstack(X), windows, np.vstack(R), np.vstack(C), np.array(C_none)


def tune_threshold(kind, pol, args, seconds=60.0):
    """Pick the trigger threshold by actually driving a practice layout.

    The offline estimate assumes one escape per encounter, which is optimistic when
    obstacles overlap, so the number that matters is measured in closed loop. This
    uses a practice layout (seed 2000+), never a benchmark one."""
    from flydrive.controller import LaneFollower, drive_command
    from flydrive.readout import EyeFeatures, FlyFeatures
    from flydrive.world import World
    brain = None
    if kind == "fly":
        from flydrive.brain import FlyBrain
        brain = FlyBrain(dt=args.brain_dt, seed=0)
    results = []
    for thr in args.thresholds:
        pol.threshold = thr
        world = World(seed=2000 + args.drives, difficulty=args.difficulty)
        world.car.speed = 12.0
        if brain is not None:
            brain.reset()
            feats = FlyFeatures(brain, loom_gain=args.loom_gain)
        else:
            feats = EyeFeatures()
        feats.reset()
        reflex = rl.RLReflex(feats, pol, kind)
        lane = LaneFollower(world)
        for _ in range(int(seconds / rl.DT)):
            steer, accel = drive_command(lane, reflex, world, rl.DT)
            world.step(steer, accel, rl.DT)
        st = world.stats
        results.append((st["collisions"], -st["distance"], thr, st["obstacles"]))
        print(f"  threshold {thr:4.2f}: {st['collisions']:3d}/{st['obstacles']} collisions "
              f"on a practice layout, {st['distance'] / seconds * 3.6:.1f} km/h", flush=True)
    best = min(results)
    pol.threshold = best[2]
    print(f"  -> trigger threshold {best[2]} (tuned by driving)")
    return best[2]


def train(kind, datas, args):
    print(f"\n=== Reinforcement learning: '{kind}' policy ===")
    val, trn = datas[-1:], datas[:-1]
    Xt, Wt, Rt, Ct, Cnt = assemble(trn, kind)
    Xv, Wv, Rv, Cv, Cnv = assemble(val, kind)
    print(f"  {len(Wt)} practice encounters for learning, {len(Wv)} held back to check it")
    best = None
    for lam in args.lams:
        pol = rl.HazardPolicy(lam=lam).setup(Xt, args.components)
        pol.fit(Xt, Wt, Rt, None, iters=args.iters)
        pol.choose_threshold(Xt, Wt, Rt, Ct, Cnt)
        ev = pol.evaluate(Xv, Wv, Rv, Cv, Cnv)
        print(f"  regularisation {lam:>5}: held-out crashes {ev['crashes']}/{ev['encounters']} "
              f"(without escaping: {ev['crashes_without']}), escapes launched {ev['escapes']}, "
              f"score {ev['score']:+.3f}", flush=True)
        if best is None or ev["score"] > best[0]:
            best = (ev["score"], lam)
    lam = best[1]
    print(f"  -> using regularisation {lam}; final fit on all {len(Wt) + len(Wv)} encounters")
    X, W, R, C, Cn = assemble(datas, kind)
    pol = rl.HazardPolicy(lam=lam).setup(X, args.components)
    pol.fit(X, W, R, None, iters=args.iters, log=print)
    thr = pol.choose_threshold(X, W, R, C, Cn)
    ev = pol.evaluate(X, W, R, C, Cn)
    print(f"  offline estimate at threshold {thr}: crashes {ev['crashes']}/"
          f"{ev['encounters']} (without escaping: {ev['crashes_without']}), "
          f"escapes launched {ev['escapes']}")
    if not args.no_tune:
        tune_threshold(kind, pol, args)
    path = rl.model_path(kind, args.difficulty)
    if args.components:
        path = path.with_name(path.stem + f"_pca{args.components}.npz")
    pol.save(path, kind, meta={"loom_gain": args.loom_gain})
    print("  saved", path)
    if kind == "fly":
        from flydrive.connectome import dn_names
        names = dn_names()[pol.keep]
        for head, col, pos, neg in (("launching an escape", 0, "-> escape now", "-> hold off"),
                                    ("escape direction", 1, "-> go left", "-> go right")):
            w = pol.neuron_weights(col)
            print(f"  descending neurons that most influence {head}:")
            for i in np.argsort(-np.abs(w))[:6]:
                print(f"     {names[i]:22s} weight {w[i]:+.3f} {pos if w[i] > 0 else neg}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["fly", "eyes", "both"], default="both")
    p.add_argument("--drives", type=int, default=8, help="number of practice drives")
    p.add_argument("--seconds", type=float, default=120.0, help="length of each practice drive")
    p.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)),
                   help="practice drives recorded in parallel (each uses ~0.6 GB RAM)")
    p.add_argument("--iters", type=int, default=600, help="policy-gradient steps")
    p.add_argument("--lams", type=float, nargs="+", default=[0.01, 0.1, 1.0],
                   help="regularisation strengths to try")
    p.add_argument("--loom-gain", type=float, default=600.0)
    p.add_argument("--brain-dt", type=float, default=0.5, help="neuron time step in ms")
    p.add_argument("--difficulty", choices=["normal", "hard"], default="normal",
                   help="hard: obstacles come faster, several at once, plus roadside clutter")
    p.add_argument("--camera", action="store_true",
                   help="practise with the car's camera instead of exact geometry (slower)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4, 0.6],
                   help="trigger thresholds to try when tuning by driving")
    p.add_argument("--components", type=int,
                   help="squeeze the brain's signals into this many principal components first")
    p.add_argument("--no-tune", action="store_true",
                   help="skip the closed-loop threshold tuning (use the offline estimate)")
    p.add_argument("--record-only", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    use_brain = args.kind != "eyes" or all(
        drive_file(i, True, args.difficulty, args.camera).exists() for i in range(args.drives))
    ensure_drives(args.drives, args.seconds, use_brain, args.jobs, args.loom_gain,
                  args.difficulty, args.camera)
    if args.record_only:
        return
    datas = [load_drive(i, use_brain, args.difficulty, args.camera) for i in range(args.drives)]
    for kind in (["eyes", "fly"] if args.kind == "both" else [args.kind]):
        train(kind, datas, args)
    print("\nDone. Compare on unseen obstacle layouts:  python run.py --benchmark 120 --seeds 1,2,3,4,5")


if __name__ == "__main__":
    main()
