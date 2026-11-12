"""Train a readout that teaches the fly brain to drive (imitation learning + DAgger).

Examples:
  python train.py                   # train the fly-brain readout and the eyes-only control
  python train.py --kind fly        # only the fly brain
  python train.py --iters 5 --seconds 120   # longer training
  python train.py --quick           # fast smoke test (lower quality)
  python train.py --resume          # continue an interrupted training run

Training uses obstacle layouts from seeds 1000+, so `python run.py --benchmark`
(seeds 1-5) tests on layouts the readout has never seen.
"""
import argparse
import time

import numpy as np

from flydrive.controller import LaneFollower, drive_command
from flydrive.readout import (EyeFeatures, FlyFeatures, Readout, Teacher, TrainedReflex,
                              model_path)
from flydrive.world import World

DT = 0.02


class Recorder:
    """Sits in the reflex slot: records features + teacher labels, and decides who drives."""

    def __init__(self, features, teacher, student=None):
        self.features, self.teacher, self.student = features, teacher, student
        self.X, self.threat, self.direction, self.brake = [], [], [], []
        self.info = {}

    def update(self, loom_left, loom_right, dt):
        if self.student is None:
            x = self.features(loom_left, loom_right, dt)
            action = self.teacher.update(loom_left, loom_right, dt)
        else:
            action = self.student.update(loom_left, loom_right, dt)
            x = self.student.last_x
        t, d, b = self.teacher.label()
        self.X.append(x)
        self.threat.append(t)
        self.direction.append(d)
        self.brake.append(b)
        return action


def episode(features, seed, seconds, student=None):
    world = World(seed=seed)
    world.car.speed = 12.0
    lane = LaneFollower(world)
    features.reset()
    rec = Recorder(features, Teacher(world), student)
    for _ in range(int(seconds / DT)):
        steer, accel = drive_command(lane, rec, world, DT)
        world.step(steer, accel, DT)
    data = (np.array(rec.X), np.array(rec.threat), np.array(rec.direction), np.array(rec.brake))
    return data, world.stats


def checkpoint_path(kind):
    return model_path(kind).with_name(f"dagger_data_{kind}.npz")


def load_checkpoint(kind):
    z = np.load(checkpoint_path(kind))
    n = int(z["n_chunks"])
    return ([z[f"X{i}"] for i in range(n)], [z[f"T{i}"] for i in range(n)],
            [z[f"D{i}"] for i in range(n)], [z[f"B{i}"] for i in range(n)])


def save_checkpoint(kind, Xs, Ts, Ds, Bs):
    path = checkpoint_path(kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"n_chunks": len(Xs)}
    for i in range(len(Xs)):
        arrays.update({f"X{i}": Xs[i], f"T{i}": Ts[i], f"D{i}": Ds[i], f"B{i}": Bs[i]})
    np.savez_compressed(path, **arrays)


def train(kind, args):
    if kind == "fly":
        from flydrive.brain import FlyBrain
        print("Loading fly brain...")
        features = FlyFeatures(FlyBrain(dt=args.brain_dt, seed=0), loom_gain=args.loom_gain)
    else:
        features = EyeFeatures()

    print(f"\n=== Training '{kind}' readout: {args.iters} DAgger iterations x {args.seconds:.0f} s ===")
    Xs, Ts, Ds, Bs = [], [], [], []
    readout = None
    if args.resume and checkpoint_path(kind).exists():
        Xs, Ts, Ds, Bs = load_checkpoint(kind)
        readout = Readout(lam=args.lam).fit(np.vstack(Xs), np.concatenate(Ts),
                                            np.concatenate(Ds), np.concatenate(Bs))
        print(f"resumed from checkpoint: {len(Xs)} iteration(s) already done")

    for it in range(len(Xs), args.iters):
        student = None if readout is None else TrainedReflex(features, readout, kind)
        who = "teacher drives" if student is None else "student drives, teacher labels"
        t0 = time.time()
        (X, T, D, B), stats = episode(features, seed=1000 + it, seconds=args.seconds, student=student)
        msg = (f"iter {it}: {who:32s} collisions {stats['collisions']:2d}/{stats['obstacles']:<2d} "
               f"({time.time() - t0:5.1f} s)")
        if readout is not None:
            acc = readout.accuracy(X, T, D)
            msg += (f"  on-policy: threat recall {acc['threat_recall']:.2f}, "
                    f"false alarms {acc['false_alarm']:.3f}, direction {acc['direction_acc']:.2f}")
        print(msg, flush=True)
        Xs.append(X); Ts.append(T); Ds.append(D); Bs.append(B)
        readout = Readout(lam=args.lam).fit(np.vstack(Xs), np.concatenate(Ts),
                                            np.concatenate(Ds), np.concatenate(Bs))
        save_checkpoint(kind, Xs, Ts, Ds, Bs)

    X, T, D = np.vstack(Xs), np.concatenate(Ts), np.concatenate(Ds)
    acc = readout.accuracy(X, T, D)
    print(f"final fit on {len(X)} samples ({int(T.sum())} threat moments, "
          f"{int(readout.keep.sum())} active features): threat recall {acc['threat_recall']:.2f}, "
          f"false alarms {acc['false_alarm']:.3f}, direction {acc['direction_acc']:.2f}")
    path = model_path(kind)
    readout.save(path, kind, meta={"loom_gain": args.loom_gain, "brain_dt": args.brain_dt})
    print("saved", path)
    if kind == "fly":
        from flydrive.connectome import dn_names
        names = dn_names()[readout.keep]
        w = readout.W_dir[:-1]
        print("descending neurons that most influence dodge direction:")
        for i in np.argsort(-np.abs(w))[:8]:
            side = "-> dodge left" if w[i] > 0 else "-> dodge right"
            print(f"   {names[i]:24s} weight {w[i]:+.3f} {side}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["fly", "eyes", "both"], default="both")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seconds", type=float, default=90.0, help="simulated seconds per iteration")
    p.add_argument("--lam", type=float, default=3.0, help="ridge regularisation strength")
    p.add_argument("--brain-dt", type=float, default=0.5)
    p.add_argument("--loom-gain", type=float, default=600.0)
    p.add_argument("--quick", action="store_true", help="2 short iterations, for testing")
    p.add_argument("--resume", action="store_true", help="continue from the saved checkpoint")
    args = p.parse_args()
    if args.quick:
        args.iters, args.seconds = 2, 40.0
    kinds = ["eyes", "fly"] if args.kind == "both" else [args.kind]
    for k in kinds:
        train(k, args)
    print("\nDone. Compare everything on unseen obstacle layouts with:  python run.py --benchmark 120")


if __name__ == "__main__":
    main()
