"""FlyDrive: a car whose emergency reflexes come from a real fruit fly connectome.

Examples:
  python run.py                      # watch the fly-brain car drive (window)
  python run.py --reflex trained     # the fly brain with a readout learned by train.py
  python run.py --reflex rl          # the fly brain trained by reinforcement learning (train_rl.py)
  python run.py --reflex none        # no reflex: see what happens without it
  python run.py --benchmark 120      # headless comparison of every reflex
  python run.py --benchmark 120 --seeds 1,2,3,4,5   # average over 5 obstacle layouts
  python run.py --probe              # inspect how the fly brain responds to looming
  python run.py --eyes camera        # drive from camera pixels alone (built-in 3D renderer)
  python run.py --eyes camera --renderer godot   # ...with the Godot 3D world as the eyes
"""
import argparse
import time
from pathlib import Path

DT = 0.02  # 50 Hz control loop (20 ms of brain time per car step)

REFLEX_NAMES = {
    "none": "no reflex",
    "direct": "hand-written formula",
    "fly": "fly brain, hand-tuned readout",
    "eyes": "eyes only, trained readout",
    "trained": "fly brain, trained readout",
    "rl-eyes": "eyes only, reinforcement learning",
    "rl": "fly brain, reinforcement learning",
    "teacher": "teacher (cheats: sees everything)",
}
_brain = {}
_renderer = {}


def get_renderer(args):
    """The camera renderer: built-in NumPy, or the Godot world (kept open between drives)."""
    if "r" not in _renderer:
        if args.renderer == "godot":
            from flydrive.godot_bridge import GodotRenderer
            _renderer["r"] = GodotRenderer(port=args.port, godot_bin=args.godot,
                                           launch=not args.no_launch)
        else:
            from flydrive.camera import NumpyRenderer
            _renderer["r"] = NumpyRenderer()
    return _renderer["r"]


def make_senses(world, args):
    """Returns (lane driver, senses). Camera mode: everything comes from pixels."""
    from flydrive.controller import LaneFollower
    if args.eyes != "camera":
        return LaneFollower(world), None
    from flydrive.vision import CameraSenses, VisionLane
    senses = CameraSenses(world, get_renderer(args))
    return VisionLane(senses), senses


def get_brain(args):
    """Load the connectome once and reuse it; reset it for each new drive."""
    if "b" not in _brain:
        from flydrive.brain import FlyBrain
        _brain["b"] = FlyBrain(dt=args.brain_dt, seed=args.seed)
    _brain["b"].reset()
    return _brain["b"]


def make_reflex(kind, args, world):
    from flydrive import controller
    if kind == "none":
        return controller.NoReflex()
    if kind == "direct":
        return controller.DirectReflex()
    if kind == "fly":
        return controller.FlyReflex(get_brain(args), loom_gain=args.loom_gain)
    from flydrive import readout as ro
    if kind == "teacher":
        return ro.Teacher(world)
    if kind in ("rl", "rl-eyes"):
        from flydrive import rl
        model_kind = "fly" if kind == "rl" else "eyes"
        path = Path(args.rl_model) if args.rl_model else rl.model_path(model_kind, args.difficulty)
        if not path.exists():
            raise SystemExit(f"No RL model at {path}. Run:  python train_rl.py "
                             f"--kind {model_kind} --difficulty {args.difficulty}")
        policy, meta = rl.HazardPolicy.load(path)
        if model_kind == "fly":
            features = ro.FlyFeatures(get_brain(args), loom_gain=float(meta["loom_gain"]))
        else:
            features = ro.EyeFeatures()
        features.reset()
        return rl.RLReflex(features, policy, kind)
    model_kind = "fly" if kind == "trained" else "eyes"
    path = ro.model_path(model_kind)
    if not path.exists():
        raise SystemExit(f"No trained model at {path}. Run:  python train.py --kind {model_kind}")
    readout, _, meta = ro.Readout.load(path)
    if model_kind == "fly":
        features = ro.FlyFeatures(get_brain(args), loom_gain=float(meta["loom_gain"]))
    else:
        features = ro.EyeFeatures()
    features.reset()
    return ro.TrainedReflex(features, readout, kind)


def hud_text(kind, world, reflex):
    s, info = world.stats, reflex.info
    txt = (f"{REFLEX_NAMES[kind]}   t {world.t:5.1f} s   {world.car.speed * 3.6:4.1f} km/h   "
           f"obstacles {s['obstacles']}   collisions {s['collisions']}")
    if "threat score" in info:
        txt += f"   escape urge {max(0.0, info['threat score']):.2f}"
    return txt


def drive(kind, args, seed, seconds=None, on_step=None):
    from flydrive.world import World
    from flydrive.controller import drive_command
    world = World(seed=seed, difficulty=args.difficulty)
    world.car.speed = 12.0
    lane, senses = make_senses(world, args)
    reflex = make_reflex(kind, args, world)
    for _ in range(int((seconds or args.benchmark) / DT)):
        if senses is not None:
            senses.hud = hud_text(kind, world, reflex)
        steer, accel = drive_command(lane, reflex, world, DT, senses)
        world.step(steer, accel, DT)
        if on_step:
            on_step(world, reflex)
    return world.stats


def benchmark(args):
    from flydrive import rl
    from flydrive.readout import model_path
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    kinds = ["none", "direct", "fly"]
    kinds += [k for k, m in (("eyes", "eyes"), ("trained", "fly")) if model_path(m).exists()]
    kinds += [k for k, m in (("rl-eyes", "eyes"), ("rl", "fly"))
              if rl.model_path(m, args.difficulty).exists()]
    kinds += ["teacher"]
    if args.only:
        kinds = [k for k in args.only.split(",") if k in REFLEX_NAMES]
    eyes = "camera pixels" if args.eyes == "camera" else "world geometry"
    print(f"World: {args.difficulty}. Eyes: {eyes}. Simulating {args.benchmark:.0f} s of driving per reflex on obstacle layout(s) "
          f"{', '.join(map(str, seeds))} (identical for every reflex)...\n", flush=True)
    rows = []
    for kind in kinds:
        per_seed, obs, dist = [], 0, 0.0
        t0 = time.time()
        for seed in seeds:
            s = drive(kind, args, seed)
            per_seed.append(s["collisions"])
            obs += s["obstacles"]
            dist += s["distance"]
            print(f"  {kind:8s} layout {seed}: {s['collisions']:2d}/{s['obstacles']} collisions",
                  flush=True)
        rows.append((kind, per_seed, obs, dist / (args.benchmark * len(seeds)) * 3.6,
                     time.time() - t0))
    print(f"\n{'reflex':36s} {'collisions per layout':>22s}  total  avoided  avg km/h")
    for kind, per_seed, obs, kmh, _ in rows:
        tot = sum(per_seed)
        print(f"{REFLEX_NAMES[kind]:36s} {' '.join(f'{c:2d}' for c in per_seed):>22s}  "
              f"{tot:2d}/{obs:<3d} {100 * (obs - tot) / obs:6.0f}%  {kmh:8.1f}")


def probe(args):
    """Print how escape neurons respond to looming on each side at several strengths."""
    brain = get_brain(args)
    print("Neuron counts:", {k: len(v) for k, v in brain.groups.items()})
    watch = ["gf_left", "gf_right", "esc_left", "esc_right"]
    print("\nstimulus            rate  GF_L  GF_R  escDN_L  escDN_R  (spikes in 300 ms)")
    for side in ["loom_left", "loom_right"]:
        for rate in [10, 30, 60, 120]:
            brain.set_drive({})
            brain.run(200, [])
            brain.set_drive({side: rate})
            c = brain.run(300, watch)
            print(f"{side:18s} {rate:5d} {c['gf_left']:5d} {c['gf_right']:5d} "
                  f"{c['esc_left']:8d} {c['esc_right']:8d}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reflex", choices=list(REFLEX_NAMES), default="fly")
    p.add_argument("--benchmark", type=float, metavar="SECONDS")
    p.add_argument("--seeds", help="comma-separated obstacle layouts for --benchmark, e.g. 1,2,3,4,5")
    p.add_argument("--only", help="comma-separated reflexes to benchmark, e.g. fly,trained")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--brain-dt", type=float, default=0.5, help="neuron time step in ms")
    p.add_argument("--loom-gain", type=float, default=600.0)
    p.add_argument("--eyes", choices=["geometry", "camera"], default="geometry",
                   help="geometry: looming computed from the world (default); camera: from pixels")
    p.add_argument("--renderer", choices=["numpy", "godot"], default="numpy",
                   help="who draws the camera image in --eyes camera mode")
    p.add_argument("--godot", help="path to the Godot 4 executable (default: look for 'godot')")
    p.add_argument("--no-launch", action="store_true",
                   help="don't start Godot; wait for you to press Play in the Godot editor")
    p.add_argument("--port", type=int, default=47111)
    p.add_argument("--rl-model", help="use this RL policy file instead of the default one")
    p.add_argument("--difficulty", choices=["normal", "hard"], default="normal",
                   help="hard: obstacles come faster, several at once, plus roadside clutter")
    p.add_argument("--seconds", type=float, default=600.0, help=argparse.SUPPRESS)
    p.add_argument("--frames", type=int, help=argparse.SUPPRESS)
    p.add_argument("--screenshot", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.renderer == "godot":
        args.eyes = "camera"
    try:
        if args.probe:
            probe(args)
        elif args.benchmark:
            benchmark(args)
        elif args.renderer == "godot":
            godot_session(args)
        else:
            from flydrive.render import run_window
            run_window(make_reflex, args, DT, make_senses)
    finally:
        if "r" in _renderer and hasattr(_renderer["r"], "close"):
            _renderer["r"].close()


def godot_session(args):
    """Drive in the Godot world. Godot shows the view; this terminal shows progress."""
    last = [0.0]

    def report(world, reflex):
        if world.t - last[0] >= 5.0:
            last[0] = world.t
            s = world.stats
            print(f"  t {world.t:6.1f} s  obstacles {s['obstacles']:3d}  collisions {s['collisions']:3d}",
                  flush=True)
    print(f"Driving '{REFLEX_NAMES[args.reflex]}' with the Godot world as its eyes "
          f"(Ctrl+C to stop)...")
    try:
        s = drive(args.reflex, args, args.seed, seconds=args.seconds, on_step=report)
        print(f"Done: {s['collisions']}/{s['obstacles']} collisions")
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
