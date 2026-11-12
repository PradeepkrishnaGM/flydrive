"""Reinforcement learning for FlyDrive: learn to escape from crashes alone, with no teacher.

The car is only ever told what happened: it crashed, or it didn't, plus small
costs for swerving and for slowing down. Nobody tells it which way to go.

How it stays affordable with a 139,000-neuron brain
---------------------------------------------------
Classic trial-and-error RL would need thousands of brain-simulated practice
drives. Two facts about the task make that unnecessary:

1. Until the car decides to escape, it just follows its lane, so what its eyes
   see, and therefore what the brain does, doesn't depend on the policy. We
   record each surprise obstacle, and the brain's response to it, once.
2. An escape is ballistic, like a fly's jump: a committed 0.6 s manoeuvre that
   runs without the brain once it starts.

So from every moment of every recorded encounter, the car practises all four
escapes (left or right, with or without braking) in a fast physics-only replay
and gets a score. It learns *when* to jump from the timing of the scores, and
*which way* from which escapes crashed.

The policy
----------
Three linear "heads" read the features (descending neuron rates for the fly,
the raw looming signal for the eyes-only control):
  trigger:   probability of launching an escape right now (a "hazard")
  direction: probability of going left rather than right
  brake:     probability of braking during the escape
It is trained by gradient ascent on the expected score (policy gradient). Because
every action at every moment was practised, the expectation is computed exactly
instead of being estimated from a few random tries, which keeps learning fast.
"""
import copy
import math

import numpy as np

from .controller import LaneFollower, drive_command
from .readout import DODGE_SPEED_CMD, MODEL_DIR, EyeFeatures, FlyFeatures
from .world import CAR_RADIUS, Car, Obstacle, World

DT = 0.02
DECIDE_EVERY = 2            # the policy decides every 40 ms
COMMIT_STEPS = 30           # an escape is a committed 0.6 s manoeuvre
REFRACTORY_STEPS = 50       # then 1 s before another escape can start
HORIZON_S = 5.0             # how long after an obstacle appears its outcome is scored
BRAKE_LEVEL = 0.7

# The score. This is the ONLY feedback the policy ever gets.
COLLISION_PENALTY = 1.0
ESCAPE_COST = 0.05          # every escape costs a little (it's uncomfortable)
DIST_COST = 0.01            # per metre of progress lost by slowing down

ACTIONS = [(+1, 0.0), (+1, BRAKE_LEVEL), (-1, 0.0), (-1, BRAKE_LEVEL)]  # (+1 = left, brake)
ACTION_NAMES = ["left", "left+brake", "right", "right+brake"]
RL_DATA_DIR = MODEL_DIR / "rl_data"


def model_path(kind, difficulty="normal", cam=None):
    """Where a policy lives. cam='blob'/'flow': policy trained on camera pixels with that looming
    method (kept apart so it never overwrites the geometry-trained one)."""
    tag = "" if difficulty == "normal" else f"{difficulty}_"
    return MODEL_DIR / f"rl_{tag}{kind}{f'_cam_{cam}' if cam else ''}.npz"


# --------------------------------------------------------------------------- #
# Snapshots: a light copy of the world at one moment, for practice replays
# --------------------------------------------------------------------------- #
def snapshot(world):
    c = world.car
    esc = world.__dict__.get("escape")
    return (world.t, (c.x, c.y, c.heading, c.speed, c.steer),
            [(o.x, o.y, o.vx, o.vy, o.radius, o.prev_theta.get("theta"), o.hit)
             for o in world.obstacles],
            dict(esc) if esc else None, float(getattr(world, "dodge_offset", 0.0)))


def restore(template, snap):
    t, car, obs, esc, offset = snap
    w = copy.copy(template)            # shares the track, nothing else
    w.t = t
    w.car = Car(*car)
    w.obstacles = []
    for x, y, vx, vy, r, theta, hit in obs:
        o = Obstacle(x, y, vx, vy, r)
        o.hit = hit
        if theta is not None:
            o.prev_theta["theta"] = theta
        w.obstacles.append(o)
    if esc is not None:
        w.escape = dict(esc)
    w.dodge_offset = offset
    w.stats = {"obstacles": 0, "collisions": 0, "offroad_time": 0.0, "distance": 0.0}
    w.next_spawn = math.inf            # nothing new appears during a replay
    return w


class _Script:
    """A committed escape: dodge (and maybe brake) for COMMIT_STEPS, then nothing."""

    def __init__(self, direction, brake):
        self.out = (DODGE_SPEED_CMD * direction, brake)
        self.n = 0
        self.info = {}

    def update(self, loom_left, loom_right, dt):
        self.n += 1
        return self.out if self.n <= COMMIT_STEPS else (0.0, 0.0)


def _threat_ahead(world):
    c = world.car
    ch, sh = math.cos(c.heading), math.sin(c.heading)
    for o in world.obstacles:
        if not o.hit and (o.x - c.x) * ch + (o.y - c.y) * sh > -(o.radius + CAR_RADIUS + 0.5):
            return True
    return False


def practice(template, snap, action, max_steps):
    """Replay one escape from a snapshot. Returns (collisions, distance, end speed, steps)."""
    w = restore(template, snap)
    lane = LaneFollower(w)
    script = _Script(*action)
    i = 0
    for i in range(1, max_steps + 1):
        steer, accel = drive_command(lane, script, w, DT)
        w.step(steer, accel, DT)
        if i >= COMMIT_STEPS and not _threat_ahead(w):
            break
    return w.stats["collisions"], w.stats["distance"], w.car.speed, i


# --------------------------------------------------------------------------- #
# Recording practice drives
# --------------------------------------------------------------------------- #
class _Recorder:
    """Sits in the reflex slot, never reacts, and records what the senses/brain do."""

    def __init__(self, fly, eyes):
        self.fly, self.eyes = fly, eyes
        self.fx = self.ex = None
        self.info = {}

    def update(self, loom_left, loom_right, dt):
        self.ex = self.eyes(loom_left, loom_right, dt)
        if self.fly is not None:
            self.fx = self.fly(loom_left, loom_right, dt)
        return 0.0, 0.0


def record_drive(seed, seconds, use_brain=True, brain_seed=0, loom_gain=600.0, brain_dt=0.5,
                 difficulty="normal", camera=False):
    """Drive without reacting, recording features and snapshots at every decision point.

    To make the practice realistic, each new obstacle sometimes appears while the
    car is still recovering from an earlier escape (slower, or off-centre)."""
    fly = None
    if use_brain:
        from .brain import FlyBrain
        fly = FlyFeatures(FlyBrain(dt=brain_dt, seed=brain_seed), loom_gain=loom_gain)
        fly.reset()
    eyes = EyeFeatures()
    rec = _Recorder(fly, eyes)
    world = World(seed=seed, difficulty=difficulty)
    world.car.speed = 12.0
    lane = LaneFollower(world)
    senses = None
    if camera:                       # practise with the car's own eyes instead of geometry
        from .vision import CameraSenses, VisionLane
        senses = CameraSenses(world, looming=camera if camera in ("blob", "flow") else "blob")
        lane = VisionLane(senses)
    rng = np.random.default_rng(seed + 7919)

    n = int(round(seconds / DT))
    base_col = np.zeros(n + 1, dtype=np.int32)
    base_dist = np.zeros(n + 1)
    base_speed = np.zeros(n + 1)
    snaps, Xf, Xe, dec_steps, spawn_steps = [], [], [], [], []
    for s in range(n):
        base_col[s], base_dist[s] = world.stats["collisions"], world.stats["distance"]
        base_speed[s] = world.car.speed
        decide = s % DECIDE_EVERY == 0
        if decide:
            snaps.append(snapshot(world))
        steer, accel = drive_command(lane, rec, world, DT, senses)
        n_obs = world.stats["obstacles"]
        world.step(steer, accel, DT)
        if decide:
            dec_steps.append(s)
            Xe.append(rec.ex)
            if fly is not None:
                Xf.append(rec.fx.astype(np.float16))
        if world.stats["obstacles"] > n_obs:
            spawn_steps.append(s + 1)
            if rng.random() < 0.5:
                world.car.speed = float(rng.uniform(6.0, 12.0))
            if rng.random() < 0.5:
                world.escape = {"dir": 0, "timer": 0.0, "offset": float(rng.uniform(-1.5, 1.5))}
    base_col[n], base_dist[n], base_speed[n] = (world.stats["collisions"], world.stats["distance"],
                                                world.car.speed)
    return {
        "seed": seed, "seconds": seconds, "snaps": snaps, "dec_steps": np.array(dec_steps),
        "X_eyes": np.array(Xe, dtype=np.float32),
        "X_fly": np.array(Xf, dtype=np.float16) if Xf else None,
        "spawn_steps": np.array(spawn_steps), "base_col": base_col, "base_dist": base_dist,
        "base_speed": base_speed,
    }


def score_drive(rec, difficulty="normal"):
    """Practise every escape at every decision point of every encounter and score it."""
    template = World(seed=0, difficulty=difficulty)
    dec, spawns = rec["dec_steps"], rec["spawn_steps"]
    n = len(rec["base_col"]) - 1
    base_col, base_dist, base_speed = rec["base_col"], rec["base_dist"], rec["base_speed"]
    H = int(round(HORIZON_S / DT))

    keep, window_of, R, C = [], [], [], []
    R_none, C_none = [], []
    for wi, start in enumerate(spawns):
        stop = spawns[wi + 1] if wi + 1 < len(spawns) else n
        horizon_end = min(start + H, stop)       # scored until the next obstacle appears
        C_none.append(int(base_col[horizon_end] - base_col[start]))
        R_none.append(-COLLISION_PENALTY * C_none[-1])
        for j in np.flatnonzero((dec >= start) & (dec < stop)):
            k = int(dec[j])
            prior = base_col[k] - base_col[start]           # crashes before escaping
            steps = max(horizon_end - k, COMMIT_STEPS + 1)
            row, crow = [], []
            for action in ACTIONS:
                col, dist, speed, used = practice(template, rec["snaps"][j], action, steps)
                if k + used <= n:
                    base_d = base_dist[k + used] - base_dist[k]
                    base_v = base_speed[k + used]
                else:
                    base_d, base_v = dist, speed
                lost = max(0.0, base_d - dist) + max(0.0, base_v - speed) * 0.67
                crashes = int(prior + (col if k < horizon_end else 0))
                row.append(-COLLISION_PENALTY * crashes - ESCAPE_COST - DIST_COST * lost)
                crow.append(crashes)
            keep.append(j)
            window_of.append(wi)
            R.append(row)
            C.append(crow)
    return {"idx": np.array(keep), "window": np.array(window_of), "R": np.array(R),
            "C": np.array(C), "R_none": np.array(R_none), "C_none": np.array(C_none)}


# --------------------------------------------------------------------------- #
# The policy and its training
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class HazardPolicy:
    """Linear trigger / direction / brake heads over standardised features."""

    def __init__(self, lam=1.0):
        self.lam = lam
        self.threshold = 0.5
        self.basis = None

    # ----- features -----
    def setup(self, X, components=None):
        """Standardise the features, and optionally squeeze them into `components`
        principal components first (fewer knobs to learn = less overfitting)."""
        X = X.astype(np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-3
        self.keep = X.std(0) > 1e-6
        Z = (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        self.basis = None
        if components and components < Z.shape[1]:
            _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
            self.basis = Vt[:components].T
        F = self.basis.shape[1] if self.basis is not None else int(self.keep.sum())
        self.W = np.zeros((F + 1, 3))
        self.W[-1, 0] = -7.0          # start almost never triggering
        return self

    def _prep(self, X):
        Z = (X[:, self.keep].astype(np.float64) - self.mu[self.keep]) / self.sd[self.keep]
        if self.basis is not None:
            Z = Z @ self.basis
        return np.hstack([Z, np.ones((len(Z), 1))])

    def neuron_weights(self, col):
        """Weights per neuron, even when the policy learned in component space."""
        w = self.W[:-1, col]
        return self.basis @ w if self.basis is not None else w

    def probs(self, X):
        P = _sigmoid(self._prep(X) @ self.W)
        return P[:, 0], P[:, 1], P[:, 2]            # trigger, left, brake

    def predict(self, x):
        h, p, q = self.probs(x[None, :])
        return float(h[0]), float(p[0]), float(q[0])

    # ----- objective -----
    @staticmethod
    def _escape_value(R, p, q):
        left = (1 - q) * R[:, 0] + q * R[:, 1]
        right = (1 - q) * R[:, 2] + q * R[:, 3]
        V = p * left + (1 - p) * right
        dV_dp = left - right
        dV_dq = p * (R[:, 1] - R[:, 0]) + (1 - p) * (R[:, 3] - R[:, 2])
        return V, dV_dp, dV_dq

    def objective(self, Z, windows, R, R_none, grad=True):
        """Expected score per encounter, and its gradient (exact policy gradient)."""
        Y = Z @ self.W
        h, p, q = _sigmoid(Y[:, 0]), _sigmoid(Y[:, 1]), _sigmoid(Y[:, 2])
        V, dV_dp, dV_dq = self._escape_value(R, p, q)
        dY = np.zeros_like(Y)
        total = 0.0
        for a, b, rn in windows:                     # each encounter: points a..b-1
            G = rn                                   # value if it never escapes
            hs, Vs = h[a:b], V[a:b]
            hl, Vl = hs.tolist(), Vs.tolist()
            gl = [0.0] * (b - a)
            for i in range(b - a - 1, -1, -1):       # value of waiting, from the end backwards
                gl[i] = G
                G = hl[i] * Vl[i] + (1 - hl[i]) * G
            Gnext = np.array(gl)
            total += G
            if grad:
                S = np.concatenate([[1.0], np.cumprod(1 - hs)[:-1]])  # P(not yet escaped)
                w = S * hs
                dY[a:b, 0] = S * hs * (1 - hs) * (Vs - Gnext)
                dY[a:b, 1] = w * dV_dp[a:b] * p[a:b] * (1 - p[a:b])
                dY[a:b, 2] = w * dV_dq[a:b] * q[a:b] * (1 - q[a:b])
        n = len(windows)
        J = total / n - 0.5 * self.lam * np.sum(self.W[:-1] ** 2) / n
        if not grad:
            return J
        gW = Z.T @ dY / n - self.lam * np.vstack([self.W[:-1], np.zeros((1, 3))]) / n
        return J, gW

    def fit(self, X, windows, R, R_none, iters=600, lr=0.05, log=None):
        Z = self._prep(X)
        m = np.zeros_like(self.W)
        v = np.zeros_like(self.W)
        for t in range(1, iters + 1):
            J, g = self.objective(Z, windows, R, R_none)
            m = 0.9 * m + 0.1 * g                    # Adam, ascending
            v = 0.999 * v + 0.001 * g * g
            self.W += lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            if log and (t == 1 or t % 150 == 0):
                log(f"      step {t:4d}: expected score per encounter {J:+.3f}")
        return self

    def evaluate(self, X, windows, R, C, C_none, threshold=None):
        """Score the *deterministic* policy that the car actually uses."""
        thr = self.threshold if threshold is None else threshold
        h, p, q = self.probs(X)
        score, crashes, escapes = 0.0, 0, 0
        for wi, (a, b, rn) in enumerate(windows):
            fire = np.flatnonzero(h[a:b] > thr)
            if len(fire):
                i = a + fire[0]
                act = (0 if p[i] > 0.5 else 2) + (1 if q[i] > 0.5 else 0)
                score += R[i, act]
                crashes += int(C[i, act])
                escapes += 1
            else:
                score += rn
                crashes += int(C_none[wi])
        return {"score": score / len(windows), "crashes": crashes, "escapes": escapes,
                "encounters": len(windows), "crashes_without": int(np.sum(C_none))}

    def choose_threshold(self, X, windows, R, C, C_none):
        best = max((self.evaluate(X, windows, R, C, C_none, t)["score"], t) for t in
                   (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
        self.threshold = best[1]
        return best[1]

    def save(self, path, kind, meta=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, kind=kind, lam=self.lam, mu=self.mu, sd=self.sd, keep=self.keep, W=self.W,
                 threshold=self.threshold,
                 basis=self.basis if self.basis is not None else np.zeros((0, 0)), **(meta or {}))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        pol = cls(float(z["lam"]))
        pol.mu, pol.sd, pol.keep, pol.W = z["mu"], z["sd"], z["keep"], z["W"]
        pol.threshold = float(z["threshold"])
        basis = z["basis"] if "basis" in z.files else np.zeros((0, 0))
        pol.basis = basis if basis.size else None
        return pol, z


# --------------------------------------------------------------------------- #
# The reflex used while driving
# --------------------------------------------------------------------------- #
class RLReflex:
    """Runs the brain (or eyes) continuously; launches committed escapes like a fly."""

    def __init__(self, features, policy, name):
        self.features, self.policy, self.name = features, policy, name
        self.n = 0
        self.commit = 0
        self.refractory = 0
        self.action = (0.0, 0.0)
        self.info = {}
        self.last_x = None

    def update(self, loom_left, loom_right, dt):
        x = self.features(loom_left, loom_right, dt)
        self.last_x = x
        self.n += 1
        h, p, q = self.policy.predict(x)
        if self.commit > 0:
            self.commit -= 1
            out = self.action
            if self.commit == 0:
                self.refractory = REFRACTORY_STEPS
        elif self.refractory > 0:
            self.refractory -= 1
            out = (0.0, 0.0)
        elif self.n % DECIDE_EVERY == 0 and h > self.policy.threshold:
            direction = 1 if p > 0.5 else -1
            brake = BRAKE_LEVEL if q > 0.5 else 0.0
            self.action = (DODGE_SPEED_CMD * direction, brake)
            self.commit = COMMIT_STEPS - 1
            out = self.action
        else:
            out = (0.0, 0.0)
        self.info = {"loom_L": loom_left, "loom_R": loom_right, "brake": out[1],
                     "threat score": h, "direction score": 2 * p - 1}
        return out
