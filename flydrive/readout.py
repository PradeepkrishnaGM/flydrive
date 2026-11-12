"""Teach the fly brain to drive: a trainable readout on top of the fixed connectome.

The connectome is never changed. Instead we learn a small linear "readout" that
turns the activity of all ~1,300 descending neurons into driving actions.  This
is reservoir computing: the brain is a huge fixed network that transforms the
input, and only the last layer is trained.

Training is imitation learning with DAgger:
  1. A privileged Teacher, which can see exactly where every obstacle is and
     where it is heading, decides the ideal dodge and braking.
  2. The car drives, the teacher labels every moment, and we fit the readout.
  3. We drive again *with the learned readout*, so the car visits the awkward
     states its own mistakes lead to; the teacher labels those too; refit.
"""
import math
from pathlib import Path

import numpy as np

from .world import CAR_RADIUS

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DODGE_SPEED_CMD = 4.0


# --------------------------------------------------------------------------- #
# Teacher
# --------------------------------------------------------------------------- #
class Teacher:
    """Privileged expert: uses exact obstacle positions and velocities."""
    name = "teacher"

    def __init__(self, world, horizon=2.6, margin=0.9):
        self.world = world
        self.horizon = horizon
        self.margin = margin
        self.info = {}

    def label(self):
        """Return (threat 0/1, direction +1 left / -1 right, brake 0..1)."""
        c = self.world.car
        ch, sh = math.cos(c.heading), math.sin(c.heading)
        best = None
        for ob in self.world.obstacles:
            if ob.hit:
                continue
            dx, dy = ob.x - c.x, ob.y - c.y
            fwd = dx * ch + dy * sh
            lat = -dx * sh + dy * ch
            if fwd <= 0 or fwd > 45:
                continue
            ob_fwd_v = ob.vx * ch + ob.vy * sh
            ob_lat_v = -ob.vx * sh + ob.vy * ch
            closing = c.speed - ob_fwd_v
            if closing <= 0.1:
                continue
            t = fwd / closing
            if t > self.horizon:
                continue
            lat_at_t = lat + ob_lat_v * t
            clearance = ob.radius + CAR_RADIUS + self.margin
            if abs(lat_at_t) > clearance:
                continue
            if best is None or t < best[0]:
                best = (t, lat_at_t, ob_lat_v, lat)
        if best is None:
            return 0, 0, 0.0
        t, lat_at_t, ob_lat_v, lat = best
        if abs(ob_lat_v) > 0.3:
            direction = 1 if ob_lat_v < 0 else -1   # pass behind a crossing pedestrian
        elif abs(lat_at_t) > 0.2:
            direction = -1 if lat_at_t > 0 else 1   # go around the side it isn't on
        else:
            direction = 1
        brake = float(np.clip((1.4 - t) / 1.0, 0.0, 1.0)) * 0.7
        return 1, direction, brake

    def update(self, loom_left, loom_right, dt):
        threat, direction, brake = self.label()
        self.info = {"loom_L": loom_left, "loom_R": loom_right, "brake": brake}
        return DODGE_SPEED_CMD * direction * threat, brake


# --------------------------------------------------------------------------- #
# Feature extractors
# --------------------------------------------------------------------------- #
class FlyFeatures:
    """Looming -> LPLC2/LC4 -> whole connectome -> descending neuron firing rates."""
    kind = "fly"

    def __init__(self, brain, loom_gain=600.0, max_rate=200.0, tau=0.06):
        self.brain = brain
        self.idx = brain.groups["dn_all"]
        self.loom_gain, self.max_rate, self.tau = loom_gain, max_rate, tau
        self.rates = np.zeros(len(self.idx), dtype=np.float32)

    def reset(self):
        self.brain.reset()
        self.rates[:] = 0

    def __call__(self, loom_left, loom_right, dt):
        self.brain.set_drive({
            "loom_left": min(self.max_rate, self.loom_gain * loom_left),
            "loom_right": min(self.max_rate, self.loom_gain * loom_right),
        })
        counts = self.brain.run_counts(dt * 1000.0, self.idx)
        a = 1 - math.exp(-dt / self.tau)
        self.rates += a * (counts / dt - self.rates)
        return np.log1p(self.rates)


class EyeFeatures:
    """Control experiment: the same looming signal with NO brain in between.

    Gives the readout recent history and simple nonlinear combinations, so it
    is a fair competitor rather than a straw man."""
    kind = "eyes"

    def __init__(self, history=4):
        self.hist = np.zeros((history, 2), dtype=np.float32)

    def reset(self):
        self.hist[:] = 0

    def __call__(self, loom_left, loom_right, dt):
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = (loom_left, loom_right)
        l, r = self.hist[:, 0], self.hist[:, 1]
        return np.concatenate([self.hist.ravel(), np.sqrt(self.hist.ravel()),
                               [l.sum() - r.sum(), l.sum() + r.sum(), (l.sum() - r.sum()) ** 2]]
                              ).astype(np.float32)


# --------------------------------------------------------------------------- #
# Readout model
# --------------------------------------------------------------------------- #
def weighted_ridge(X, y, w, lam):
    Xw = X * w[:, None]
    A = X.T @ Xw + lam * np.eye(X.shape[1], dtype=np.float64)
    return np.linalg.solve(A, Xw.T @ y)


class Readout:
    """Three linear heads: is there a threat? which way? how hard to brake?"""

    def __init__(self, lam=3.0):
        self.lam = lam

    def fit(self, X, threat, direction, brake):
        X = X.astype(np.float64)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-3
        self.keep = X.std(0) > 1e-6          # ignore neurons that never fired
        Z = self._prep(X)

        pos = threat.mean()
        w_threat = np.where(threat == 1, 0.5 / max(pos, 1e-3), 0.5 / max(1 - pos, 1e-3))
        self.W_threat = weighted_ridge(Z, threat.astype(np.float64), w_threat, self.lam)

        m = threat == 1
        if m.sum() < 5:
            raise ValueError("Not enough threat examples to learn a direction.")
        n_left = (direction[m] > 0).sum()
        w_dir = np.where(direction[m] > 0, 0.5 / max(n_left, 1), 0.5 / max(m.sum() - n_left, 1)) * m.sum()
        self.W_dir = weighted_ridge(Z[m], direction[m].astype(np.float64), w_dir, self.lam)
        self.W_brake = weighted_ridge(Z, brake.astype(np.float64), w_threat, self.lam)
        return self

    def _prep(self, X):
        Z = (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        return np.hstack([Z, np.ones((len(Z), 1))])

    def predict(self, x):
        z = self._prep(x[None, :].astype(np.float64))[0]
        return float(z @ self.W_threat), float(z @ self.W_dir), float(z @ self.W_brake)

    def accuracy(self, X, threat, direction):
        Z = self._prep(X.astype(np.float64))
        pt = (Z @ self.W_threat) > 0.5
        m = threat == 1
        pd = np.sign(Z[m] @ self.W_dir)
        return {
            "threat_recall": float(pt[m].mean()) if m.any() else float("nan"),
            "false_alarm": float(pt[~m].mean()) if (~m).any() else float("nan"),
            "direction_acc": float((pd == direction[m]).mean()) if m.any() else float("nan"),
        }

    def save(self, path, kind, meta=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, kind=kind, lam=self.lam, mu=self.mu, sd=self.sd, keep=self.keep,
                 W_threat=self.W_threat, W_dir=self.W_dir, W_brake=self.W_brake,
                 **(meta or {}))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        r = cls(float(z["lam"]))
        for k in ("mu", "sd", "keep", "W_threat", "W_dir", "W_brake"):
            setattr(r, k, z[k])
        return r, str(z["kind"]), z


class TrainedReflex:
    """A reflex that uses a feature extractor plus a trained Readout."""

    def __init__(self, features, readout, name):
        self.features, self.readout, self.name = features, readout, name
        self.info = {}
        self.last_x = None

    def update(self, loom_left, loom_right, dt):
        x = self.features(loom_left, loom_right, dt)
        self.last_x = x
        threat, direction, brake = self.readout.predict(x)
        brake = float(np.clip(brake, 0, 0.7)) if threat > 0.5 else 0.0
        dodge = DODGE_SPEED_CMD * (1 if direction > 0 else -1) if threat > 0.5 else 0.0
        self.info = {"loom_L": loom_left, "loom_R": loom_right, "brake": brake,
                     "threat score": threat, "direction score": direction}
        return dodge, brake


def model_path(kind):
    return MODEL_DIR / f"readout_{kind}.npz"
