"""The car's 'cortex' (a plain lane follower) plus interchangeable reflex layers.

  none    - lane follower only. It cannot see obstacles at all.
  direct  - looming signal wired straight to brake/steer, no neurons (a control
            experiment: does the fly brain add anything over a simple formula?)
  fly     - looming signal -> fly LPLC2/LC4 neurons -> whole connectome ->
            Giant Fiber + escape descending neurons -> brake/steer
"""
import math

import numpy as np

CRUISE_SPEED = 12.0   # m/s (~43 km/h)


class LaneFollower:
    """Pure-pursuit steering towards the road centreline, with cruise control."""

    def __init__(self, world, lookahead=7):
        self.world = world
        self.lookahead = lookahead

    def act(self, offset=0.0):
        """offset: desired sideways shift from the centreline in metres (+ = left)."""
        w, c = self.world, self.world.car
        i = w.nearest_index(c.x, c.y)
        p, q = w.track_point(i + self.lookahead), w.track_point(i + self.lookahead + 1)
        d = (q - p) / np.linalg.norm(q - p)
        tx, ty = p[0] - d[1] * offset, p[1] + d[0] * offset
        alpha = math.atan2(ty - c.y, tx - c.x) - c.heading
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        steer = math.atan2(2 * 2.7 * math.sin(alpha), self.lookahead)
        accel = 1.5 * (CRUISE_SPEED - c.speed)
        return steer, accel


class NoReflex:
    name = "none"

    def __init__(self, *_, **__):
        self.info = {}

    def update(self, loom_left, loom_right, dt):
        self.info = {"loom_L": loom_left, "loom_R": loom_right}
        return 0.0, 0.0  # (dodge m/s sideways, brake 0..1)


class DirectReflex:
    """Hand-written rule: brake and swerve in proportion to looming."""
    name = "direct"

    def __init__(self, threshold=0.08, *_, **__):
        self.threshold = threshold
        self.info = {}

    def update(self, loom_left, loom_right, dt):
        total = loom_left + loom_right
        brake = float(np.clip((total - self.threshold) / 0.3, 0, 1))
        # threat on the left -> dodge right (negative). Sideways speed in m/s.
        dodge = float(np.clip(-40.0 * (loom_left - loom_right), -4.0, 4.0))
        if total > 0.02 and abs(dodge) < 1.0:
            dodge = 4.0 if loom_right >= loom_left else -4.0
        self.info = {"loom_L": loom_left, "loom_R": loom_right, "brake": brake}
        return dodge, brake


class FlyReflex:
    """Drive the real connectome with looming input and read out escape neurons."""
    name = "fly"

    def __init__(self, brain, loom_gain=600.0, max_rate=200.0):
        self.brain = brain
        self.loom_gain = loom_gain       # Hz of LPLC2/LC4 firing per rad/s of expansion
        self.max_rate = max_rate
        self.rates = {"gf": 0.0, "esc_left": 0.0, "esc_right": 0.0}
        self.info = {}
        self.calibrate()

    def calibrate(self, ms=300):
        """Discover how the wiring is lateralised instead of assuming it.

        Stimulate the left-eye looming detectors and see which side's escape
        neurons respond more. Real flies jump away from a looming threat, so we
        steer away from the side that activates."""
        watch = ["esc_left", "esc_right"]
        self.brain.set_drive({"loom_left": 100.0})
        c = self.brain.run(ms, watch)
        self.left_threat_dn = "esc_left" if c["esc_left"] >= c["esc_right"] else "esc_right"
        self.right_threat_dn = "esc_right" if self.left_threat_dn == "esc_left" else "esc_left"
        self.brain.set_drive({})
        self.brain.run(200, [])  # let activity die down
        print(f"[calibration] left-eye looming mostly drives {self.left_threat_dn} "
              f"({c['esc_left']} vs {c['esc_right']} spikes)")

    def update(self, loom_left, loom_right, dt):
        b = self.brain
        rl = min(self.max_rate, self.loom_gain * loom_left)
        rr = min(self.max_rate, self.loom_gain * loom_right)
        b.set_drive({"loom_left": rl, "loom_right": rr})

        watch = ["gf_left", "gf_right", "esc_left", "esc_right"]
        counts = b.run(dt * 1000.0, watch)
        sec = dt
        inst = {
            "gf": (counts["gf_left"] + counts["gf_right"]) / 2 / sec,     # Hz per neuron
            "esc_left": counts["esc_left"] / len(b.groups["esc_left"]) / sec,
            "esc_right": counts["esc_right"] / len(b.groups["esc_right"]) / sec,
        }
        a = 1 - math.exp(-dt / 0.06)  # ~60 ms smoothing of firing rates
        for k in self.rates:
            self.rates[k] += a * (inst[k] - self.rates[k])

        gf = self.rates["gf"]
        # Giant Fiber = "escape now!": scales braking and the dodge
        urgency = float(np.clip((gf - 5.0) / 50.0, 0, 1))
        brake = 0.7 * urgency
        left_threat = self.rates[self.left_threat_dn]
        right_threat = self.rates[self.right_threat_dn]
        # dodge away from the side whose escape neurons fire more
        direction = np.tanh((right_threat - left_threat) / 8.0)
        dodge = float(4.0 * direction * max(urgency, 0.3 if max(left_threat, right_threat) > 5 else 0.0))
        self.info = {
            "loom_L": loom_left, "loom_R": loom_right,
            "LPLC2/LC4 L Hz": rl, "LPLC2/LC4 R Hz": rr,
            "GiantFiber Hz": gf,
            "escDN L Hz": self.rates["esc_left"], "escDN R Hz": self.rates["esc_right"],
            "brake": brake,
        }
        return dodge, brake


MAX_DODGE = 3.5          # metres the car may shift sideways from the lane centre
MIN_ESCAPE_SPEED = 6.0   # m/s
ESCAPE_HOLD = 1.2        # s the escape continues after the reflex goes quiet
DODGE_SPEED = 6.0        # m/s sideways


def drive_command(lane, reflex, world, dt, senses=None):
    """Combine lane following with the reflex layer.

    Like a fly's escape jump, a dodge is ballistic: the first strong reflex
    signal picks a direction, and the car commits to it (shifting sideways in
    its lane) for as long as the reflex keeps firing plus a short hold time.
    Afterwards it drifts back to the lane centre.
    """
    loom_l, loom_r = (senses or world).looming(dt)   # camera mode: looming from pixels
    dodge, brake = reflex.update(loom_l, loom_r, dt)
    st = world.__dict__.setdefault("escape", {"dir": 0, "timer": 0.0, "offset": 0.0})

    if abs(dodge) > 0.5:
        if st["timer"] <= 0:
            st["dir"] = 1 if dodge > 0 else -1   # commit to a direction
        st["timer"] = ESCAPE_HOLD
    st["timer"] -= dt

    if st["timer"] > 0:
        target = st["dir"] * MAX_DODGE
        step = DODGE_SPEED * dt
        st["offset"] += float(np.clip(target - st["offset"], -step, step))
    else:
        st["offset"] *= math.exp(-dt / 1.5)
    world.dodge_offset = st["offset"]

    steer, accel = lane.act(st["offset"])
    # Braking helps, but a car needs speed to steer around things, so an escape
    # never slows below MIN_ESCAPE_SPEED.
    if brake > 0 and world.car.speed > MIN_ESCAPE_SPEED:
        accel = -9.0 * brake
    elif brake > 0:
        accel = min(accel, 0.0)
    return steer, accel
