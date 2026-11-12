"""A simple 2D top-down driving world: a looping road, a car and surprise obstacles.

Units are metres, seconds and radians.  x = east, y = north.
"""
import math
from dataclasses import dataclass, field

import numpy as np

ROAD_HALF_WIDTH = 5.0     # road is 10 m wide
CAR_RADIUS = 1.2          # collision radius of the car
WHEELBASE = 2.7


def make_track(straight=220.0, radius=60.0, spacing=1.0):
    """Stadium-shaped loop: two straights joined by two half circles."""
    pts = []
    for x in np.arange(0, straight, spacing):
        pts.append((x, -radius))
    for a in np.arange(-math.pi / 2, math.pi / 2, spacing / radius):
        pts.append((straight + radius * math.cos(a), radius * math.sin(a)))
    for x in np.arange(straight, 0, -spacing):
        pts.append((x, radius))
    for a in np.arange(math.pi / 2, 3 * math.pi / 2, spacing / radius):
        pts.append((radius * math.cos(a), radius * math.sin(a)))
    return np.array(pts)


@dataclass
class Car:
    x: float = 0.0
    y: float = -60.0
    heading: float = 0.0
    speed: float = 0.0
    steer: float = 0.0

    def update(self, steer_cmd, accel_cmd, dt):
        max_steer = 0.6
        self.steer += np.clip(steer_cmd - self.steer, -2.5 * dt, 2.5 * dt)  # steering rate limit
        self.steer = float(np.clip(self.steer, -max_steer, max_steer))
        self.speed = max(0.0, self.speed + float(np.clip(accel_cmd, -9.0, 3.0)) * dt)
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt
        self.heading += self.speed / WHEELBASE * math.tan(self.steer) * dt


@dataclass
class Obstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    prev_theta: dict = field(default_factory=dict)
    hit: bool = False


DIFFICULTY = {
    # spawn interval, how many at once, pedestrian speed, roadside clutter every N spawns
    "normal": {"every": (4.0, 8.0), "count": (1, 1), "ped_speed": (1.2, 2.5), "clutter": 0},
    "hard": {"every": (2.5, 5.0), "count": (1, 3), "ped_speed": (2.5, 5.0), "clutter": 2},
}


class World:
    def __init__(self, seed=0, spawn_every=None, difficulty="normal"):
        self.rng = np.random.default_rng(seed)
        self.track = make_track()
        self.car = Car()
        self.obstacles = []
        self.t = 0.0
        self.difficulty = difficulty
        self.rules = DIFFICULTY[difficulty]
        self.spawn_every = spawn_every or self.rules["every"]
        self.spawns = 0
        self.next_spawn = 3.0
        self.stats = {"obstacles": 0, "collisions": 0, "offroad_time": 0.0, "distance": 0.0}

    # ---------- track helpers ----------
    def nearest_index(self, x, y):
        d = (self.track[:, 0] - x) ** 2 + (self.track[:, 1] - y) ** 2
        return int(np.argmin(d))

    def track_point(self, i):
        return self.track[i % len(self.track)]

    def lateral_offset(self):
        i = self.nearest_index(self.car.x, self.car.y)
        p, q = self.track_point(i), self.track_point(i + 1)
        tx, ty = q - p
        n = math.hypot(tx, ty)
        # positive = car is left of the centreline
        return ((self.car.x - p[0]) * -ty + (self.car.y - p[1]) * tx) / n

    # ---------- obstacles ----------
    def _frame_ahead(self, ahead):
        """A point `ahead` metres along the road, with the road's left direction there."""
        i = self.nearest_index(self.car.x, self.car.y)
        p, q = self.track_point(i + ahead), self.track_point(i + ahead + 1)
        tx, ty = (q - p) / np.linalg.norm(q - p)
        return p, -ty, tx

    def _spawn_one(self):
        """One hazard. The order of random draws here is fixed, so a given seed
        always produces the same 'normal' world it did in earlier versions."""
        ahead = int(self.rng.uniform(22, 38))
        p, nx, ny = self._frame_ahead(ahead)
        if self.rng.random() < 0.5:
            off = self.rng.uniform(-2.0, 2.0)          # static obstacle in the lane
            vx = vy = 0.0
            radius = self.rng.uniform(0.8, 1.3)
        else:
            side = self.rng.choice([-1, 1])            # someone crossing the road
            off = side * self.rng.uniform(4.0, 6.0)
            speed = self.rng.uniform(*self.rules["ped_speed"])
            vx, vy = -side * nx * speed, -side * ny * speed
            radius = 0.6
        self.obstacles.append(Obstacle(p[0] + nx * off, p[1] + ny * off, vx, vy, radius))
        self.stats["obstacles"] += 1

    def _spawn_clutter(self):
        """Harmless roadside things (posts, bins). They never hit the car, but they do
        loom as the car passes, so a good reflex must tell them from real threats."""
        for _ in range(int(self.rng.integers(1, 3))):
            ahead = int(self.rng.uniform(20, 45))
            p, nx, ny = self._frame_ahead(ahead)
            off = self.rng.choice([-1, 1]) * self.rng.uniform(6.0, 9.0)
            ob = Obstacle(p[0] + nx * off, p[1] + ny * off, 0.0, 0.0, self.rng.uniform(0.3, 0.7))
            ob.hazard = False
            self.obstacles.append(ob)

    def spawn_obstacle(self):
        """Something suddenly appears ahead: a parked box or someone crossing.

        In the 'hard' world several can appear at once, people cross faster, and
        harmless clutter stands beside the road."""
        self.spawns += 1
        lo, hi = self.rules["count"]
        for _ in range(lo if lo == hi else int(self.rng.integers(lo, hi + 1))):
            self._spawn_one()
        if self.rules["clutter"] and self.spawns % self.rules["clutter"] == 0:
            self._spawn_clutter()

    # ---------- vision ----------
    def looming(self, dt, fov=math.radians(75), ttc_max=2.5):
        """Return (left, right) looming strength: rate of angular expansion in rad/s.

        This is what fly LPLC2 / LC4 neurons respond to: an object whose image
        grows quickly on the eye, i.e. something on a collision course.  Like
        the real neurons, we ignore slow, distant expansion and only respond
        when the estimated time to collision (theta / d_theta) is short.
        """
        left = right = 0.0
        c = self.car
        for ob in self.obstacles:
            dx, dy = ob.x - c.x, ob.y - c.y
            fwd = dx * math.cos(c.heading) + dy * math.sin(c.heading)
            lat = -dx * math.sin(c.heading) + dy * math.cos(c.heading)
            dist = math.hypot(fwd, lat)
            if fwd <= 0 or dist < 1e-3:
                continue
            bearing = math.atan2(lat, fwd)
            if abs(bearing) > fov:
                continue
            theta = 2 * math.atan(ob.radius / dist)
            prev = ob.prev_theta.get("theta")
            ob.prev_theta["theta"] = theta
            if prev is None:
                continue
            expansion = max(0.0, (theta - prev) / dt)
            if expansion <= 0 or theta / expansion > ttc_max:
                continue
            # smooth split between eyes; objects dead ahead are seen by both
            w_left = 0.5 + 0.5 * math.tanh(bearing / 0.08)
            left += expansion * w_left
            right += expansion * (1 - w_left)
        return left, right

    # ---------- main update ----------
    def step(self, steer_cmd, accel_cmd, dt):
        self.t += dt
        self.car.update(steer_cmd, accel_cmd, dt)
        self.stats["distance"] += self.car.speed * dt
        if abs(self.lateral_offset()) > ROAD_HALF_WIDTH:
            self.stats["offroad_time"] += dt

        if self.t >= self.next_spawn:
            self.spawn_obstacle()
            self.next_spawn = self.t + self.rng.uniform(*self.spawn_every)

        crashed = False
        keep = []
        for ob in self.obstacles:
            ob.x += ob.vx * dt
            ob.y += ob.vy * dt
            d = math.hypot(ob.x - self.car.x, ob.y - self.car.y)
            if not ob.hit and d < ob.radius + CAR_RADIUS:
                ob.hit = True
                crashed = True
                self.stats["collisions"] += 1
            if d < 60:
                keep.append(ob)
        self.obstacles = keep
        return crashed
