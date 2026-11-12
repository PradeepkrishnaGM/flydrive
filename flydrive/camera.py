"""The car's camera: what the world looks like from the driver's seat.

This module defines the camera (where it sits, how wide it sees) and the look of
the world (road, markings, grass, sky, obstacles). Two renderers draw it:

  * NumpyRenderer (here): a small ray-casting renderer in pure NumPy. Fast, needs
    no extra software, and can also report the true identity of every pixel so
    the vision system can be graded.
  * The Godot project in godot_world/: prettier, with lighting and shadows. It gets
    the same geometry and colours from Python (see godot_bridge.py).

Both only *render*. The world's rules (car physics, obstacles, collisions) stay in
world.py, so results are comparable with every other mode.
"""
import itertools
import math

import numpy as np

from .world import ROAD_HALF_WIDTH

# ---- camera -------------------------------------------------------------------
IMG_W, IMG_H = 320, 96
HFOV_DEG = 110.0                    # horizontal field of view
CAM_HEIGHT = 1.2                    # metres above the road (roughly a driver's eyes)
CAM_FORWARD = 1.9                   # metres ahead of the car's centre
FOCAL = (IMG_W / 2) / math.tan(math.radians(HFOV_DEG) / 2)   # pixels
HORIZON_ROW = IMG_H / 2             # the camera looks level

# ---- the look of the world (also sent to Godot) --------------------------------
COLORS = {
    "grass": (0.30, 0.52, 0.22),
    "road": (0.34, 0.34, 0.36),
    "line": (0.92, 0.92, 0.90),
    "sky_top": (0.42, 0.62, 0.92),
    "sky_horizon": (0.80, 0.87, 0.95),
    "skin": (0.85, 0.66, 0.52),
}
BOX_COLORS = [(0.86, 0.47, 0.16), (0.66, 0.38, 0.14), (0.93, 0.62, 0.22)]
PERSON_COLORS = [(0.80, 0.14, 0.18), (0.85, 0.20, 0.55), (0.92, 0.80, 0.12),
                 (0.93, 0.42, 0.62), (0.85, 0.35, 0.10)]
CENTER_LINE_HALF = 0.08             # dashed centre line, 16 cm wide
DASH_PERIOD, DASH_ON = 6.0, 3.0     # 3 m dash every 6 m
EDGE_LINE = (4.62, 4.78)            # solid edge lines near the road edges
PERSON_RADIUS = 0.32                # what a pedestrian looks like (collisions use world.py)
HEAD = 0.25                         # top 25 cm of a pedestrian is skin coloured
SUN = np.array([0.45, -0.35, 0.82]) / np.linalg.norm([0.45, -0.35, 0.82])

# stadium track used by world.make_track()
STRAIGHT, RADIUS = 220.0, 60.0


def appearance(ob, world):
    """How an obstacle looks: (uid, kind, size, height, rgb). Deterministic per drive."""
    uid = ob.__dict__.get("uid")
    if uid is None:
        counter = world.__dict__.setdefault("_uid_counter", itertools.count(1))
        uid = ob.uid = next(counter)
    if ob.vx or ob.vy:
        return uid, "person", PERSON_RADIUS, 1.62 + 0.06 * (uid % 3), PERSON_COLORS[uid % len(PERSON_COLORS)]
    return uid, "box", ob.radius * 0.8, 0.9 + 0.1 * (uid % 4), BOX_COLORS[uid % len(BOX_COLORS)]


def track_coords(x, y):
    """Distance from the road centreline and distance along the track, for many points."""
    d = np.empty_like(x)
    s = np.empty_like(x)
    mid = (x >= 0) & (x <= STRAIGHT)
    bot = mid & (y < 0)
    top = mid & (y >= 0)
    d[bot] = np.abs(y[bot] + RADIUS)
    s[bot] = x[bot]
    d[top] = np.abs(y[top] - RADIUS)
    s[top] = STRAIGHT + math.pi * RADIUS + (STRAIGHT - x[top])
    r = x > STRAIGHT
    ang = np.arctan2(y[r], x[r] - STRAIGHT)
    d[r] = np.abs(np.hypot(x[r] - STRAIGHT, y[r]) - RADIUS)
    s[r] = STRAIGHT + RADIUS * (ang + math.pi / 2)
    lft = x < 0
    ang = np.mod(np.arctan2(y[lft], x[lft]), 2 * math.pi)
    d[lft] = np.abs(np.hypot(x[lft], y[lft]) - RADIUS)
    s[lft] = 2 * STRAIGHT + math.pi * RADIUS + RADIUS * (ang - math.pi / 2)
    return d, s


def _cell_noise(x, y):
    """Smooth-ish texture: a random brightness per 1 m ground cell, in [-0.5, 0.5]."""
    v = np.sin(np.floor(x) * 12.9898 + np.floor(y) * 78.233) * 43758.5453
    return v - np.floor(v) - 0.5


def camera_pose(car):
    ch, sh = math.cos(car.heading), math.sin(car.heading)
    return car.x + CAM_FORWARD * ch, car.y + CAM_FORWARD * sh, ch, sh


# label ids (used by the vision system and for grading it)
SKY, GRASS, ROAD, LINE, OBSTACLE = 0, 1, 2, 3, 4


class NumpyRenderer:
    """Ray-casts one camera frame from the car's current position."""

    def __init__(self, w=IMG_W, h=IMG_H):
        self.w, self.h = w, h
        u = (np.arange(w) + 0.5 - w / 2) / FOCAL          # right is positive
        v = (np.arange(h) + 0.5 - h / 2) / FOCAL          # down is positive
        self.xc, self.yc = np.meshgrid(u, v)
        frac = np.clip((h / 2 - np.arange(h)) / (h / 2), 0, 1)[:, None, None]
        top, hor = np.array(COLORS["sky_top"]), np.array(COLORS["sky_horizon"])
        self.sky = np.broadcast_to(hor + (top - hor) * frac, (h, w, 3)).astype(np.float32)
        self.ground = self.yc > 1e-6

    def render(self, world, labels=False, **_):
        cx, cy, ch, sh = camera_pose(world.car)
        # ray directions with forward component 1: forward + x * right, and z = -y (up)
        dx = ch + self.xc * sh
        dy = sh - self.xc * ch
        dz = -self.yc
        img = self.sky.copy()
        lab = np.zeros((self.h, self.w), dtype=np.uint8)
        depth = np.full((self.h, self.w), np.inf)

        g = self.ground
        t = CAM_HEIGHT / self.yc[g]
        gx, gy = cx + t * dx[g], cy + t * dy[g]
        d, s = track_coords(gx, gy)
        on_road = d < ROAD_HALF_WIDTH
        line = on_road & (((d < CENTER_LINE_HALF) & (np.mod(s, DASH_PERIOD) < DASH_ON))
                          | ((d > EDGE_LINE[0]) & (d < EDGE_LINE[1])))
        noise = _cell_noise(gx, gy)[:, None]
        col = np.where(on_road[:, None],
                       np.array(COLORS["road"]) * (1 + 0.06 * noise),
                       np.array(COLORS["grass"]) * (1 + 0.14 * noise))
        col[line] = COLORS["line"]
        img[g] = col
        lab[g] = np.where(line, LINE, np.where(on_road, ROAD, GRASS))
        depth[g] = t

        for ob in world.obstacles:
            uid, kind, size, height, rgb = appearance(ob, world)
            if (ob.x - cx) * ch + (ob.y - cy) * sh < -size - 1:
                continue                                   # behind the camera
            if kind == "person":
                hit, t_hit, shade, head = self._cylinder(cx, cy, dx, dy, dz, ob.x, ob.y, size, height)
            else:
                hit, t_hit, shade, head = self._box(cx, cy, dx, dy, dz, ob.x, ob.y, size, height)
            hit &= t_hit < depth
            if not hit.any():
                continue
            base = np.where(head[hit][:, None], np.array(COLORS["skin"]), np.array(rgb))
            img[hit] = base * shade[hit][:, None]
            lab[hit] = OBSTACLE
            depth[hit] = t_hit[hit]

        out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return (out, lab) if labels else out

    @staticmethod
    @np.errstate(invalid="ignore")
    def _cylinder(ox, oy, dx, dy, dz, cx, cy, r, height):
        fx, fy = ox - cx, oy - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        ok = disc >= 0
        t = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0))) / (2 * a), np.inf)
        z = CAM_HEIGHT + t * dz
        hit = ok & (t > 0) & (z >= 0) & (z <= height)
        nx = (fx + t * dx) / r
        ny = (fy + t * dy) / r
        shade = 0.55 + 0.45 * np.clip(nx * SUN[0] + ny * SUN[1] + 0.3, 0, 1)
        return hit, t, shade, z > height - HEAD

    @staticmethod
    def _box(ox, oy, dx, dy, dz, cx, cy, a, height):
        with np.errstate(divide="ignore", invalid="ignore"):
            tx1, tx2 = (cx - a - ox) / dx, (cx + a - ox) / dx
            ty1, ty2 = (cy - a - oy) / dy, (cy + a - oy) / dy
            tz1, tz2 = (0 - CAM_HEIGHT) / dz, (height - CAM_HEIGHT) / dz
        tmins = np.stack([np.minimum(tx1, tx2), np.minimum(ty1, ty2), np.minimum(tz1, tz2)])
        tmaxs = np.stack([np.maximum(tx1, tx2), np.maximum(ty1, ty2), np.maximum(tz1, tz2)])
        tmins = np.nan_to_num(tmins, nan=-np.inf)
        tmaxs = np.nan_to_num(tmaxs, nan=np.inf)
        tnear, tfar = tmins.max(0), tmaxs.min(0)
        face = tmins.argmax(0)
        hit = (tnear <= tfar) & (tnear > 0)
        # simple sun shading: top face brightest, sides by their normal
        sx = np.where(dx > 0, -SUN[0], SUN[0])
        sy = np.where(dy > 0, -SUN[1], SUN[1])
        shade = np.select([face == 0, face == 1], [0.55 + 0.45 * np.clip(sx + 0.3, 0, 1),
                                                   0.55 + 0.45 * np.clip(sy + 0.3, 0, 1)], 1.0)
        return hit, np.where(hit, tnear, np.inf), shade, np.zeros_like(hit)
