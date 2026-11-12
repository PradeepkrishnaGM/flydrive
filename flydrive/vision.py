"""Driving with the eyes alone: road following and looming from camera pixels.

Nothing here reads the world's geometry. Every decision comes from the image:

1. Look at each pixel's colour. Pixels that look like road, road markings, grass or
   sky are "background"; anything else is "something in the way". What road, grass
   and sky look like is learned from the first frame (the car starts on an empty
   road), and keeps adapting slowly, so it copes with different lighting such as
   Godot's shading and shadows.
2. Road following: a few rows of the image correspond to the road 5-8 m ahead
   (because the ground is flat and the camera height is known, like a human's
   sense of how far away the road is). In each row, find where the road starts and
   ends and steer for the middle, plus whatever sideways dodge the reflex wants.
3. Looming: group "something in the way" pixels into blobs, follow each blob from
   frame to frame, and measure how fast its apparent width grows. That growth rate
   is looming, the same signal fly LPLC2/LC4 neurons respond to, and it feeds the
   brain exactly as the geometric looming did before.
"""
import math

import numpy as np
from scipy import ndimage

from .camera import (CAM_FORWARD, CAM_HEIGHT, FOCAL, GRASS, HORIZON_ROW, IMG_H, IMG_W, LINE,
                     NumpyRenderer, OBSTACLE, ROAD, SKY)
from .controller import CRUISE_SPEED
from .world import ROAD_HALF_WIDTH

LANE_ROWS = list(range(64, 78, 2))      # image rows looking about 8 m down to 4.6 m ahead
TTC_MAX = 2.5                           # like LPLC2: only react to things arriving soon
MIN_BLOB = 4                            # pixels
HISTORY = 5                             # frames used to measure growth (100 ms)


def _chroma(f):
    V = f.mean(-1)
    return (f - V[..., None]) / (V[..., None] + 0.08), V


class Vision:
    def __init__(self, w=IMG_W, h=IMG_H):
        self.w, self.h = w, h
        self.proto = None
        self.flip = False
        self.tracks = {}
        self.next_id = 1
        self.lane_L = 0.0                   # road centre, metres to the left, at LANE_D
        self.lane_D = 6.0
        self.lane_seen = False
        self.blobs = []                     # for drawing: (u0, v0, u1, v1, looming?)
        self.above = (np.arange(h) < HORIZON_ROW)[:, None]
        self.row_D = {r: CAM_HEIGHT * FOCAL / (r + 0.5 - h / 2) for r in LANE_ROWS}

    # ---------------- 1. what is each pixel? ----------------
    def calibrate(self, img):
        f = img.astype(np.float32) / 255
        # Is the image upside down? (some renderers read back flipped) Sky is bluer.
        top, bot = f[:6].mean((0, 1)), f[-6:].mean((0, 1))
        self.flip = (bot[2] - bot[0]) > (top[2] - top[0]) + 0.1
        if self.flip:
            f = f[::-1]
        road = np.median(f[84:96, 80:240].reshape(-1, 3), 0)
        sky = np.median(f[:6].reshape(-1, 3), 0)
        grass = np.median(np.concatenate([f[53:59, :80], f[53:59, 240:]], 1).reshape(-1, 3), 0)
        self.proto = {}
        for k, c in (("road", road), ("grass", grass), ("sky", sky)):
            ch, V = _chroma(c[None])
            self.proto[k] = (ch[0], float(V[0]))

    def classify(self, img):
        if self.proto is None:
            self.calibrate(img)
        if self.flip:
            img = img[::-1]
        f = img.astype(np.float32) / 255
        ch, V = _chroma(f)
        road_c, road_V = self.proto["road"]
        grass_c, _ = self.proto["grass"]
        d_road = np.linalg.norm(ch - road_c, axis=-1)
        d_grass = np.linalg.norm(ch - grass_c, axis=-1)
        gray = d_road < 0.10
        grass = d_grass < 0.20
        bluish = ch[..., 2] >= ch[..., 0] - 0.02
        lab = np.full(V.shape, OBSTACLE, dtype=np.uint8)
        lab[gray & (V > 0.35 * road_V)] = ROAD
        lab[gray & (V >= 1.6 * road_V)] = LINE
        lab[grass] = GRASS
        lab[self.above & (bluish | (gray & (V > 0.6))) & ~grass] = SKY
        self._adapt(ch, V, lab)
        return lab

    def _adapt(self, ch, V, lab, rate=0.02):
        """Slowly follow changes in lighting, using pixels we're confident about."""
        for k, cls, region in (("road", ROAD, (slice(84, 96), slice(100, 220))),
                               ("grass", GRASS, (slice(53, 60), slice(None)))):
            m = lab[region] == cls
            if m.sum() > 50:
                c0, v0 = self.proto[k]
                self.proto[k] = (c0 + rate * (ch[region][m].mean(0) - c0),
                                 v0 + rate * (float(V[region][m].mean()) - v0))

    # ---------------- 2. where is the road? ----------------
    def lane(self, lab):
        centres = []
        drivable = lab >= ROAD                     # road, marking, or something on it
        for r, D in self.row_D.items():
            row = drivable[r]
            u = int(np.clip(round(self.w / 2 - FOCAL * self.lane_L / D), 0, self.w - 1))
            if not row[u]:
                near = np.flatnonzero(row[max(0, u - 60):u + 61])
                if not len(near):
                    continue
                u = max(0, u - 60) + near[np.argmin(np.abs(near + max(0, u - 60) - u))]
            left = u
            while left > 0 and row[left - 1]:
                left -= 1
            right = u
            while right < self.w - 1 and row[right + 1]:
                right += 1
            L_left = -D * (left - self.w / 2) / FOCAL            # metres to the left
            L_right = -D * (right + 1 - self.w / 2) / FOCAL
            see_left, see_right = left > 0, right < self.w - 1
            if see_left and see_right:
                if not 7.0 < L_left - L_right < 13.5:
                    continue
                centres.append((D, (L_left + L_right) / 2))
            elif see_left:                                       # one edge: we know roads are 10 m wide
                centres.append((D, L_left - ROAD_HALF_WIDTH))
            elif see_right:
                centres.append((D, L_right + ROAD_HALF_WIDTH))
        self.lane_seen = bool(centres)
        if centres:
            D = float(np.mean([c[0] for c in centres]))
            L = float(np.median([c[1] for c in centres]))
            self.lane_L += 0.6 * (L - self.lane_L)
            self.lane_D = D
        return self.lane_L, self.lane_D

    # ---------------- 3. what is looming? ----------------
    def looming(self, lab, t):
        mask = lab == OBSTACLE
        comps, n = ndimage.label(mask, structure=np.ones((3, 3)))
        found = []
        if n:
            for i, sl in enumerate(ndimage.find_objects(comps)):
                area = int((comps[sl] == i + 1).sum())
                if area < MIN_BLOB:
                    continue
                rows = sl[0].stop - sl[0].start
                width = area / rows                              # mean width, sub-pixel
                cols = np.nonzero(comps[sl] == i + 1)[1] + sl[1].start
                cu = cols.mean() + 0.5
                ul, ur = cu - width / 2, cu + width / 2
                theta = math.atan((ur - self.w / 2) / FOCAL) - math.atan((ul - self.w / 2) / FOCAL)
                bearing = -math.atan((cu - self.w / 2) / FOCAL)  # left positive
                found.append((cu, (sl[0].start + sl[0].stop) / 2, theta, bearing, sl, area))

        # follow blobs from frame to frame (nearest match)
        unmatched = set(self.tracks)
        for cu, cv, theta, bearing, sl, area in found:
            best, bd = None, 30.0
            for tid in unmatched:
                tr = self.tracks[tid]
                d = math.hypot(cu - tr["u"], cv - tr["v"])
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"hist": []}
            else:
                unmatched.discard(best)
            tr = self.tracks[best]
            tr.update(u=cu, v=cv, bearing=bearing, sl=sl, missing=0)
            tr["theta"] = theta
            tr["hist"] = (tr["hist"] + [(t, 0.5 * math.log(area))])[-HISTORY:]
        for tid in unmatched:
            self.tracks[tid]["missing"] = self.tracks[tid].get("missing", 0) + 1
            if self.tracks[tid]["missing"] > 3:
                del self.tracks[tid]

        left = right = 0.0
        self.blobs = []
        for tid, tr in self.tracks.items():
            if tr.get("missing"):
                continue
            h = tr["hist"]
            loom = False
            if len(h) >= 3:
                # relative growth rate of the blob's size, from its area (many pixels, so
                # much less jumpy than its width): e = d ln(size)/dt = 1 / time-to-contact
                ts = np.array([p[0] for p in h])
                ls = np.array([p[1] for p in h])
                e = np.polyfit(ts - ts[-1], ls, 1)[0]
                if e > 1.0 / TTC_MAX:
                    rate = tr["theta"] * e                   # looming, rad/s
                    w_left = 0.5 + 0.5 * math.tanh(tr["bearing"] / 0.08)
                    left += rate * w_left
                    right += rate * (1 - w_left)
                    loom = True
            sl = tr["sl"]
            self.blobs.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, loom))
        return left, right


class CameraSenses:
    """The car's only senses in camera mode: one camera (plus its own speedometer)."""

    def __init__(self, world, renderer=None):
        self.world = world
        self.renderer = renderer or NumpyRenderer()
        self.vision = Vision()
        self.frame = self.labels = None
        self.hud = ""

    def looming(self, dt):
        self.frame = self.renderer.render(self.world, hud=self.hud)
        self.labels = self.vision.classify(self.frame)
        self.vision.lane(self.labels)
        return self.vision.looming(self.labels, self.world.t)


class VisionLane:
    """Steer for the middle of the road the camera sees (pure pursuit), plus the dodge."""

    def __init__(self, senses):
        self.senses = senses
        self.world = senses.world

    def act(self, offset=0.0):
        v = self.senses.vision
        target = v.lane_L + offset
        ahead = v.lane_D + CAM_FORWARD
        alpha = math.atan2(target, ahead)
        steer = math.atan2(2 * 2.7 * math.sin(alpha), math.hypot(ahead, target))
        accel = 1.5 * (CRUISE_SPEED - self.world.car.speed)
        return steer, accel
