"""Pygame window: top-down road view plus a live panel of the fly's neurons."""
import math
import os
import time
from collections import deque

import numpy as np
import pygame

from .controller import LaneFollower, drive_command
from .world import World, ROAD_HALF_WIDTH, CAR_RADIUS

SEEN_COLORS = np.array([(70, 110, 165), (48, 96, 40), (62, 62, 66), (235, 235, 230), (255, 70, 60)],
                       dtype=np.uint8)   # sky, grass, road, marking, something in the way

W, H = 1280, 760
VIEW_W = 820
SCALE = 9.0  # pixels per metre

GRASS = (58, 92, 58)
ROAD = (70, 70, 76)
LINE = (230, 230, 230)
PANEL = (22, 24, 30)
TEXT = (230, 232, 238)
DIM = (140, 145, 155)
ACCENT = (255, 190, 60)
RED = (235, 70, 70)
BLUE = (90, 170, 255)
GREEN = (90, 210, 120)

HELP = ("1 fly  2 formula  3 none  4 trained  5 eyes  6 teacher  7 RL fly  8 RL eyes|"
        "SPACE pause   O obstacle   R reset   V camera view   S screenshot   ESC quit")
KEY_KINDS = {"1": "fly", "2": "direct", "3": "none", "4": "trained", "5": "eyes", "6": "teacher",
             "7": "rl", "8": "rl-eyes"}
KIND_LABELS = {"fly": "fly brain (hand-tuned)", "direct": "hand-written formula", "none": "no reflex",
               "trained": "fly brain (trained)", "eyes": "eyes only (trained)", "teacher": "teacher",
               "rl": "fly brain (RL)", "rl-eyes": "eyes only (RL)"}
BRAIN_KINDS = ("fly", "trained", "rl")


class App:
    def __init__(self, make_reflex, args, dt, make_senses=None):
        self.make_reflex, self.args, self.dt = make_reflex, args, dt
        self.make_senses = make_senses
        self.senses = None
        self.show_seen = False
        pygame.init()
        pygame.display.set_caption("FlyDrive - fruit fly connectome reflexes")
        self.screen = pygame.display.set_mode((W, H))
        self.font = pygame.font.SysFont("dejavusans,liberationsans,arial", 16)
        self.small = pygame.font.SysFont("dejavusans,liberationsans,arial", 13)
        self.big = pygame.font.SysFont("dejavusans,liberationsans,arial", 24, bold=True)
        self.kind = args.reflex
        self.dn_names = None
        self.paused = False
        self.flash = 0.0
        self.gf_trace = deque(maxlen=250)
        self.reset()

    def reset(self):
        self.world = World(seed=self.args.seed, difficulty=getattr(self.args, "difficulty", "normal"))
        self.world.car.speed = 12.0
        if self.make_senses is not None:
            self.lane, self.senses = self.make_senses(self.world, self.args)
        else:
            self.lane, self.senses = LaneFollower(self.world), None
        if self.kind in BRAIN_KINDS:
            self.draw_message("Loading fly brain... (139k neurons)")
        try:
            self.reflex = self.make_reflex(self.kind, self.args, self.world)
        except SystemExit as e:  # e.g. no trained model yet
            print(e)
            self.kind = "fly"
            self.reflex = self.make_reflex(self.kind, self.args, self.world)
        self.gf_trace.clear()

    def draw_message(self, msg):
        self.screen.fill(PANEL)
        t = self.big.render(msg, True, TEXT)
        self.screen.blit(t, t.get_rect(center=(W // 2, H // 2)))
        pygame.display.flip()
        pygame.event.pump()

    # ---------- coordinate helpers ----------
    def to_screen(self, x, y):
        c = self.world.car
        return (VIEW_W / 2 + (x - c.x) * SCALE, H / 2 - (y - c.y) * SCALE)

    # ---------- drawing ----------
    def draw_world(self):
        w, c = self.world, self.world.car
        self.screen.fill(GRASS, (0, 0, VIEW_W, H))
        i = w.nearest_index(c.x, c.y)
        idx = [(i + k) % len(w.track) for k in range(-70, 90)]
        pts = [self.to_screen(*w.track[j]) for j in idx]
        road_px = int(2 * ROAD_HALF_WIDTH * SCALE)
        pygame.draw.lines(self.screen, ROAD, False, pts, road_px)
        for p in pts:
            pygame.draw.circle(self.screen, ROAD, p, road_px // 2)
        for k in range(0, len(pts) - 3, 6):  # dashed centre line
            pygame.draw.line(self.screen, LINE, pts[k], pts[k + 3], 2)

        # field of view wedge
        for sgn in (-1, 1):
            a = c.heading + sgn * math.radians(75)
            end = self.to_screen(c.x + 30 * math.cos(a), c.y + 30 * math.sin(a))
            pygame.draw.line(self.screen, (110, 120, 110), self.to_screen(c.x, c.y), end, 1)

        for ob in w.obstacles:
            p = self.to_screen(ob.x, ob.y)
            r = max(3, int(ob.radius * SCALE))
            if ob.vx or ob.vy:
                pygame.draw.circle(self.screen, RED if ob.hit else BLUE, p, r)
            else:
                rect = pygame.Rect(0, 0, 2 * r, 2 * r)
                rect.center = p
                pygame.draw.rect(self.screen, RED if ob.hit else ACCENT, rect, border_radius=3)

        # car
        L, Wd = 4.4 * SCALE, 2.0 * SCALE
        surf = pygame.Surface((L, Wd), pygame.SRCALPHA)
        body = (255, 255, 255) if self.flash <= 0 else RED
        pygame.draw.rect(surf, body, (0, 0, L, Wd), border_radius=6)
        pygame.draw.rect(surf, (40, 60, 90), (L * 0.55, 3, L * 0.22, Wd - 6), border_radius=3)
        rot = pygame.transform.rotate(surf, math.degrees(c.heading))
        self.screen.blit(rot, rot.get_rect(center=self.to_screen(c.x, c.y)))

        if self.senses is not None and self.senses.frame is not None:
            self.draw_camera()
        if self.paused:
            t = self.big.render("PAUSED", True, TEXT)
            self.screen.blit(t, (20, 20))

    def draw_camera(self, scale=2):
        """Inset: the camera image the car is actually driving on."""
        v = self.senses.vision
        img = self.senses.frame
        if self.show_seen and self.senses.labels is not None:
            img = SEEN_COLORS[self.senses.labels]
        h, w = img.shape[:2]
        surf = pygame.surfarray.make_surface(np.repeat(np.repeat(img, scale, 0), scale, 1)
                                             .transpose(1, 0, 2))
        x0, y0 = (VIEW_W - w * scale) // 2, 10
        pygame.draw.rect(self.screen, (15, 16, 20), (x0 - 3, y0 - 3, w * scale + 6, h * scale + 6))
        self.screen.blit(surf, (x0, y0))
        for u0, r0, u1, r1, loom in v.blobs:          # what it thinks is in the way
            pygame.draw.rect(self.screen, RED if loom else ACCENT,
                             (x0 + u0 * scale, y0 + r0 * scale,
                              max(3, (u1 - u0) * scale), max(3, (r1 - r0) * scale)), 1)
        from .camera import FOCAL                      # where it thinks the road centre is
        col = w / 2 - FOCAL * v.lane_L / max(v.lane_D, 0.1)
        row = h / 2 + FOCAL * 1.2 / max(v.lane_D, 0.1)
        pygame.draw.circle(self.screen, GREEN, (x0 + col * scale, y0 + row * scale), 4, 1)
        cap = "what the car sees" + (" (classified)" if self.show_seen else "")
        self.screen.blit(self.small.render(cap, True, DIM), (x0, y0 + h * scale + 4))

    def bar(self, x, y, label, value, vmax, color, unit="", fmt="{:6.1f}"):
        self.screen.blit(self.small.render(label, True, DIM), (x, y))
        bx, bw = x + 150, 220
        pygame.draw.rect(self.screen, (50, 53, 62), (bx, y + 2, bw, 12), border_radius=4)
        frac = max(0.0, min(1.0, value / vmax)) if vmax else 0
        pygame.draw.rect(self.screen, color, (bx, y + 2, int(bw * frac), 12), border_radius=4)
        self.screen.blit(self.small.render(fmt.format(value) + unit, True, TEXT), (bx + bw + 8, y))

    def draw_panel(self, sim_speed):
        x0 = VIEW_W
        self.screen.fill(PANEL, (x0, 0, W - x0, H))
        x = x0 + 24
        y = 18
        self.screen.blit(self.big.render("FlyDrive", True, ACCENT), (x, y))
        self.screen.blit(self.font.render(f"Reflex: {KIND_LABELS[self.kind]}", True, TEXT), (x + 120, y + 6))
        y += 44

        s, c = self.world.stats, self.world.car
        lines = [
            f"time {self.world.t:6.1f} s     speed {c.speed * 3.6:5.1f} km/h     sim speed x{sim_speed:.2f}",
            f"obstacles {s['obstacles']}     collisions {s['collisions']}",
        ]
        for ln in lines:
            self.screen.blit(self.font.render(ln, True, TEXT), (x, y))
            y += 24
        y += 10

        info = self.reflex.info
        self.screen.blit(self.font.render("Eyes (looming, rad/s)", True, ACCENT), (x, y)); y += 26
        self.bar(x, y, "left field", info.get("loom_L", 0), 0.4, BLUE, fmt="{:6.3f}"); y += 22
        self.bar(x, y, "right field", info.get("loom_R", 0), 0.4, BLUE, fmt="{:6.3f}"); y += 34

        if self.kind == "fly":
            self.screen.blit(self.font.render("Fly neurons (firing rate)", True, ACCENT), (x, y)); y += 26
            self.bar(x, y, "LPLC2/LC4 left", info.get("LPLC2/LC4 L Hz", 0), 200, BLUE, " Hz"); y += 22
            self.bar(x, y, "LPLC2/LC4 right", info.get("LPLC2/LC4 R Hz", 0), 200, BLUE, " Hz"); y += 22
            self.bar(x, y, "Giant Fiber", info.get("GiantFiber Hz", 0), 120, RED, " Hz"); y += 22
            self.bar(x, y, "escape DNs left", info.get("escDN L Hz", 0), 80, GREEN, " Hz"); y += 22
            self.bar(x, y, "escape DNs right", info.get("escDN R Hz", 0), 80, GREEN, " Hz"); y += 34
        if self.kind in ("trained", "eyes", "rl", "rl-eyes"):
            rl_mode = self.kind.startswith("rl")
            title = "Learned policy (reinforcement learning)" if rl_mode else "Trained readout"
            self.screen.blit(self.font.render(title, True, ACCENT), (x, y)); y += 26
            label = "escape urge" if rl_mode else "threat score"
            self.bar(x, y, label, max(0, info.get("threat score", 0)), 1.0, RED, fmt="{:6.2f}"); y += 22
            d = info.get("direction score", 0)
            self.bar(x, y, "dodge direction", abs(d), 1.0, GREEN if d > 0 else BLUE,
                     unit=" L" if d > 0 else " R", fmt="{:6.2f}"); y += 30
        if self.kind in ("trained", "rl"):
            if self.dn_names is None:
                from .connectome import dn_names
                self.dn_names = dn_names()
            x_feat = self.reflex.last_x
            rates = np.expm1(x_feat) if x_feat is not None else None
            self.screen.blit(self.font.render("Most active descending neurons", True, ACCENT), (x, y)); y += 24
            if rates is not None and rates.max() > 0.5:
                for i in np.argsort(-rates)[:5]:
                    if rates[i] < 0.5:
                        break
                    self.bar(x, y, self.dn_names[i], rates[i], 150, GREEN, " Hz"); y += 20
            else:
                self.screen.blit(self.small.render("(quiet: nothing looming)", True, DIM), (x, y)); y += 20
            y += 12
        self.screen.blit(self.font.render("Car", True, ACCENT), (x, y)); y += 26
        self.bar(x, y, "brake", info.get("brake", 0), 1.0, RED, fmt="{:6.2f}"); y += 22
        off = getattr(self.world, "dodge_offset", 0.0)
        self.bar(x, y, "dodge left", max(0, off), 3.5, GREEN, " m"); y += 22
        self.bar(x, y, "dodge right", max(0, -off), 3.5, GREEN, " m"); y += 34

        if self.kind == "fly":
            self.screen.blit(self.font.render("Giant Fiber activity, last 5 s", True, ACCENT), (x, y)); y += 24
            gh, gw = 90, 400
            pygame.draw.rect(self.screen, (34, 37, 45), (x, y, gw, gh), border_radius=6)
            if len(self.gf_trace) > 1:
                pts = [(x + gw * k / 249, y + gh - 4 - (gh - 8) * min(1, v / 120))
                       for k, v in enumerate(self.gf_trace)]
                pygame.draw.lines(self.screen, RED, False, pts, 2)
            y += gh + 10

        for k, part in enumerate(HELP.split("|")):
            self.screen.blit(self.small.render(part.strip(), True, DIM), (x, H - 44 + k * 18))

    # ---------- main loop ----------
    def switch(self, kind):
        if kind != self.kind:
            self.kind = kind
            self.reset()

    def run(self, max_frames=None, screenshot=None):
        clock = pygame.time.Clock()
        running, frames, sim_speed = True, 0, 1.0
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif ev.unicode in KEY_KINDS:
                        self.switch(KEY_KINDS[ev.unicode])
                    elif ev.key == pygame.K_v:
                        self.show_seen = not self.show_seen
                    elif ev.key == pygame.K_o:
                        self.world.spawn_obstacle()
                    elif ev.key == pygame.K_r:
                        self.reset()
                    elif ev.key == pygame.K_s:
                        name = f"flydrive_{int(time.time())}.png"
                        pygame.image.save(self.screen, name)
                        print("saved", name)

            if not self.paused:
                if self.senses is not None:
                    self.senses.hud = f"{KIND_LABELS[self.kind]}  {self.world.car.speed * 3.6:.0f} km/h"
                t0 = time.perf_counter()
                steer, accel = drive_command(self.lane, self.reflex, self.world, self.dt,
                                             self.senses)
                if self.world.step(steer, accel, self.dt):
                    self.flash = 0.4
                self.flash -= self.dt
                self.gf_trace.append(self.reflex.info.get("GiantFiber Hz", 0.0))
                took = time.perf_counter() - t0
                sim_speed = 0.9 * sim_speed + 0.1 * min(1.0, self.dt / max(took, 1e-6))

            self.draw_world()
            self.draw_panel(sim_speed)
            pygame.display.flip()
            clock.tick(int(1 / self.dt))
            frames += 1
            if max_frames and frames >= max_frames:
                if screenshot:
                    pygame.image.save(self.screen, screenshot)
                running = False
        pygame.quit()


def run_window(make_reflex, args, dt, make_senses=None):
    app = App(make_reflex, args, dt, make_senses)
    app.run(max_frames=getattr(args, "frames", None), screenshot=getattr(args, "screenshot", None))
