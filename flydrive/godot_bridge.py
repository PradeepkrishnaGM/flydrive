"""Use the Godot 3D world as the car's eyes.

Python stays in charge of the world (car physics, obstacles, collisions) so results
match every other mode. Godot only draws it, and hands back what the car's camera
sees. One round trip per simulation step, over a local TCP connection:

    Python  --{car, obstacles, HUD text}-->  Godot     (JSON, 4-byte length prefix)
    Python  <--{camera image, raw RGB}-----  Godot     (4-byte length prefix)

The Godot project lives in godot_world/. Start it yourself (open the project and
press Play) or let this class launch it.
"""
import json
import shlex
import shutil
import socket
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import camera as cam

PROJECT_DIR = Path(__file__).resolve().parent.parent / "godot_world"
CANDIDATES = ["godot", "godot4", "godot-4", "Godot", "godot-editor"]


class GodotRenderer:
    def __init__(self, port=47111, host="127.0.0.1", godot_bin=None, launch=True,
                 timeout=180.0, project=PROJECT_DIR):
        self.project = Path(project)
        self.proc = None
        self.said_hello = False
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        self.srv.settimeout(timeout)
        if launch:
            self._launch(godot_bin, port)
        else:
            print(f"Open {self.project} in Godot and press Play (waiting on port {port})...",
                  flush=True)
        try:
            self.conn, _ = self.srv.accept()
        except socket.timeout:
            raise SystemExit(
                f"Godot never connected on port {port}.\n"
                f"  Install Godot 4 (Fedora: sudo dnf install godot), then either\n"
                f"  run:  python run.py --eyes camera --renderer godot --godot /path/to/godot\n"
                f"  or open {self.project} in the Godot editor, press Play, and add --no-launch.")
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Godot connected.", flush=True)

    def _launch(self, godot_bin, port):
        # --godot may be a path or a whole command, e.g. "flatpak run org.godotengine.Godot"
        cmd = shlex.split(godot_bin) if godot_bin else None
        if cmd is None:
            found = next((shutil.which(c) for c in CANDIDATES if shutil.which(c)), None)
            cmd = [found] if found else None
        binary = cmd[0] if cmd else None
        if binary is None:
            print("Could not find a 'godot' executable in PATH.\n"
                  f"  Install Godot 4 (Fedora: sudo dnf install godot), pass --godot /path/to/godot,\n"
                  f"  or open {self.project} in Godot, press Play, and add --no-launch.\n"
                  f"  Waiting on port {port} in case you start it yourself...", flush=True)
            return
        log = self.project / "godot.log"
        full = cmd + ["--path", str(self.project)]
        print(f"Starting Godot: {' '.join(full)}   (its output goes to {log})", flush=True)
        self.proc = subprocess.Popen(full, stdout=open(log, "wb"), stderr=subprocess.STDOUT)

    # ---- wire format ----
    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.conn.sendall(struct.pack("<I", len(data)) + data)

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise SystemExit("The Godot world closed the connection.")
            buf += chunk
        return bytes(buf)

    def _hello(self, world):
        self._send({
            "type": "hello",
            "image": [cam.IMG_W, cam.IMG_H],
            "hfov": cam.HFOV_DEG,
            "cam": [cam.CAM_FORWARD, cam.CAM_HEIGHT],
            "track": [[round(float(x), 3), round(float(y), 3)] for x, y in world.track],
            "road_half": float(cam.ROAD_HALF_WIDTH),
            "center_line_half": cam.CENTER_LINE_HALF,
            "dash": [cam.DASH_PERIOD, cam.DASH_ON],
            "edge_line": list(cam.EDGE_LINE),
            "colors": {k: list(v) for k, v in cam.COLORS.items()},
        })
        struct.unpack("<I", self._recv(4))[0]        # ack
        self.said_hello = True

    def render(self, world, hud="", **_):
        if not self.said_hello:
            self._hello(world)
        c = world.car
        obstacles = []
        for ob in world.obstacles:
            uid, kind, size, height, rgb = cam.appearance(ob, world)
            obstacles.append([uid, 1 if kind == "person" else 0, round(ob.x, 4), round(ob.y, 4),
                              round(size, 3), round(height, 3), *[round(v, 3) for v in rgb]])
        self._send({"type": "frame", "car": [round(c.x, 4), round(c.y, 4), round(c.heading, 6)],
                    "obstacles": obstacles, "hud": hud})
        n = struct.unpack("<I", self._recv(4))[0]
        want = cam.IMG_W * cam.IMG_H * 3
        if n != want:
            raise SystemExit(f"Godot sent {n} bytes, expected {want} "
                             f"({cam.IMG_W}x{cam.IMG_H} RGB). Check the image size in main.gd.")
        return np.frombuffer(self._recv(n), dtype=np.uint8).reshape(cam.IMG_H, cam.IMG_W, 3)

    def close(self):
        try:
            self._send({"type": "quit"})
        except Exception:
            pass
        for closer in (getattr(self, "conn", None), self.srv):
            try:
                closer.close()
            except Exception:
                pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


if __name__ == "__main__":       # quick check: python -m flydrive.godot_bridge
    from .world import World
    r = GodotRenderer(launch="--no-launch" not in sys.argv)
    w = World(seed=1)
    w.car.speed = 12.0
    img = r.render(w, hud="bridge test")
    print("received frame", img.shape, "mean colour", img.reshape(-1, 3).mean(0).round(1))
    r.close()
