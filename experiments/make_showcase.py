"""Make the README GIF, a side-by-side video and social images, all headless.

    python experiments/make_showcase.py find                  # find a layout where the lesioned car crashes
    python experiments/make_showcase.py build --seed 3 --t 12.4   # record, then build everything

Outputs: docs/demo.gif, docs/social-preview.png (GitHub, 1280x640), docs/linkedin-card.png
(1200x627), docs/linkedin-square.png (1080x1080), and showcase/lesion-vs-intact.mp4 (kept out
of git). The video compares the hand-tuned fly brain intact against the same brain with its 10
escape neurons silenced, driving the same layout.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root: run.py, flydrive/
import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DOCS, SHOW = ROOT / "docs", ROOT / "showcase"
WORK = Path(os.environ.get("SHOWCASE_WORK", "/tmp/flydrive_showcase"))
LESION = "gf_left,gf_right,esc_left,esc_right"
BG, INK, MUTED, ACCENT, BLUE = (22, 24, 30), (240, 239, 236), (170, 170, 164), (255, 190, 60), (57, 135, 229)
FONT = "/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf"


def font(size, bold=False):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_axes([700 if bold else 400])
    except Exception:
        pass
    return f


def new_app(lesion, seed):
    import run
    from flydrive.render import App
    args = run.build_parser().parse_args(["--reflex", "fly", "--seed", str(seed), "--lesion", lesion])
    return App(run.make_reflex, args, run.DT, run.make_senses)


def step(app):
    from flydrive.controller import drive_command
    steer, accel = drive_command(app.lane, app.reflex, app.world, app.dt, app.senses)
    if app.world.step(steer, accel, app.dt):
        app.flash = 0.4
    app.flash -= app.dt
    app.gf_trace.append(app.reflex.info.get("GiantFiber Hz", 0.0))


def first_collision(lesion, seed, seconds):
    app = new_app(lesion, seed)
    for s in range(int(seconds / app.dt)):
        step(app)
        if app.world.stats["collisions"]:
            return (s + 1) * app.dt, app
    return None, app


def find(seeds, seconds):
    """Seeds where the lesioned car crashes and the intact one stays clear for 4 s after."""
    for seed in seeds:
        tl, _ = first_collision(LESION, seed, seconds)
        ti, _ = first_collision("", seed, (tl or 0) + 4.0) if tl else (None, None)
        print(f"seed {seed}: lesioned first crash {tl and round(tl, 1)} s, "
              f"intact crash by then: {ti is not None}", flush=True)


def record(lesion, seed, t_start, t_end, every, out):
    """Save every `every`-th frame between t_start and t_end (seconds) as PNGs."""
    import pygame
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    app = new_app(lesion, seed)
    n = 0
    for s in range(int(t_end / app.dt)):
        step(app)
        t = (s + 1) * app.dt
        if t >= t_start and s % every == 0:
            app.draw_world()
            app.draw_panel(1.0)
            pygame.image.save(app.screen, str(out / f"f{n:04d}.png"))
            n += 1
    return n


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def make_gif(frames, out, fps, width):
    pal = WORK / "palette.png"
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    ffmpeg("-framerate", "25", "-i", str(frames / "f%04d.png"), "-vf", f"{scale},palettegen=max_colors=96", str(pal))
    ffmpeg("-framerate", "25", "-i", str(frames / "f%04d.png"), "-i", str(pal), "-lavfi",
           f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", str(out))


def side_by_side(intact, lesioned, n, out_mp4):
    """1920x1080 frames: intact (left) vs escape neurons silenced (right)."""
    comp = WORK / "comp"
    comp.mkdir(parents=True, exist_ok=True)
    for f in comp.glob("*.png"):
        f.unlink()
    k = 0.72
    for i in range(n):
        im = Image.new("RGB", (1920, 1080), BG)
        d = ImageDraw.Draw(im)
        for col, (src, title, sub, colour) in enumerate((
                (intact, "Intact fly brain", "dodges the obstacle", BLUE),
                (lesioned, "10 escape neurons silenced", "same road, same obstacle", ACCENT))):
            x0 = 0 if col == 0 else 960
            wx = x0 + 20 if col == 0 else x0 + 10
            d.text((x0 + 40, 36), title, font=font(52, True), fill=colour)
            d.text((x0 + 40, 104), sub, font=font(30), fill=MUTED)
            fr = Image.open(src / f"f{i:04d}.png").resize((int(1280 * k), int(760 * k)), Image.LANCZOS)
            im.paste(fr, (wx, 170))
        d.text((40, 800), "FlyDrive: a car whose reflexes run on the real fruit-fly connectome "
               "(139,000 neurons, 15 million synapses)", font=font(34), fill=INK)
        d.text((40, 860), "Hand-tuned reflex, 29 obstacles over 3 layouts: 6 collisions intact, "
               "27 with the escape neurons silenced.", font=font(34), fill=INK)
        d.text((40, 960), "github.com/PradeepkrishnaGM/flydrive", font=font(30), fill=MUTED)
        im.save(comp / f"c{i:04d}.png")
    ffmpeg("-framerate", "25", "-i", str(comp / "c%04d.png"), "-c:v", "libopenh264", "-pix_fmt", "yuv420p",
           "-b:v", "6M", "-movflags", "+faststart", str(out_mp4))


def card(size, shot, headline, lines, out, shot_frac=0.52):
    w, h = size
    im = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(im)
    pad = int(h * 0.09)
    sw = int(w * shot_frac)
    s = Image.open(shot).convert("RGB")
    s = s.crop((0, 120, 820, 700))                     # the road view, no side panel
    s = s.resize((sw, int(sw * s.height / s.width)), Image.LANCZOS)
    im.paste(s, (w - sw - pad, (h - s.height) // 2))
    tw = w - sw - 3 * pad
    d.text((pad, pad), "FlyDrive", font=font(int(h * 0.14), True), fill=ACCENT)
    y = pad + int(h * 0.2)
    for ln in headline:
        d.text((pad, y), ln, font=font(int(h * 0.066), True), fill=INK)
        y += int(h * 0.085)
    y += int(h * 0.03)
    for ln in lines:
        d.text((pad, y), ln, font=font(int(h * 0.043)), fill=MUTED)
        y += int(h * 0.058)
    d.text((pad, h - pad - int(h * 0.04)), "github.com/PradeepkrishnaGM/flydrive", font=font(int(h * 0.04)), fill=MUTED)
    im.save(out)


def square(out):
    im = Image.new("RGB", (1080, 1080), BG)
    d = ImageDraw.Draw(im)
    d.text((60, 50), "Silence 10 escape neurons.", font=font(64, True), fill=INK)
    d.text((60, 128), "The hand-tuned car crashes.", font=font(64, True), fill=ACCENT)
    chart = Image.open(DOCS / "lesion.png").convert("RGB")
    chart = chart.resize((1000, int(1000 * chart.height / chart.width)), Image.LANCZOS)
    im.paste(chart, (40, 240))
    y = 240 + chart.height + 36
    d.text((60, y), "The learned car doesn't: it reads all 1,299 output neurons.", font=font(34, True), fill=INK)
    y += 70
    for ln in ("A driving simulator whose reflexes run on the real fruit-fly",
               "connectome: 139,000 neurons, 15 million synapses."):
        d.text((60, y), ln, font=font(30), fill=MUTED)
        y += 44
    d.text((60, 1080 - 70), "github.com/PradeepkrishnaGM/flydrive", font=font(30), fill=MUTED)
    im.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["find", "build"])
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t", type=float, default=10.0, help="time of the lesioned car's crash (s)")
    a = ap.parse_args()
    if a.action == "find":
        find([int(s) for s in a.seeds.split(",")], a.seconds)
        return
    WORK.mkdir(parents=True, exist_ok=True)
    SHOW.mkdir(exist_ok=True)
    t0, t1 = max(0.0, a.t - 3.5), a.t + 2.5
    n = record("", a.seed, t0, t1, 2, WORK / "intact")
    n2 = record(LESION, a.seed, t0, t1, 2, WORK / "lesion")
    n = min(n, n2)
    print(f"recorded {n} frames per run")
    make_gif(WORK / "intact", DOCS / "demo.gif", fps=12, width=760)
    side_by_side(WORK / "intact", WORK / "lesion", n, SHOW / "lesion-vs-intact.mp4")
    shot = DOCS / "hard-world.png"
    card((1280, 640), shot, ["A car driven by the", "real fruit-fly brain"],
         ["139,000 neurons, 15 million synapses", "Looming neurons to escape neurons", "to steering and braking"],
         DOCS / "social-preview.png")
    card((1200, 627), shot, ["Silence 10 neurons.", "The car crashes."],
         ["Hand-tuned reflex, 29 obstacles:", "6 collisions intact, 27 silenced", "Real FlyWire connectome"],
         DOCS / "linkedin-card.png")
    square(DOCS / "linkedin-square.png")
    for f in (DOCS / "demo.gif", SHOW / "lesion-vs-intact.mp4"):
        print(f, f"{f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
