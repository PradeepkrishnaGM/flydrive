# FlyDrive's 3D world (Godot 4)

This is the car's **eyes**, not its brain. Python runs the simulation (car physics,
obstacles, collisions, the fly connectome) and this project draws it in 3D, sending
back the image from the camera on the car's bonnet. The car then drives on those
pixels alone.

```
Python  --{car pose, obstacles, HUD text}-->  Godot     JSON, 4-byte length prefix
Python  <--{camera image, raw RGB}----------  Godot     4-byte length prefix
```

One round trip per simulation step, so Godot waits for Python and Python waits for
Godot. If the fly brain is thinking, the picture simply advances more slowly.

## Running it

Install Godot 4 (Fedora: `sudo dnf install godot`, or Flatpak, or godotengine.org),
then from the FlyDrive folder:

```bash
python run.py --eyes camera --renderer godot --reflex rl      # starts Godot for you
python run.py --eyes camera --renderer godot --godot /path/to/godot
```

Or start Godot yourself: open this folder in the editor, press **Play**, and run
Python with `--no-launch` (the world waits and retries until Python appears):

```bash
python run.py --eyes camera --renderer godot --no-launch --reflex rl
```

The window shows a chase camera, the car's own camera view in the corner, and a line
of text from Python (reflex, speed, obstacles, collisions, escape urge).

No Godot? Everything works without it: `python run.py --eyes camera` uses FlyDrive's
own small 3D renderer, and the car's vision code is exactly the same.

## What's here

| File | What it does |
|---|---|
| `main.gd` | the whole world, built in code: sky, road, obstacles, car, cameras, HUD, and the TCP link |
| `main.tscn` | an empty scene that just loads `main.gd` |
| `project.godot` | project settings (OpenGL compatibility renderer, 1280x720) |

The road is built from the track points Python sends, so both renderers draw the same
road, and the colours come from `flydrive/camera.py`. Change them there and both
follow.

## Notes

- The car's camera is on visual layer 1 and the car's body on layer 2, so the camera
  doesn't see the car it's mounted on.
- Godot renders with lighting and shadows, so the image doesn't match FlyDrive's own
  renderer pixel for pixel. The vision system calibrates itself on the first frame and
  keeps adapting, which is the point of testing here: the same eyes on a different
  picture.
- If the image ever comes back upside down (renderer differences), Python detects it
  from where the sky is and flips it.
- **Tested how far:** the GDScript is syntax-checked and the protocol was tested against
  a stand-in client, but this project has not been run inside Godot itself, because Godot
  could not be installed in the machine that wrote it. If something misbehaves on first
  run, the likely suspects are the viewport read-back and the sky/lighting colours.
