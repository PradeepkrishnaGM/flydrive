#!/usr/bin/env bash
# One-time setup for FlyDrive on Fedora.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing system packages (needs sudo)"
sudo dnf install -y python3 python3-pip python3-devel gcc unzip

echo "==> Creating virtual environment in .venv"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Downloading FlyWire connectome data (~135 MB)"
python download_data.py

echo "==> Building connectome cache and checking the brain"
python run.py --probe

echo
echo "All set! Start the simulator with:"
echo "  source .venv/bin/activate"
echo "  python run.py                          # hand-tuned fly brain"
echo "  python run.py --reflex rl              # fly brain trained by reinforcement learning"
echo "  python run.py --eyes camera --reflex rl   # driving on camera pixels only (press V)"
echo "  python run.py --difficulty hard        # the crowded world"
echo "  python run.py --benchmark 120 --seeds 1,2,3,4,5   # the comparison table"
echo "  (in the window: 1-8 switch reflex, V camera view, O add an obstacle, R reset)"
echo
echo "Optional 3D world:  sudo dnf install godot"
echo "  python run.py --eyes camera --renderer godot --reflex rl"
echo "Retrain any time with:  python train_rl.py   (or python train.py for the teacher version)"
