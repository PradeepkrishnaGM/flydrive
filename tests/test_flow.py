"""Optic-flow expansion estimator: invariants on synthetic images (fast, no brain needed)."""
import numpy as np
import pytest

from flydrive.vision import expansion_rate

N = 96
YY, XX = np.mgrid[0:N, 0:N].astype(np.float32)
CX = CY = N / 2
MASK = (XX - CX) ** 2 + (YY - CY) ** 2 < 30 ** 2
DT = 0.02


def blob(scale=1.0, shift=0.0):
    """A smooth textured object whose size is `scale` times its original size."""
    x, y = (XX - CX - shift) / scale, (YY - CY) / scale
    return np.exp(-(x ** 2 + y ** 2) / (2 * 12 ** 2)) * (1 + 0.3 * np.sin(x / 4) * np.cos(y / 5))


@pytest.mark.parametrize("e_true", [0.8, 1.5, 3.0])
def test_expansion_matches_one_over_ttc(e_true):
    e = expansion_rate(blob(1.0), blob(1.0 + e_true * DT), MASK, DT)
    assert e == pytest.approx(e_true, rel=0.15)


def test_static_scene_gives_zero():
    assert expansion_rate(blob(), blob(), MASK, DT) == pytest.approx(0.0, abs=1e-6)


def test_sideways_motion_is_not_expansion():
    e = expansion_rate(blob(), blob(shift=0.5), MASK, DT)
    assert abs(e) < 0.15                      # far below the 0.4 /s looming threshold


def test_receding_object_is_negative():
    assert expansion_rate(blob(1.0), blob(1.0 - 1.0 * DT), MASK, DT) < -0.5


def test_deterministic():
    a = expansion_rate(blob(1.0), blob(1.02), MASK, DT)
    b = expansion_rate(blob(1.0), blob(1.02), MASK, DT)
    assert a == b


def test_textureless_region_returns_none():
    flat = np.full((N, N), 0.5, np.float32)
    assert expansion_rate(flat, flat, MASK, DT) is None
