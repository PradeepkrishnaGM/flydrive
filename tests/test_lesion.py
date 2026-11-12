"""Lesions: silenced neurons never spike, and unlesion restores the exact baseline."""
import numpy as np
import pytest

from flydrive import connectome
from flydrive.lesion import resolve

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not connectome.CACHE.exists(), reason="needs data/ (python download_data.py)"),
]


@pytest.fixture(scope="module")
def make_brain():
    from flydrive.brain import FlyBrain
    return lambda: FlyBrain(dt=0.5, seed=0)


def spikes(brain, stim, ms=150):
    brain.set_drive(stim)
    watch = ["gf_left", "gf_right", "esc_left", "esc_right", "loom_left"]
    return brain.run(ms, watch)


def test_lesioned_neurons_never_spike(make_brain):
    b = make_brain()
    b.lesion(b.groups["loom_left"])
    assert spikes(b, {"loom_left": 100.0})["loom_left"] == 0      # even though driven


def test_silenced_input_silences_downstream(make_brain):
    b = make_brain()
    base = spikes(b, {"loom_left": 100.0})
    assert base["esc_left"] + base["esc_right"] > 0
    b.reset()
    b.lesion(b.groups["loom_left"])
    c = spikes(b, {"loom_left": 100.0})
    assert c["esc_left"] + c["esc_right"] < base["esc_left"] + base["esc_right"]


def test_unlesion_restores_identical_behaviour(make_brain):
    a, b = make_brain(), make_brain()
    base = spikes(a, {"loom_left": 60.0})
    b.lesion(b.groups["gf_left"])
    b.unlesion()
    assert spikes(b, {"loom_left": 60.0}) == base


def test_empty_lesion_is_baseline(make_brain):
    a, b = make_brain(), make_brain()
    b.lesion(np.empty(0, dtype=np.int32))
    assert spikes(a, {"loom_right": 60.0}) == spikes(b, {"loom_right": 60.0})


def test_reset_keeps_the_lesion(make_brain):
    b = make_brain()
    b.lesion(b.groups["gf_left"])
    b.reset()
    assert len(b.lesioned) == len(b.groups["gf_left"])


def test_resolve_groups_types_and_random(make_brain):
    b = make_brain()
    assert np.array_equal(resolve("loom_left", b.groups, b.n), np.sort(b.groups["loom_left"]))
    by_type = resolve("LPLC2@left,LC4@left", b.groups, b.n)
    assert np.array_equal(by_type, np.sort(b.groups["loom_left"]))
    r1, r2 = resolve("random:10:3", b.groups, b.n), resolve("random:10:3", b.groups, b.n)
    assert len(r1) == 10 and np.array_equal(r1, r2)
    circuit = np.concatenate([b.groups[g] for g in ("loom_left", "loom_right", "gf_left",
                                                    "gf_right", "esc_left", "esc_right")])
    assert not np.intersect1d(r1, circuit).size
    with pytest.raises(ValueError):
        resolve("NoSuchCellType", b.groups, b.n)
