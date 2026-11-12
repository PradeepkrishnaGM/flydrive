"""Whole-brain leaky integrate-and-fire (LIF) simulation of the fly connectome.

Neuron and synapse parameters follow Shiu et al. 2024 (Nature), which used the
same model to predict feeding and grooming behaviour.  Instead of Brian2 this
uses plain NumPy and only propagates spikes that actually happened, so all
~139,000 neurons run fine on a laptop CPU.
"""
from collections import deque

import numpy as np

from . import connectome


class FlyBrain:
    # --- Shiu et al. 2024 parameters (mV, ms) ---
    V_REST = -52.0
    V_RESET = -52.0
    V_TH = -45.0
    T_MBR = 20.0     # membrane time constant
    TAU_SYN = 5.0    # synaptic conductance decay
    T_REF = 2.2      # refractory period
    T_DELAY = 1.8    # synaptic delay
    W_SYN = 0.275    # mV per synapse

    def __init__(self, dt=0.5, seed=0):
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        W, self.groups = connectome.load()
        self.n = W.shape[0]
        self.indptr = W.indptr
        self.indices = W.indices
        self.weights = (W.data * self.W_SYN).astype(np.float32)

        self.v = np.full(self.n, self.V_REST, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.ref_left = np.zeros(self.n, dtype=np.float32)

        delay_steps = max(1, int(round(self.T_DELAY / dt)))
        self.pending = deque([None] * delay_steps)  # queued synaptic input

        self.decay_g = np.float32(np.exp(-dt / self.TAU_SYN))
        self._k_mbr = np.float32(dt / self.T_MBR)
        self._buf = np.zeros(self.n, dtype=np.float32)
        self._refractory = np.empty(0, dtype=np.int32)
        self._n_refractory = 0
        # Poisson drive: neuron index -> firing rate (Hz)
        self.drive_idx = np.empty(0, dtype=np.int32)
        self.drive_rate = np.empty(0, dtype=np.float32)
        self.total_spikes = 0
        self.lesioned = np.empty(0, dtype=np.int32)   # silenced neurons (see lesion())

    def lesion(self, idx):
        """Silence these neurons: they can never spike, so they send nothing downstream.

        Unlike reset(), a lesion stays in place until unlesion() is called."""
        self.lesioned = np.unique(np.asarray(idx, dtype=np.int32))

    def unlesion(self):
        self.lesioned = np.empty(0, dtype=np.int32)

    def set_drive(self, rates_by_group):
        """rates_by_group: {group_name: rate_Hz}. Stimulated neurons fire as Poisson."""
        idx, rate = [], []
        for name, r in rates_by_group.items():
            if r <= 0:
                continue
            members = self.groups[name]
            idx.append(members)
            rate.append(np.full(len(members), r, dtype=np.float32))
        if idx:
            self.drive_idx = np.concatenate(idx)
            self.drive_rate = np.concatenate(rate)
        else:
            self.drive_idx = np.empty(0, dtype=np.int32)
            self.drive_rate = np.empty(0, dtype=np.float32)

    def _propagate(self, spiked):
        if spiked.size == 0:
            return None
        starts, ends = self.indptr[spiked], self.indptr[spiked + 1]
        lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            return None
        # gather all outgoing synapses of spiking neurons without Python loops
        offsets = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(total)
        return self.indices[offsets], self.weights[offsets]

    def step(self):
        """Advance one dt. Returns array of neuron indices that spiked."""
        dt = self.dt
        v, g, buf = self.v, self.g, self._buf
        arriving = self.pending.popleft()
        if arriving is not None:
            np.add.at(g, arriving[0], arriving[1])

        # dv = (V_rest - v + g) * dt / tau_m, frozen while refractory
        np.subtract(g, v, out=buf)
        buf += self.V_REST
        buf *= self._k_mbr
        if self._n_refractory:
            self.ref_left[self._refractory] -= dt
            still = self.ref_left[self._refractory] > 0
            buf[self._refractory] = 0.0
            self._refractory = self._refractory[still]
            self._n_refractory = self._refractory.size
        v += buf
        g *= self.decay_g

        # Poisson sensory drive forces spikes in stimulated neurons
        if self.drive_idx.size:
            fire = self.rng.random(self.drive_idx.size) < self.drive_rate * dt / 1000.0
            v[self.drive_idx[fire]] = self.V_TH + 1.0

        if self.lesioned.size:
            v[self.lesioned] = self.V_REST          # silenced, even if driven

        spiked = np.flatnonzero(v > self.V_TH).astype(np.int32)
        if spiked.size:
            v[spiked] = self.V_RESET
            g[spiked] = 0.0
            self.ref_left[spiked] = self.T_REF
            self._refractory = np.union1d(self._refractory, spiked).astype(np.int32)
            self._n_refractory = self._refractory.size
            self.total_spikes += spiked.size
        self.pending.append(self._propagate(spiked))
        return spiked

    def reset(self):
        """Silence the brain (all neurons back to rest) without reloading anything."""
        self.v[:] = self.V_REST
        self.g[:] = 0.0
        self.ref_left[:] = 0.0
        self._refractory = np.empty(0, dtype=np.int32)
        self._n_refractory = 0
        self.pending = deque([None] * len(self.pending))
        self.set_drive({})

    def run_counts(self, ms, idx):
        """Simulate `ms` milliseconds; return spike counts for every neuron in `idx`."""
        if getattr(self, "_pos_for", None) is not idx:
            self._pos = np.full(self.n, -1, dtype=np.int32)
            self._pos[idx] = np.arange(len(idx), dtype=np.int32)
            self._pos_for = idx
        counts = np.zeros(len(idx) + 1, dtype=np.float32)
        for _ in range(int(round(ms / self.dt))):
            spiked = self.step()
            if spiked.size:
                counts += np.bincount(self._pos[spiked] + 1, minlength=len(idx) + 1)
        return counts[1:]

    def run(self, ms, watch):
        """Simulate `ms` milliseconds; return spike counts for each watched group."""
        label = np.full(self.n, -1, dtype=np.int16)
        for j, k in enumerate(watch):
            label[self.groups[k]] = j
        tally = np.zeros(len(watch) + 1, dtype=np.int64)
        for _ in range(int(round(ms / self.dt))):
            spiked = self.step()
            if spiked.size:
                tally += np.bincount(label[spiked] + 1, minlength=len(watch) + 1)
        return {k: int(tally[j + 1]) for j, k in enumerate(watch)}
