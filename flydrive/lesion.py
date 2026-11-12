"""Choosing which neurons to silence.

A lesion spec is a comma-separated list of:
  * a group name from the brain ('loom_left', 'gf_right', 'esc_left', ...),
  * a cell type, optionally restricted to a side ('LC4', 'LPLC2@left', 'DNp01'),
  * 'ids:12;345;6789': explicit neuron indices (separated by semicolons),
  * 'random:N' or 'random:N:SEED': N random neurons, never from the looming / escape groups,
    so a control lesion can't accidentally hit the circuit under test.
"""
import numpy as np

from . import connectome

_CIRCUIT = ("loom_left", "loom_right", "gf_left", "gf_right", "esc_left", "esc_right")


def resolve(spec, groups, n):
    """Return the sorted neuron indices named by `spec` (empty spec: no neurons)."""
    out = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if tok.startswith("ids:"):
            idx = np.array([int(x) for x in tok[4:].split(";") if x], dtype=np.int32)
        elif tok.startswith("random:"):
            parts = tok.split(":")
            count, seed = int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
            banned = np.concatenate([groups[g] for g in _CIRCUIT])
            pool = np.setdiff1d(np.arange(n, dtype=np.int32), banned)
            idx = np.random.default_rng(seed).choice(pool, size=count, replace=False)
        elif tok in groups:
            idx = groups[tok]
        else:
            name, _, side = tok.partition("@")
            idx = connectome.neurons_of_type([name], side or None)
        if len(idx) == 0:
            raise ValueError(f"lesion '{tok}' matches no neurons")
        out.append(np.asarray(idx, dtype=np.int32))
    return np.unique(np.concatenate(out)) if out else np.empty(0, dtype=np.int32)
