"""Load the FlyWire v783 connectome and find the neurons we care about.

Data sources (downloaded by download_data.py):
  * Connectivity_783.parquet / Completeness_783.csv
      from Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model)
  * Supplemental_file1_neuron_annotations.tsv
      from Schlegel et al. 2024 (github.com/flyconnectome/flywire_annotations)

The first load builds a compact cache (data/connectome_cache.npz) so later
runs start in a couple of seconds.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE = DATA_DIR / "connectome_cache.npz"
CACHE_VERSION = 2  # bump when the cache contents change

# Visual looming detectors (optic lobe -> central brain)
LOOM_TYPES = ["LPLC2", "LC4"]
# Descending neurons involved in escape / avoidance
GIANT_FIBER = ["DNp01"]
ESCAPE_DNS = ["DNp02", "DNp04", "DNp06", "DNp11"]


def _ids_for(annot, types, side=None):
    mask = annot["cell_type"].isin(types) | annot["hemibrain_type"].isin(types)
    if side is not None:
        mask &= annot["side"] == side
    return annot.loc[mask, "root_id"].astype(np.int64).to_numpy()


def build_cache():
    conn_path = DATA_DIR / "Connectivity_783.parquet"
    comp_path = DATA_DIR / "Completeness_783.csv"
    annot_path = DATA_DIR / "Supplemental_file1_neuron_annotations.tsv"
    for p in (conn_path, comp_path, annot_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run:  python download_data.py")

    print("Building connectome cache (first run only, ~1 minute)...")
    comp = pd.read_csv(comp_path, index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    n = len(root_ids)

    conn = pd.read_parquet(
        conn_path,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = conn["Presynaptic_Index"].to_numpy(np.int32)
    post = conn["Postsynaptic_Index"].to_numpy(np.int32)
    w = conn["Excitatory x Connectivity"].to_numpy(np.float32)
    # CSC: column j lists all postsynaptic targets of neuron j
    W = sp.csc_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()

    annot = pd.read_csv(annot_path, sep="\t", low_memory=False,
                        usecols=["root_id", "super_class", "cell_type", "hemibrain_type", "side"])
    index_of = {rid: i for i, rid in enumerate(root_ids)}

    def idx(types, side=None):
        return np.array([index_of[r] for r in _ids_for(annot, types, side) if r in index_of],
                        dtype=np.int32)

    groups = {
        "loom_left": idx(LOOM_TYPES, "left"),
        "loom_right": idx(LOOM_TYPES, "right"),
        "gf_left": idx(GIANT_FIBER, "left"),
        "gf_right": idx(GIANT_FIBER, "right"),
        "esc_left": idx(ESCAPE_DNS, "left"),
        "esc_right": idx(ESCAPE_DNS, "right"),
    }
    # Every descending neuron: the brain's output cables to the body.
    # These are the features the trainable readout learns from.
    dn = annot[annot["super_class"] == "descending"]
    dn = dn[dn["root_id"].isin(index_of)]
    dn_idx = np.array([index_of[r] for r in dn["root_id"]], dtype=np.int32)
    dn_names = (dn["cell_type"].fillna("DN?").astype(str) + "_"
                + dn["side"].fillna("?").astype(str)).to_numpy()

    np.savez_compressed(
        CACHE, version=CACHE_VERSION, n=n, indptr=W.indptr, indices=W.indices, data=W.data,
        dn_idx=dn_idx, dn_names=dn_names.astype("U40"),
        **{f"g_{k}": v for k, v in groups.items()},
    )
    print("Cache written to", CACHE)


def load():
    """Return (W as CSC matrix, dict of neuron index groups)."""
    if not CACHE.exists() or _cache_version() != CACHE_VERSION:
        build_cache()
    z = np.load(CACHE)
    n = int(z["n"])
    W = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    groups = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    groups["dn_all"] = z["dn_idx"]
    return W, groups


def _cache_version():
    z = np.load(CACHE)
    return int(z["version"]) if "version" in z.files else 1


def dn_names():
    """Readable names ('DNp01_left', ...) for the neurons in groups['dn_all']."""
    return np.load(CACHE)["dn_names"]
