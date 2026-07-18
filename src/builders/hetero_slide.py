"""D1 hetero sheet-on-sheet slide assembly (consumes hetero.py's structure builder)."""
from pathlib import Path
from typing import List, Dict
import numpy as np
from ase.io import read

PAD = 1.0  # Å pad above/below each layer's atom z-range


def compute_layer_zbands(data_path, layers) -> List[Dict]:
    """Cluster hetero.data atom z into len(layers) bands, ordered bottom->top.

    Hetero layers share atom types within a material, so slide layer groups
    cannot be type-based; instead each physical layer is isolated by a z-slab
    region. Returns one {'idx', 'zlo', 'zhi'} per layer (idx 1..n bottom->top),
    padded so the region brackets the layer's atoms without overlapping the next.
    """
    n = len(layers)
    z = np.sort(read(str(data_path), format="lammps-data").get_positions()[:, 2])
    # split into n clusters at the (n-1) largest gaps
    gaps = np.diff(z)
    cut = np.sort(np.argsort(gaps)[-(n - 1):]) if n > 1 else np.array([], dtype=int)
    bands, start = [], 0
    for i, c in enumerate(list(cut) + [len(z) - 1]):
        seg = z[start:c + 1]
        bands.append({"idx": i + 1, "zlo": float(seg.min() - PAD), "zhi": float(seg.max() + PAD)})
        start = c + 1
    return bands
