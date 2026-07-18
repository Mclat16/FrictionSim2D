"""D1 hetero sheet-on-sheet slide assembly (consumes hetero.py's structure builder)."""
from typing import List, Dict
from ase.io import read

PAD = 1.0  # Å pad above/below each layer's atom z-range


def compute_layer_zbands(data_path, layers) -> List[Dict]:
    """Compute per-layer z-bands from each layer's KNOWN placement.

    Each layer in ``layers`` (a :class:`~src.builders.hetero.HeteroStackLayer`)
    records ``.source`` (the per-material supercell data file it was read
    from) and ``.z`` (the z-shift applied via
    ``read_data ... shift 0 0 z``, see templates/hetero/stack_hetero.lmp).
    So the placed atoms of a layer occupy exactly
    ``[native_zmin + z, native_zmax + z]``, where ``native_z`` comes from
    reading ``.source``. This is exact and independent of gap heuristics, so
    (unlike clustering on the (n-1) largest z-gaps) it cannot be fooled by a
    puckered layer whose internal sub-plane gap exceeds an inter-layer gap
    (e.g. black phosphorus, GeS).

    Multiple layers of the SAME material share one ``.source`` file but have
    DIFFERENT ``.z``, so their bands are still distinct.

    Returns one ``{'idx', 'zlo', 'zhi'}`` per layer (idx 1..n bottom->top).

    Raises:
        ValueError: If ``data_path``'s actual atom z-coordinates disagree with
            the bands derived from known placement (i.e. some atom falls in
            zero or more than one band). This is a self-check that makes a
            silent mis-grouping impossible -- it fails loud instead.
    """
    native_zrange: Dict[str, tuple] = {}
    placed = []
    for layer in layers:
        key = str(layer.source)
        if key not in native_zrange:
            zz = read(key, format="lammps-data").get_positions()[:, 2]
            native_zrange[key] = (float(zz.min()), float(zz.max()))
        zmin, zmax = native_zrange[key]
        zlo, zhi = zmin + layer.z - PAD, zmax + layer.z + PAD
        center = 0.5 * (zmin + zmax) + layer.z
        placed.append((center, zlo, zhi))

    placed.sort(key=lambda t: t[0])
    bands = [{"idx": i + 1, "zlo": zlo, "zhi": zhi} for i, (_, zlo, zhi) in enumerate(placed)]

    # Self-check (fail loud): every atom in the actual assembled data file
    # must fall in exactly one band. Catches any placement/write discrepancy
    # instead of silently mis-grouping.
    z = read(str(data_path), format="lammps-data").get_positions()[:, 2]
    for zi in z:
        hits = [b for b in bands if b["zlo"] <= zi <= b["zhi"]]
        if len(hits) != 1:
            raise ValueError(
                f"compute_layer_zbands: atom at z={zi:.4f} falls in {len(hits)} "
                f"bands (expected exactly 1); bands={bands}. The assembled "
                f"{data_path} disagrees with the known layer placement "
                f"(source native z-range + z-shift) -- check read_data shift "
                f"semantics or PAD={PAD}."
            )
    return bands
