"""Apply lattice-matching results to LAMMPS monolayer data files.

Bridges the pure-math ``lattice_matching`` module to on-disk structures. Kept
separate so ``lattice_matching`` stays dependency-free (math only).
"""
from __future__ import annotations

import numpy as np
from ase.io import read, write

from .lattice_matching import _max_strain


def strain_monolayer_to_cell(input_data_path: str,
                             output_data_path: str,
                             target_cell_2x2: np.ndarray) -> float:
    """Affinely strain a monolayer so its in-plane cell becomes ``target_cell_2x2``.

    The out-of-plane (z) vector is left unchanged. Atom positions are carried
    with the box (affine remap). Returns the max absolute principal strain applied.
    """
    atoms = read(input_data_path, format="lammps-data")
    cell = np.array(atoms.cell)
    source_2x2 = cell[:2, :2].copy()

    new_cell = cell.copy()
    new_cell[0, :2] = target_cell_2x2[0]
    new_cell[1, :2] = target_cell_2x2[1]
    atoms.set_cell(new_cell, scale_atoms=True)   # affine remap of positions

    write(output_data_path, atoms, format="lammps-data")
    return _max_strain(np.asarray(target_cell_2x2, dtype=float), source_2x2)
