"""Apply lattice-matching results to LAMMPS monolayer data files.

Bridges the pure-math ``lattice_matching`` module to on-disk structures. Kept
separate so ``lattice_matching`` stays dependency-free (math only).
"""
from __future__ import annotations

import numpy as np

from .lattice_matching import _max_strain

_TILT_EPS = 1e-9


def _create_lammps_instance():
    """Start a headless LAMMPS instance.

    Mirrors ``src/builders/components.py::_create_lammps_instance`` (same
    import style and cmdargs) so geometry ops are driven from Python the same
    way everywhere in the codebase.
    """
    from lammps import lammps  # pylint: disable=import-outside-toplevel

    return lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])


def _detect_atom_style(path: str) -> str:
    """Detect the LAMMPS ``atom_style`` from a data file's ``Atoms`` header.

    LAMMPS data files annotate the Atoms section with the style used to write
    it, e.g. ``Atoms # charge``. Falls back to ``atomic`` if the comment is
    absent (older/hand-written files).
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                if "#" in stripped:
                    style = stripped.split("#", 1)[1].strip()
                    return style or "atomic"
                return "atomic"
    return "atomic"


def _has_masses_section(path: str) -> bool:
    """Return True if the data file already defines a ``Masses`` section."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip() == "Masses":
                return True
    return False


def strain_monolayer_to_cell(input_data_path: str,
                             output_data_path: str,
                             target_cell_2x2: np.ndarray) -> float:
    """Affinely strain a monolayer so its in-plane cell becomes ``target_cell_2x2``.

    Drives native LAMMPS ``change_box ... remap`` to apply the strain, so atom
    motion is a pure affine transform (fractional coordinates preserved)
    while every other data-file field -- atom_style, per-atom charges, type
    IDs, masses, velocities, etc. -- round-trips through LAMMPS's own
    read_data/write_data untouched. This avoids the metadata loss of a bare
    ASE read/write round-trip (which silently drops per-atom charges on
    ``atom_style=charge`` files and can mangle multi-element type fidelity).

    The out-of-plane (z) vector and any xz/yz tilt are left unchanged. Only
    the in-plane box (x, y extents and the xy tilt factor) and atom x/y
    coordinates are modified.

    Args:
        input_data_path: Path to the source LAMMPS data file.
        output_data_path: Path to write the strained LAMMPS data file.
        target_cell_2x2: 2x2 target in-plane cell (row vectors, Å), in
            LAMMPS-canonical form (first vector along x, i.e. ``target[0][1]``
            must be ~0).

    Returns:
        Max absolute principal strain applied (deforming the source in-plane
        cell to ``target_cell_2x2``).
    """
    target = np.asarray(target_cell_2x2, dtype=float)
    assert abs(target[0][1]) < _TILT_EPS, \
        "target_cell_2x2 must have its first vector along x (LAMMPS-canonical form)"

    atom_style = _detect_atom_style(input_data_path)
    needs_mass_fallback = not _has_masses_section(input_data_path)

    lmp = _create_lammps_instance()
    try:
        lmp.command("clear")
        lmp.command("units metal")
        lmp.command(f"atom_style {atom_style}")
        lmp.command(f'read_data "{input_data_path}"')

        if needs_mass_fallback:
            # write_data refuses to run unless every type has a mass. Files
            # without a Masses section (e.g. bare test fixtures) never had
            # real per-type masses to preserve, so a placeholder is safe --
            # it adds nothing that was meaningfully present before.
            lmp.command("mass * 1.0")

        boxlo, boxhi, xy, _yz, _xz, _periodicity, _box_change = lmp.extract_box()
        xlo, ylo, _zlo = boxlo
        xhi, yhi, _zhi = boxhi
        source_2x2 = np.array([[xhi - xlo, 0.0], [xy, yhi - ylo]])

        lx = float(target[0][0])
        ly = float(target[1][1])
        xy_final = float(target[1][0])
        needs_xy_clause = abs(xy) > _TILT_EPS or abs(xy_final) > _TILT_EPS

        cmd = "change_box all "
        if needs_xy_clause:
            # Converting box style to triclinic is safe even if it already is
            # (LAMMPS treats a repeated `triclinic` keyword as a no-op).
            cmd += "triclinic "
        cmd += f"x final {xlo} {xlo + lx} y final {ylo} {ylo + ly} "
        if needs_xy_clause:
            cmd += f"xy final {xy_final} "
        cmd += "remap units box"
        lmp.command(cmd)

        lmp.command(f'write_data "{output_data_path}"')
    finally:
        lmp.close()

    return _max_strain(target, source_2x2)
