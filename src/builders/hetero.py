"""Build matched per-material supercells for heterostructure stacking.

Given two monolayer LAMMPS data files and a Zur-McGill coincidence-lattice
:class:`~src.core.lattice_matching.MatchResult`, :func:`build_matched_supercells`
builds each material's HNF coincidence supercell and strains the
smaller-area supercell onto the larger (left unstrained), so both output
supercells end up sharing exactly ONE in-plane periodic box and can be
stacked into a single heterostructure cell.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
from ase.build import make_supercell
from ase.io import read, write

from ..core.lattice_apply import canonicalize_cell_to_lammps, strain_monolayer_to_cell
from ..core.lattice_matching import MatchResult

PathLike = Union[str, Path]

# Absolute strain margin (added on top of the match's own strain_a +
# strain_b) that absorbs floating-point / build-pipeline noise between the
# pure lattice-parameter estimate (from CIF cell parameters) and the real
# geometric strain measured from built LAMMPS data files (CIF read -> write
# -> orthogonalize -> duplicate all round trip through finite precision).
_STRAIN_SAFETY_MARGIN = 1e-3

# Tolerance used when verifying that ASE's own LAMMPS-data write (which
# internally canonicalizes the box to LAMMPS form) reproduces the target
# cell computed independently by canonicalize_cell_to_lammps.
_CANON_ATOL = 1e-6


@dataclass
class MatchedStack:
    """Two per-material coincidence supercells sharing one in-plane box.

    Attributes:
        supercell_a: Path to material A's output supercell data file.
        supercell_b: Path to material B's output supercell data file.
        reference: Which input material ('a' or 'b') supplied the
            unstrained reference box -- the one with the larger in-plane
            coincidence-supercell area.
        strain_reference: Always 0.0; the reference material is never
            strained, only rotated/canonicalized into LAMMPS form.
        strain_applied: Max absolute principal strain applied to the
            non-reference material to reach the shared (reference) box.
    """
    supercell_a: Path
    supercell_b: Path
    reference: str
    strain_reference: float
    strain_applied: float


def _embed_3x3(matrix_2x2: np.ndarray) -> np.ndarray:
    """Embed a 2x2 integer in-plane supercell matrix into a 3x3 matrix.

    ``ase.build.make_supercell`` requires a full 3x3 integer matrix; the
    z-row/column is left as identity so the out-of-plane lattice vector
    (and periodicity) is left untouched.
    """
    p = np.eye(3, dtype=int)
    p[:2, :2] = np.asarray(matrix_2x2, dtype=int)
    return p


def _in_plane_area(cell_2x2: np.ndarray) -> float:
    """Absolute area (Å²) of a 2x2 row-vector in-plane cell."""
    return abs(float(np.linalg.det(np.asarray(cell_2x2, dtype=float))))


def _build_raw_supercell(mono_path: PathLike, matrix_2x2: np.ndarray):
    """Tile a monolayer LAMMPS data file into a coincidence supercell (in memory)."""
    atoms = read(str(mono_path), format="lammps-data")
    return make_supercell(atoms, _embed_3x3(matrix_2x2))


def _default_strain_tolerance(match: MatchResult) -> float:
    """Strain budget for concentrating both materials' mismatch onto one.

    ``find_coincidence_lattice`` computes ``strain_a``/``strain_b`` against
    a common *mean* cell -- i.e. the lattice mismatch is conceptually split
    between both materials. Here the reference material is left completely
    unstrained, so the non-reference material must absorb roughly the
    *combined* mismatch (~``strain_a + strain_b``), plus a small absolute
    margin for build-pipeline floating point noise.
    """
    return match.strain_a + match.strain_b + _STRAIN_SAFETY_MARGIN


def build_matched_supercells(
    mono_a_path: PathLike,
    mono_b_path: PathLike,
    match: MatchResult,
    workdir: PathLike,
) -> MatchedStack:
    """Build A's and B's coincidence supercells sharing one in-plane box.

    Builds each material's HNF coincidence supercell (``match.matrix_a`` /
    ``match.matrix_b`` applied to the respective monolayer's own cell via
    :func:`ase.build.make_supercell`), picks whichever has the larger
    in-plane area as an unstrained reference, canonicalizes its box to
    LAMMPS form (a1 along +x), and strains the other material's supercell
    onto that exact box via :func:`~src.core.lattice_apply.strain_monolayer_to_cell`.

    Args:
        mono_a_path: LAMMPS data file for material A's monolayer.
        mono_b_path: LAMMPS data file for material B's monolayer.
        match: Coincidence-lattice match from
            :func:`src.core.lattice_matching.find_coincidence_lattice` (or
            ``find_coincidence_lattice_from_cif``) describing the integer
            HNF supercell matrices for A and B.
        workdir: Directory to write the two output supercell data files
            into (created if missing).

    Returns:
        :class:`MatchedStack` with both output supercell paths (ordered A,
        then B, regardless of which was the reference), which material
        ('a' or 'b') was left unstrained, and the strain applied to the
        other.

    Raises:
        ValueError: If straining the smaller-area supercell onto the
            larger's box requires more strain than the match's own
            predicted mismatch (plus a small numerical margin) allows --
            i.e. the two monolayers are not actually commensurate at this
            match, so building the pair would silently produce an
            incommensurate "matched" stack.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    materials: Dict[str, Dict[str, object]] = {
        "a": {"mono_path": Path(mono_a_path), "matrix": match.matrix_a},
        "b": {"mono_path": Path(mono_b_path), "matrix": match.matrix_b},
    }

    raw = {
        label: _build_raw_supercell(info["mono_path"], info["matrix"])
        for label, info in materials.items()
    }
    cells = {label: np.array(atoms.cell)[:2, :2] for label, atoms in raw.items()}
    areas = {label: _in_plane_area(cell) for label, cell in cells.items()}

    ref_label = "a" if areas["a"] >= areas["b"] else "b"
    other_label = "b" if ref_label == "a" else "a"

    # Reference: canonicalize to LAMMPS form (a1 along +x) and write as-is
    # (unstrained). ASE's lammps-data writer already internally rotates the
    # cell + atoms into this same canonical form on write, so writing the
    # raw supercell directly reproduces ref_canon -- verified below rather
    # than assumed.
    ref_canon, _rot_deg = canonicalize_cell_to_lammps(cells[ref_label])

    ref_out = workdir / f"{ref_label}_supercell.lmp"
    write(str(ref_out), raw[ref_label], format="lammps-data")

    written_ref_cell = np.array(read(str(ref_out), format="lammps-data").cell)[:2, :2]
    if not np.allclose(written_ref_cell, ref_canon, atol=_CANON_ATOL):
        raise RuntimeError(
            f"Internal error building heterostructure reference: LAMMPS-canonical "
            f"write of material '{ref_label}' supercell ({ref_out}) does not match "
            f"the independently computed canonical cell "
            f"{ref_canon.tolist()} (got {written_ref_cell.tolist()})."
        )

    # Non-reference: write its own coincidence supercell, then strain it
    # onto the reference's exact canonical box.
    other_raw_out = workdir / f"{other_label}_supercell_raw.lmp"
    write(str(other_raw_out), raw[other_label], format="lammps-data")

    other_out = workdir / f"{other_label}_supercell.lmp"
    strain = strain_monolayer_to_cell(str(other_raw_out), str(other_out), ref_canon)

    tol = _default_strain_tolerance(match)
    if strain > tol:
        raise ValueError(
            f"Cannot build a commensurate heterostructure stack: straining "
            f"material '{other_label}' ({materials[other_label]['mono_path'].name}) "
            f"onto material '{ref_label}' "
            f"({materials[ref_label]['mono_path'].name})'s reference box requires "
            f"{strain:.4%} strain, exceeding the tolerance of {tol:.4%} predicted "
            f"from the match (strain_a={match.strain_a:.4%}, "
            f"strain_b={match.strain_b:.4%})."
        )

    outputs = {ref_label: ref_out, other_label: other_out}

    return MatchedStack(
        supercell_a=outputs["a"],
        supercell_b=outputs["b"],
        reference=ref_label,
        strain_reference=0.0,
        strain_applied=strain,
    )
