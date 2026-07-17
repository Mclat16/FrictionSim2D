"""Tests for src.builders.hetero.build_matched_supercells.

Covers:

* A real MoS2/WS2 pair (built via the real monolayer builder), whose
  orthogonalized cells match at IDENTITY -- verifies the two per-material
  supercells share exactly one in-plane box, the larger-area material is left
  unstrained, and the smaller is strained onto it within tolerance.
* A genuinely SHEARED, non-identity coincidence (both match matrices have a
  non-zero off-diagonal), fed as synthetic monolayer data files -- verifies the
  two BUILT supercell boxes coincide within a tight tolerance at the true (low)
  strain. This is the case the identity-only MoS2/WS2 test cannot exercise:
  applying a match to the wrong cell, or straining in an independently
  re-canonicalized frame, would produce non-coinciding boxes / spurious strain.
* Fail-loud behavior when no low-strain coincidence exists.
"""
import re

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from src.builders import components
from src.builders.hetero import build_matched_supercells, register_hetero_potentials
from src.core.config import SheetConfig, load_settings
from src.core.lattice_matching import find_coincidence_lattice

MOS2_CIF = "examples/materials/h-MoS2.cif"
WS2_CIF = "examples/materials/h-WS2.cif"
MOS2_POT = "examples/potentials/sw/MoS2_wen.sw"
# ws2_jrmidas.sw defines a third element ('C', for graphene coupling) that is
# not present in h-WS2.cif and is rejected by get_potential_element_order, so
# a plain WS2-only SW potential is used instead. Both this file and
# MoS2_wen.sw assign exactly one LAMMPS type per element (no sublattice
# splitting), so both monolayers' orthogonalized unit cells are a plain 1x2
# doubling of their CIF primitive cell -- i.e. directly comparable to each
# other and matched (internally) from the real orthogonalized cells.
WS2_POT = "examples/potentials/sw/sw_lammps/t-WS2.sw"


def _sheet_config(mat: str, cif_path: str, pot_path: str) -> SheetConfig:
    """Build a minimal single-monolayer :class:`SheetConfig` for a material.

    x/y = 4.0 Å is smaller than one duplicated unit cell (~6.4 Å) for both
    materials, so build_monolayer's target-size rounding keeps dup=(1, 1):
    the smallest structure it can produce. For potential registration the
    in-plane size is irrelevant (only cif_path/pot_path/pot_type are read),
    but keeping a single, shared constructor mirrors the real build path.
    """
    return SheetConfig(
        mat=mat,
        pot_type="sw",
        pot_path=pot_path,
        cif_path=cif_path,
        x=4.0,
        y=4.0,
        layers=[1],
    )


def _build_small_monolayer(mat: str, cif_path: str, pot_path: str):
    """Build a minimal (single orthogonalized unit cell) monolayer."""
    config = _sheet_config(mat, cif_path, pot_path)
    path, _dims, _pot_counts, _total_types, _supercell_dims = components.build_monolayer(config)
    return path


def test_hetero_type_blocks_disjoint_and_potentials_complete(tmp_path):
    """Each stacked material becomes its own PotentialManager component with a
    disjoint, contiguous atom-type block, its own many-body ``sw`` in a
    ``pair_style hybrid``, and UFF ``lj/cut`` cross-terms for every A-B element
    pair -- all emitted into ``system.in.settings``.

    Guards the old branch's half-done type mapping: the two materials' global
    type-ID sets must be disjoint (no overlap) AND together contiguous/complete
    (cover every registered type, 1..N, no gaps). Type IDs are read from the PM
    type registry, never parsed from text.
    """
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT),
    ]
    settings = load_settings()

    hp = register_hetero_potentials(sheets, settings, workdir=tmp_path)

    text = open(hp.settings_path).read()

    # --- hybrid pair_style with one many-body (sw) entry per material ---
    assert "pair_style hybrid" in text
    style_line = next(l for l in text.splitlines() if l.startswith("pair_style hybrid"))
    assert style_line.split().count("sw") == len(sheets)

    # --- UFF lj/cut cross-terms for every A-B element pair ---
    assert "lj/cut" in text
    cross_lines = [
        l for l in text.splitlines()
        if "lj/cut" in l and l.strip().startswith("pair_coeff")
    ]
    els_a = hp.pm.types.elements_in_component("sheet_1")
    els_b = hp.pm.types.elements_in_component("sheet_2")
    expected_pairs = {frozenset((x, y)) for x in els_a for y in els_b}
    got_pairs = set()
    for line in cross_lines:
        m = re.search(r"#\s*(\w+)\(sheet_1\)-(\w+)\(sheet_2\)", line)
        assert m is not None, f"unexpected cross-interaction line: {line!r}"
        got_pairs.add(frozenset((m.group(1), m.group(2))))
    assert got_pairs == expected_pairs  # e.g. {Mo-W, Mo-S, S-W, S-S}

    # --- Type registry: disjoint AND contiguous/complete (from the PM, not text) ---
    assert len(hp.type_ids) == len(sheets)
    id_sets = [set(ids) for ids in hp.type_ids]
    a_ids, b_ids = id_sets
    assert a_ids and b_ids
    assert a_ids.isdisjoint(b_ids)                      # no overlap
    union = a_ids | b_ids
    assert union == set(range(1, max(union) + 1))       # contiguous from 1, no gaps
    assert len(union) == len(hp.pm.types)               # covers every registered type

    # Cross-check against the live registry and the read_data offsets.
    for name, ids in zip(hp.component_names, hp.type_ids):
        assert hp.pm.types.ids_by_component(name) == ids
    assert hp.type_offsets == [ids[0] - 1 for ids in hp.type_ids]


def _write_synthetic_monolayer(path, cell_2x2, charges=None):
    """Write a minimal 2-atom monolayer LAMMPS data file with a given in-plane cell.

    The cell must be LAMMPS-canonical (a1 along +x). A 20 Å out-of-plane vector
    plus two placeholder atoms give ASE/LAMMPS a valid structure to tile; only
    the in-plane cell shape matters for the coincidence match and box geometry.
    """
    cell_2x2 = np.asarray(cell_2x2, dtype=float)
    cell3 = np.array([
        [cell_2x2[0, 0], 0.0, 0.0],
        [cell_2x2[1, 0], cell_2x2[1, 1], 0.0],
        [0.0, 0.0, 20.0],
    ])
    atoms = Atoms(symbols=["H", "He"], positions=[[0.0, 0.0, 10.0], [0.4, 0.3, 10.0]],
                  cell=cell3, pbc=True)
    style = "atomic"
    if charges is not None:
        atoms.set_initial_charges(charges)
        style = "charge"
    write(str(path), atoms, format="lammps-data", atom_style=style, specorder=["H", "He"])
    return path


def test_matched_supercells_share_one_box_and_reference_is_unstrained(tmp_path):
    mono_a = _build_small_monolayer("h-MoS2", MOS2_CIF, MOS2_POT)
    mono_b = _build_small_monolayer("h-WS2", WS2_CIF, WS2_POT)

    stack = build_matched_supercells(
        mono_a, mono_b, workdir=tmp_path, strain_tol=0.05, max_supercell=6,
    )

    ca = np.array(read(str(stack.supercell_a), format="lammps-data").cell)[:2, :2]
    cb = np.array(read(str(stack.supercell_b), format="lammps-data").cell)[:2, :2]

    assert np.allclose(ca, cb, atol=1e-4)          # shared periodic box
    assert stack.strain_reference == 0.0            # larger material unstrained
    assert stack.strain_applied <= 0.10 + 1e-9       # within one-sided bound (2*strain_tol)
    assert stack.reference in ("a", "b")


# A genuinely sheared, non-identity coincidence between two canonical (a1 along
# +x) in-plane cells: find_coincidence_lattice matches these at
#   matrix_a = [[1, 1], [0, 2]]   (sheared, det 2)
#   matrix_b = [[1, 2], [0, 3]]   (sheared, det 3)
# with a true one-sided strain of ~1.39 %. Both matrices have a non-zero
# off-diagonal, so make_supercell tilts each supercell off-axis; the two built
# boxes only coincide if the strain is applied in the shared co-oriented frame
# (not by independently re-canonicalizing each supercell).
_SHEAR_CELL_A = np.array([[2.889595214489674, 0.0], [-1.5942583249814066, 4.747923033943079]])
_SHEAR_CELL_B = np.array([[3.148752463798015, 0.0], [-1.106026312939178, 2.9183412862744404]])


def test_sheared_nonidentity_match_builds_coinciding_boxes(tmp_path):
    mono_a = _write_synthetic_monolayer(tmp_path / "shear_a.lmp", _SHEAR_CELL_A, charges=[0.5, -0.5])
    mono_b = _write_synthetic_monolayer(tmp_path / "shear_b.lmp", _SHEAR_CELL_B, charges=[0.6, -0.6])

    # Sanity: the coincidence really is sheared and non-identity.
    match = find_coincidence_lattice(_SHEAR_CELL_A, _SHEAR_CELL_B,
                                     strain_tol=0.02, max_supercell=3, area_tol=0.06)
    assert match is not None
    assert match.matrix_a[0, 1] != 0 or match.matrix_b[0, 1] != 0, "expected a sheared match"
    assert not (np.array_equal(match.matrix_a, np.eye(2, dtype=int))
                and np.array_equal(match.matrix_b, np.eye(2, dtype=int)))

    stack = build_matched_supercells(
        mono_a, mono_b, workdir=tmp_path, strain_tol=0.02, max_supercell=3, area_tol=0.06,
    )

    ca = np.array(read(str(stack.supercell_a), format="lammps-data").cell)[:2, :2]
    cb = np.array(read(str(stack.supercell_b), format="lammps-data").cell)[:2, :2]

    # The two BUILT boxes must coincide to high precision (the whole point).
    assert np.allclose(ca, cb, atol=1e-4)
    # And at the true, low strain -- not the ~54 %/24 % a wrong-frame build gives.
    assert stack.strain_applied < 0.02
    assert stack.strain_reference == 0.0
    # The match was built via genuinely non-scalar/sheared matrices.
    assert stack.match.matrix_a[0, 1] != 0 or stack.match.matrix_b[0, 1] != 0

    # Metadata (per-atom charges) survives the build for both materials.
    qa = read(str(stack.supercell_a), format="lammps-data", style="charge").get_initial_charges()
    qb = read(str(stack.supercell_b), format="lammps-data", style="charge").get_initial_charges()
    assert np.allclose(np.abs(qa), np.abs(qa[0]))
    assert np.allclose(np.abs(qb), np.abs(qb[0]))
    assert np.any(qa != 0) and np.any(qb != 0)


def test_incommensurate_pair_raises_instead_of_silently_mismatching(tmp_path):
    """Two monolayers with no low-strain coincidence (within the given repeat/
    strain limits) must be rejected loudly rather than silently written out as
    a "matched" (but actually incommensurate) pair of supercells.
    """
    # Nearly-golden-ratio length ratio: no small commensurate supercell exists
    # under a tight strain budget and a small max repeat.
    cell_a = np.array([[3.0, 0.0], [0.0, 3.0]])
    cell_b = np.array([[3.0 * 1.6180339887, 0.0], [0.0, 3.0]])
    mono_a = _write_synthetic_monolayer(tmp_path / "inc_a.lmp", cell_a)
    mono_b = _write_synthetic_monolayer(tmp_path / "inc_b.lmp", cell_b)

    with pytest.raises(ValueError):
        build_matched_supercells(
            mono_a, mono_b, workdir=tmp_path, strain_tol=0.005, max_supercell=2, area_tol=0.02,
        )
