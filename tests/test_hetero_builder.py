"""Tests for src.builders.hetero.build_matched_supercells.

Builds real (small) MoS2 and WS2 monolayers via the existing builder,
finds their Zur-McGill coincidence lattice, and verifies that
``build_matched_supercells`` produces two per-material supercells that
share exactly one in-plane periodic box, with the larger-area material
left unstrained and the smaller strained onto it within tolerance.
"""
import numpy as np
import pytest
from ase.io import read

from src.builders import components
from src.builders.hetero import build_matched_supercells
from src.core.config import SheetConfig
from src.core.lattice_matching import find_coincidence_lattice_from_cif

MOS2_CIF = "examples/materials/h-MoS2.cif"
WS2_CIF = "examples/materials/h-WS2.cif"
MOS2_POT = "examples/potentials/sw/MoS2_wen.sw"
# ws2_jrmidas.sw defines a third element ('C', for graphene coupling) that is
# not present in h-WS2.cif and is rejected by get_potential_element_order, so
# a plain WS2-only SW potential is used instead. Both this file and
# MoS2_wen.sw assign exactly one LAMMPS type per element (no sublattice
# splitting), so both monolayers' orthogonalized unit cells are a plain 1x2
# doubling of their CIF primitive cell -- i.e. directly comparable to each
# other and to the coincidence match found from the raw CIF cells below.
WS2_POT = "examples/potentials/sw/sw_lammps/t-WS2.sw"


def _build_small_monolayer(mat: str, cif_path: str, pot_path: str):
    """Build a minimal (single orthogonalized unit cell) monolayer.

    x/y = 4.0 Å is smaller than one duplicated unit cell (~6.4 Å) for both
    materials, so build_monolayer's target-size rounding keeps dup=(1, 1):
    the smallest structure it can produce, i.e. exactly its orthogonalized
    unit cell with no extra tiling.
    """
    config = SheetConfig(
        mat=mat,
        pot_type="sw",
        pot_path=pot_path,
        cif_path=cif_path,
        x=4.0,
        y=4.0,
        layers=[1],
    )
    path, _dims, _pot_counts, _total_types, _supercell_dims = components.build_monolayer(config)
    return path


def _find_match():
    match = find_coincidence_lattice_from_cif(
        cif_a=MOS2_CIF, cif_b=WS2_CIF, strain_tol=0.05, max_supercell=6,
    )
    assert match is not None, "expected a low-strain MoS2/WS2 coincidence match"
    return match


def test_matched_supercells_share_one_box_and_reference_is_unstrained(tmp_path):
    mono_a = _build_small_monolayer("h-MoS2", MOS2_CIF, MOS2_POT)
    mono_b = _build_small_monolayer("h-WS2", WS2_CIF, WS2_POT)

    match = _find_match()

    stack = build_matched_supercells(mono_a, mono_b, match, workdir=tmp_path)

    ca = np.array(read(str(stack.supercell_a), format="lammps-data").cell)[:2, :2]
    cb = np.array(read(str(stack.supercell_b), format="lammps-data").cell)[:2, :2]

    assert np.allclose(ca, cb, atol=1e-4)          # shared periodic box
    assert stack.strain_reference == 0.0            # larger material unstrained
    assert stack.strain_applied <= 0.05 + 1e-9       # smaller within tol
    assert stack.reference in ("a", "b")


def test_tampered_match_raises_instead_of_silently_mismatching(tmp_path):
    """A match whose declared matrices don't actually reproduce a low-strain
    coincidence must be rejected loudly rather than silently written out as
    a "matched" (but actually incommensurate) pair of supercells.
    """
    mono_a = _build_small_monolayer("h-MoS2", MOS2_CIF, MOS2_POT)
    mono_b = _build_small_monolayer("h-WS2", WS2_CIF, WS2_POT)

    match = _find_match()
    # Tamper with B's supercell matrix: quadruple its area while the match
    # still claims (via strain_a/strain_b, left untouched) a near-zero
    # strain. build_matched_supercells must independently measure the real
    # geometric strain rather than trust the match object, and reject this.
    match.matrix_b = np.array([[2, 0], [0, 2]])

    with pytest.raises(ValueError):
        build_matched_supercells(mono_a, mono_b, match, workdir=tmp_path)
