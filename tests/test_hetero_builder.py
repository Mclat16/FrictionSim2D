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
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from src.builders import components
from src.builders.hetero import (
    _detect_atom_style,
    _run_lammps,
    assemble_hetero_stack,
    build_hetero_structure,
    build_matched_supercells,
    register_hetero_potentials,
)
from src.core.config import SheetConfig, SheetOnSheetSimulationConfig, load_settings
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

    # --- Emitted masses are correct for EVERY type (range-emission + override) ---
    # `get_masses_string` emits ranges (e.g. `mass 2*5 #S`) that can transiently
    # engulf a type a later line overrides (`mass 3 #W`). A run-0 energy does NOT
    # validate mass values, so assert the FINAL mass per type (last matching
    # `mass` line wins, LAMMPS semantics) equals its element's standard mass.
    from ase.data import atomic_masses, atomic_numbers
    n_types = len(union)
    final_mass: dict[int, float] = {}
    final_elem: dict[int, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("mass"):
            continue
        parts = s.split()
        spec, val = parts[1], float(parts[2])
        elem = s.split("#", 1)[1].strip().split()[0] if "#" in s else None
        if "*" in spec:
            lo_s, hi_s = spec.split("*")
            lo, hi = (int(lo_s) if lo_s else 1), (int(hi_s) if hi_s else n_types)
        else:
            lo = hi = int(spec)
        for t in range(lo, hi + 1):
            final_mass[t], final_elem[t] = val, elem
    assert set(final_mass) == set(range(1, n_types + 1))   # every type got a mass
    for t in range(1, n_types + 1):
        el = final_elem[t]
        assert el is not None and final_mass[t] > 0
        expected = atomic_masses[atomic_numbers[el]]
        assert abs(final_mass[t] - expected) < 0.5, (
            f"type {t} ({el}) got mass {final_mass[t]}, expected ~{expected:.2f}"
        )
    assert len(union) == len(hp.pm.types)               # covers every registered type

    # Cross-check against the live registry and the read_data offsets.
    for name, ids in zip(hp.component_names, hp.type_ids):
        assert hp.pm.types.ids_by_component(name) == ids
    assert hp.type_offsets == [ids[0] - 1 for ids in hp.type_ids]


def _sw_element_list(pair_coeff_line):
    """Extract the trailing element list from a hybrid ``pair_coeff * * sw ...``
    line, i.e. every token after the potential-file path (the token containing a
    ``/`` or ``.``), stripped of any comment. For ``pair_coeff * * sw 1
    provenance/potentials/MoS2_wen.sw Mo S NULL NULL`` this returns
    ``['Mo', 'S', 'NULL', 'NULL']`` -- one entry per box atom type, which is what
    LAMMPS requires for a many-body pair_coeff.
    """
    main = pair_coeff_line.split("#", 1)[0].strip()
    tokens = main.split()
    path_idx = next(i for i, t in enumerate(tokens) if "/" in t or "." in t)
    return tokens[path_idx + 1:]


def test_reserve_virtual_type_sizes_sw_element_lists_and_default_unchanged(tmp_path):
    """``reserve_virtual_type=True`` registers ONE extra non-interacting virtual
    type BEFORE the many-body self-interactions (the homogeneous drive-atom
    ordering), so every ``pair_coeff * * sw ...`` element list is sized to the
    full box type count (real types + a trailing NULL for the virtual type) the
    first time -- LAMMPS-acceptable, no post-hoc rebuild -- and the settings
    carry the virtual type's zero-LJ ``pair_coeff``. The default
    (``reserve_virtual_type=False``) is the real-types-only spine: no virtual
    type, no ``1e-100`` term, sw lists sized to the real count.
    """
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT),
    ]
    settings = load_settings()

    # --- default: NO virtual type, sw lists sized to the real count ----------
    hp_def = register_hetero_potentials(sheets, settings, workdir=tmp_path / "default")
    real_count = sum(len(ids) for ids in hp_def.type_ids)
    assert len(hp_def.pm.types) == real_count            # no extra type
    def_text = Path(hp_def.settings_path).read_text()
    assert "1e-100" not in def_text                       # no virtual pair_coeff
    assert "Virtual" not in def_text
    def_sw = [l for l in def_text.splitlines() if l.strip().startswith("pair_coeff * * sw")]
    assert len(def_sw) == len(sheets)
    for line in def_sw:
        assert len(_sw_element_list(line)) == real_count

    # --- reserve_virtual_type=True: exactly ONE extra type -------------------
    hp = register_hetero_potentials(
        sheets, settings, workdir=tmp_path / "virtual", reserve_virtual_type=True
    )
    # type_ids/component_names still describe the REAL materials only ...
    assert sum(len(ids) for ids in hp.type_ids) == real_count
    assert hp.component_names == hp_def.component_names
    # ... while pm.types holds one extra (virtual) type.
    assert len(hp.pm.types) == real_count + 1
    n_box_types = len(hp.pm.types)

    text = Path(hp.settings_path).read_text()
    # (a) the virtual type's zero-LJ pair_coeff + mass are present on type N.
    assert f"pair_coeff * {n_box_types} lj/cut 1e-100 1e-100" in text
    assert "Virtual atom" in text                         # mass N 1.0 #Virtual atom
    # (b) EVERY many-body sw element list is sized to the full box type count
    #     (real types + trailing NULL for the virtual type) -> LAMMPS accepts it.
    sw_lines = [l for l in text.splitlines() if l.strip().startswith("pair_coeff * * sw")]
    assert len(sw_lines) == len(sheets)
    for line in sw_lines:
        elems = _sw_element_list(line)
        assert len(elems) == n_box_types, (
            f"sw element list has {len(elems)} entries, expected {n_box_types} "
            f"(one per box type incl. trailing NULL for the virtual type): {line!r}"
        )
        # the reserved virtual type's slot (last) is a non-interacting NULL.
        assert elems[-1] == "NULL"


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
    qa = read(str(stack.supercell_a), format="lammps-data", atom_style="charge").get_initial_charges()
    qb = read(str(stack.supercell_b), format="lammps-data", atom_style="charge").get_initial_charges()
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


def _wrap_vec_mod_cell(vec, cell_2x2):
    """Wrap a 2-D Cartesian vector into the primitive cell (nearest-image)."""
    inv = np.linalg.inv(np.asarray(cell_2x2, dtype=float))
    frac = np.asarray(vec, dtype=float) @ inv
    frac -= np.round(frac)
    return frac @ np.asarray(cell_2x2, dtype=float)


def test_hetero_stack_is_periodic_and_complete(tmp_path):
    """Assemble two matched supercells into ONE periodic LAMMPS cell and verify
    the built geometry: correct atom count, a physical (ordered, non-overlapping,
    non-exploded) z-extent, a single shared in-plane box (no seam), and -- the
    key regression guard -- the cross-material RI offset SURVIVES placement (the
    old branch re-centered both materials' centroids, cancelling it).
    """
    # The two pre-aligned supercells are already in AA (max-RI) registry at zero
    # shift, so AA would give a (0, 0) offset that cannot exercise the survival
    # guard. AB (anti-registry) is a genuine, non-zero half-cell shift -- use it
    # so the guard is meaningful. Only the non-reference (top) material's
    # stack_type drives the cross-material offset; set both for robustness.
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT).model_copy(update={"stack_type": "AB"}),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT).model_copy(update={"stack_type": "AB"}),
    ]
    settings = load_settings()

    mono_a = _build_small_monolayer("h-MoS2", MOS2_CIF, MOS2_POT)
    mono_b = _build_small_monolayer("h-WS2", WS2_CIF, WS2_POT)

    matched = build_matched_supercells(
        mono_a, mono_b, workdir=tmp_path, strain_tol=0.05, max_supercell=6,
    )
    hp = register_hetero_potentials(sheets, settings, workdir=tmp_path)

    stack = assemble_hetero_stack(matched, sheets, hp, "grouped", settings, workdir=tmp_path)

    # --- total atoms == sum of both matched supercells (1 layer each) ---
    na = len(read(str(matched.supercell_a), format="lammps-data"))
    nb = len(read(str(matched.supercell_b), format="lammps-data"))
    expected_total = na + nb

    at = read(str(stack.data_path), format="lammps-data")
    assert len(at) == expected_total

    # --- distinct, ordered, non-overlapping, non-exploded z stacking ---
    zc = at.get_positions()[:, 2]
    zext = zc.max() - zc.min()
    assert 4.0 < zext < 20.0

    # --- single shared periodic in-plane box (matches the matched supercell) ---
    matched_box_2x2 = np.array(read(str(matched.supercell_a), format="lammps-data").cell)[:2, :2]
    assert np.allclose(np.array(at.cell)[:2, :2], matched_box_2x2, atol=1e-4)

    # --- RI-offset-survival regression guard ---------------------------------
    # Identify (via the stack's own bookkeeping) which stacked layers belong to
    # the reference vs the displaced non-reference material, split the atoms by
    # z accordingly, and check the applied lateral offset actually survives.
    ref_z = np.mean([l.z for l in stack.layers if l.material_index == stack.reference_index])
    non_z = np.mean([l.z for l in stack.layers if l.material_index == stack.nonreference_index])
    # The two materials form two well-separated z-clusters; split the ACTUAL
    # atom z at the midpoint of their range. The material placed at the lower
    # z-shift is the lower atom cluster.
    z_split = 0.5 * (zc.min() + zc.max())
    xy = at.get_positions()[:, :2]
    low_xy, high_xy = xy[zc < z_split], xy[zc >= z_split]
    if non_z < ref_z:
        non_xy, ref_xy = low_xy, high_xy
    else:
        non_xy, ref_xy = high_xy, low_xy

    diff = non_xy.mean(axis=0) - ref_xy.mean(axis=0)
    offset = np.asarray(stack.ri_offset, dtype=float)

    # (a) the non-reference material sits at the intended RI offset (mod cell)...
    assert np.linalg.norm(_wrap_vec_mod_cell(diff - offset, matched_box_2x2)) < 1e-2
    # (b) ...and that offset is genuinely non-zero -- NOT centroid-cancelled.
    assert np.linalg.norm(_wrap_vec_mod_cell(offset, matched_box_2x2)) > 0.5


# =============================================================================
# Task 6: end-to-end structure smoke -- build_hetero_structure -> real LAMMPS
# =============================================================================


def _two_material_config() -> SheetOnSheetSimulationConfig:
    """A real 2-material (MoS2/WS2) SheetOnSheetSimulationConfig via the config path.

    Mirrors the raw-dict [2D-1]/[2D-2] shape of
    ``test_config.py::test_two_material_sections_fold_into_sheets_list`` but with
    the REAL CIFs + SW potentials (so ``build_monolayer`` actually runs) and one
    layer each. ``x``/``y`` are in ANGSTROMS; 4.0 Å is smaller than one
    duplicated unit cell (~6.4 Å) for both materials, so ``build_monolayer``'s
    target-size rounding keeps each supercell at the minimal single ortho cell
    (6 atoms) -- the fastest structure that still exercises the full
    match -> register -> assemble -> LAMMPS path. (30 Å, by contrast, rounds up
    to a ~9x5 supercell of 270 atoms per material -- needlessly slow for a smoke
    test whose point is structural self-consistency, not size.)
    """
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": MOS2_CIF, "pot_path": MOS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": WS2_CIF, "pot_path": WS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def _lammps_run0_energy(data_path, settings_path) -> float:
    """Drive real LAMMPS on the produced files and return the ``run 0`` energy.

    Emits a tiny self-contained script (``units metal``, ``atom_style`` detected
    from the data file, ``boundary p p p``, ``read_data`` + ``include`` the real
    settings, ``run 0``) and runs ``lmp_serial`` headless with ``cwd = workdir``
    -- mandatory, because the settings' ``pair_coeff`` lines reference the staged
    potentials by the run-dir-relative prefix ``provenance/potentials/...``. Both
    ``data_path`` and ``settings_path`` are emitted as absolute paths so they
    resolve independent of cwd. Asserts the log is free of "Lost atoms" (a lost
    atom means the structure is not self-consistent / not genuinely periodic) and
    returns the parsed potential energy.
    """
    data_path = Path(data_path).resolve()
    settings_path = Path(settings_path).resolve()
    workdir = data_path.parent  # hetero.data lives directly in the run dir
    atom_style = _detect_atom_style(data_path)

    script = f"""units           metal
atom_style      {atom_style}
boundary        p p p
read_data       {data_path}
include         {settings_path}
run             0
variable        pe equal pe
variable        na equal atoms
print           "RESULT_PE ${{pe}}"
print           "RESULT_NATOMS ${{na}}"
"""
    script_path = workdir / "run0_smoke.lmp"
    script_path.write_text(script, encoding="utf-8")
    out = _run_lammps(workdir, script_path)

    assert "Lost atoms" not in out, f"LAMMPS reported lost atoms:\n{out[-2000:]}"

    energy = None
    for line in out.splitlines():
        if line.startswith("RESULT_PE"):
            energy = float(line.split()[1])
    assert energy is not None, f"could not parse RESULT_PE from LAMMPS:\n{out[-2000:]}"
    return energy


def test_end_to_end_structure_assembles_in_lammps(tmp_path):
    """Capstone: the single ``build_hetero_structure`` entrypoint ties Tasks 3-5
    together and yields a ``hetero.data`` + ``system.in.settings`` that assemble
    in LAMMPS. A finite, physical ``run 0`` energy with no lost atoms proves the
    structure + atom types + potentials are self-consistent and the box is
    genuinely periodic (p p p).
    """
    cfg = _two_material_config()
    settings = cfg.settings

    stack = build_hetero_structure(cfg, settings, workdir=tmp_path)

    # The entrypoint returns a ready stack pointing at real, existing files.
    assert Path(stack.data_path).exists()
    assert Path(stack.settings_path).exists()

    at = read(str(stack.data_path), format="lammps-data")

    # --- atom count == sum over placed layers of their source-supercell atoms ---
    expected_total = sum(
        len(read(str(layer.source), format="lammps-data")) for layer in stack.layers
    )
    assert len(at) == expected_total

    # --- genuinely 3D-periodic box (all pbc, positive volume) -------------------
    assert bool(np.all(at.pbc))
    assert at.get_volume() > 0.0

    # --- real LAMMPS run 0: finite, physical energy; no lost atoms (in helper) --
    energy = _lammps_run0_energy(stack.data_path, stack.settings_path)
    assert np.isfinite(energy) and abs(energy) < 1e6  # physical, not exploded (~1e9 = failure)


def test_lattice_match_config_is_threaded_into_build(tmp_path):
    """``build_hetero_structure`` must thread the ``[lattice_match]`` section into
    the coincidence search (design §5), not use hardcoded defaults.

    Proof: a ``strain_tol`` far tighter than the ~0.03% MoS2/WS2 mismatch turns
    the otherwise-successful match into a fail-loud, and the raised error reports
    *that* tolerance -- so the config value reached the matcher (with the 2%
    default the match would succeed). Fails at the match step, before any LAMMPS
    run, so it is fast.
    """
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": MOS2_CIF, "pot_path": MOS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": WS2_CIF, "pot_path": WS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
        "lattice_match": {"strain_tol": 1e-6, "max_supercell": 3},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    assert cfg.lattice_match is not None and cfg.lattice_match.strain_tol == 1e-6

    with pytest.raises(ValueError, match=r"strain_tol=0\.0001%"):
        build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
