"""Tests for D1 hetero sheet-on-sheet slide assembly helpers."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from src.core.config import SheetOnSheetSimulationConfig, load_settings
from src.builders.hetero import build_hetero_structure
from src.builders.hetero_slide import (
    HeteroSheetOnSheetSimulation,
    compute_layer_zbands,
    render_hetero_slide,
)
from src.builders.sheetonsheet import SheetOnSheetSimulation
from src.core.run import _select_sheet_builder_cls

MAT, POT = "examples/materials", "examples/potentials/sw"


def _hetero_2p2_config():
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [2]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [2]},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def _hetero_1p1_config():
    raw = {  # 1+1 = 2 layers -> below MIN_LAYERS
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [1]},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def test_layer_zbands_partition_all_atoms(tmp_path):
    cfg = _hetero_2p2_config()
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    bands = compute_layer_zbands(stack.data_path, stack.layers)
    assert [b["idx"] for b in bands] == [1, 2, 3, 4]           # 2+2 -> 4 layers, bottom->top
    # ordered, non-overlapping
    for lo, hi in ((bands[i]["zhi"], bands[i+1]["zlo"]) for i in range(3)):
        assert lo <= hi + 1e-9
    # every atom falls in exactly one band
    z = read(str(stack.data_path), format="lammps-data").get_positions()[:, 2]
    for zi in z:
        hits = [b for b in bands if b["zlo"] <= zi <= b["zhi"]]
        assert len(hits) == 1, f"z={zi} in {len(hits)} bands"

    # Physical correctness: each band must contain exactly the atom count of
    # the layer actually placed at that z-order -- not merely satisfy a
    # contiguous non-overlapping split (any gap-clustering split, including a
    # WRONG one, would still satisfy the assertions above). Bands are ordered
    # bottom->top by layer placement (source native z-range + z-shift), which
    # is exact and independent of gap heuristics -- so this catches silent
    # mis-grouping that gap-clustering could produce for puckered layers
    # (e.g. black phosphorus, GeS) where an intra-layer sub-plane gap exceeds
    # an inter-layer gap.
    native_zrange = {}
    for layer in stack.layers:
        key = str(layer.source)
        if key not in native_zrange:
            zz = read(key, format="lammps-data").get_positions()[:, 2]
            native_zrange[key] = (float(zz.min()), float(zz.max()))
    layers_sorted_by_z = sorted(
        stack.layers,
        key=lambda layer: 0.5 * sum(native_zrange[str(layer.source)]) + layer.z,
    )
    assert len(layers_sorted_by_z) == len(bands)
    for band, layer in zip(bands, layers_sorted_by_z):
        expected = len(read(str(layer.source), format="lammps-data"))
        actual = int(np.sum((band["zlo"] <= z) & (z <= band["zhi"])))
        assert actual == expected, (
            f"band idx={band['idx']} (z=[{band['zlo']:.3f}, {band['zhi']:.3f}]) has "
            f"{actual} atoms, expected {expected} (atom count of the layer placed "
            f"at this z-order, source={layer.source})"
        )


def _write_zcol_atoms(path, zs, cell_xy=(10.0, 10.0)):
    """Write a minimal LAMMPS data file with one atom per given z (fixed xy)."""
    cell3 = np.array([
        [cell_xy[0], 0.0, 0.0],
        [0.0, cell_xy[1], 0.0],
        [0.0, 0.0, 50.0],
    ])
    positions = [[1.0 + 0.1 * i, 1.0, z] for i, z in enumerate(zs)]
    atoms = Atoms(symbols=["H"] * len(zs), positions=positions, cell=cell3, pbc=True)
    write(str(path), atoms, format="lammps-data", atom_style="atomic")
    return path


def test_layer_zbands_catches_puckered_layer_misgrouping(tmp_path):
    """Reviewer's counter-example: a puckered layer (two sub-planes with an
    INTERNAL z-gap larger than the real inter-layer gap) fools gap-clustering
    (the old "(n-1) largest z-gaps" heuristic) into cutting INSIDE the
    puckered layer instead of at the true material boundary -- silently
    mis-grouping atoms. Real puckered 2D materials (black phosphorus, GeS)
    can have exactly this shape. The placement-based fix (known `.source`
    native z-range + `.z` shift) is exact and immune to this, because it
    never looks at gaps at all.

    This test is RED against the old gap-clustering implementation and GREEN
    against the placement-based fix.
    """
    # Layer A: puckered, native atoms at z = 0.0 and 10.0 (internal gap 10.0),
    # placed with no z-shift -> occupies real z in {0.0, 10.0}.
    src_a = _write_zcol_atoms(tmp_path / "puckered_a.lmp", [0.0, 10.0])
    # Layer B: a single atom at native z = 0.0, placed with a 13.0 z-shift ->
    # occupies real z = 13.0. The true inter-layer gap (A's top at 10.0 to B's
    # bottom at 13.0) is only 3.0 -- smaller than A's own internal 10.0 gap.
    src_b = _write_zcol_atoms(tmp_path / "flat_b.lmp", [0.0])

    layers = [
        SimpleNamespace(source=src_a, z=0.0),
        SimpleNamespace(source=src_b, z=13.0),
    ]
    # The assembled structure is literally A's atoms (unshifted) + B's atom
    # shifted by 13.0 -- exactly what `read_data ... shift 0 0 13.0` produces.
    data_path = _write_zcol_atoms(tmp_path / "hetero.lmp", [0.0, 10.0, 13.0])

    bands = compute_layer_zbands(data_path, layers)
    assert [b["idx"] for b in bands] == [1, 2]

    z = read(str(data_path), format="lammps-data").get_positions()[:, 2]
    counts = [int(np.sum((b["zlo"] <= z) & (z <= b["zhi"]))) for b in bands]
    # Band 1 (layer A, the puckered layer) must contain BOTH of A's atoms;
    # band 2 (layer B) must contain exactly B's single atom. The old gap
    # heuristic instead cuts at the largest gap (A's own internal 10.0 gap),
    # yielding counts=[1, 2] -- silently wrong.
    assert counts == [2, 1], f"counts={counts} (old gap-clustering mis-groups this exact case)"


def _minimal_slide_context(layers):
    """Fill every key referenced by ``SheetOnSheetSimulation.write_inputs``'s
    ``base_context`` (src/builders/sheetonsheet.py), plus ``layers`` (Task 1's
    ``compute_layer_zbands`` output), so ``hetero/slide.lmp`` can render
    without hitting an undefined Jinja variable.
    """
    return {
        "temp": 300,
        "xlo": 0.0,
        "xhi": 40.0,
        "ylo": 0.0,
        "yhi": 40.0,
        "zhi": 40.0,
        "data_file": "run/build/hetero.data",
        "potential_file": "run/lammps/system.in.settings",
        "num_atom_types": 4,
        "ngroups": 4,
        "n_layers": 4,
        "constraint_mode": "none",
        "n_bond_types": 0,
        "drive_method": "virtual_atom",
        "thermostat_type": "langevin",
        "atom_style": "atomic",
        "pot_type": "sw",
        "has_internal_lj": True,
        "pressures": 0.0,
        "scan_speed_config": 1,
        "scan_angle_config": 0,
        "scan_angle_force": None,
        "timestep": 0.001,
        "thermo": 1000,
        "run_steps": 10000,
        "min_style": "cg",
        "minimization_command": "minimize 1.0e-8 1.0e-10 10000 100000",
        "neighbor_list": "2.0",
        "neigh_modify_command": "neigh_modify every 1 delay 0 check yes",
        "results_freq": 1000,
        "results_file_pattern": "run/results/friction_p${pressure}_a${a}_s${speed}",
        "dump_enabled": False,
        "dump_freq": 1000,
        "dump_file_pattern": "run/visuals/slide.lammpstrj",
        "driving_spring_ev": 3.121,
        "bond_spring_ev": 4.994,
        "lat_c": 3.16,
        "ev_a_to_nn": 160.2176565,
        "ev_a3_to_gpa": 160.2176565,
        "layers": layers,
    }


def test_hetero_slide_defines_layer_groups_by_region():
    layers = [{"idx": i, "zlo": 5.0 * i, "zhi": 5.0 * i + 3.0} for i in range(1, 5)]
    ctx = _minimal_slide_context(layers)
    script = render_hetero_slide(ctx)

    # Layer groups defined by region, NOT by `group layer_N type` (hetero
    # layers share atom types, so type-based groups are impossible).
    assert "region          reg_layer_1" in script or "region reg_layer_1" in script
    for i in range(1, 5):
        assert "group" in script and f"layer_{i}" in script
        assert f"group           layer_{i} region reg_layer_{i}" in script
    assert "group layer_1 type" not in script

    # Single read of the already-assembled hetero stack -- no `add append`
    # multi-stack read like the homogeneous template's create_box branch.
    assert script.count("read_data") == 1
    assert "add append" not in script

    # The homogeneous slide body (thermostat, drive, pressure, friction proxy)
    # is reused verbatim.
    assert "center" in script and "fix" in script
    assert "aveforce" in script


def test_builder_requires_four_layers(tmp_path):
    cfg = _hetero_1p1_config()  # 1+1 = 2 layers -> below MIN_LAYERS
    sim = HeteroSheetOnSheetSimulation(cfg, str(tmp_path))
    with pytest.raises(ValueError, match=r"[Aa]t least 4"):
        sim.build()


def test_builder_writes_slide_in_referencing_hetero_stack(tmp_path):
    cfg = _hetero_2p2_config()
    sim = HeteroSheetOnSheetSimulation(cfg, str(tmp_path))
    sim.build()

    slide = next(Path(tmp_path).rglob("slide*.in"))
    text = slide.read_text()
    assert "hetero.data" in text                       # sources the assembled stack
    assert "system.in.settings" in text
    assert "group           layer_1 region" in text or "layer_1 region" in text


def _single_material_config():
    """A homogeneous config with a single [2D] section -> cfg.sheets has length 1."""
    raw = {
        "general": {"temp": 300, "scan_speed": 1},
        "2D": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
               "pot_type": "sw", "x": 8, "y": 8, "layers": [4]},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def test_routing_selects_hetero_for_two_materials():
    assert _select_sheet_builder_cls("sheetonsheet", _hetero_2p2_config()) is HeteroSheetOnSheetSimulation


def test_routing_selects_homogeneous_for_single_material():
    cfg = _single_material_config()
    assert len(cfg.sheets) == 1
    assert _select_sheet_builder_cls("sheetonsheet", cfg) is SheetOnSheetSimulation
