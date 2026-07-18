"""Tests for D1 hetero sheet-on-sheet slide assembly helpers."""
import shutil
import subprocess
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
LMP = shutil.which("lmp_serial")


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


# --------------------------------------------------------------------------- #
# End-to-end smoke: build a real 2+2 MoS2/WS2 hetero slide and RUN it in LAMMPS
# --------------------------------------------------------------------------- #

def _run_lammps_slide(run_dir, slide_in_name, timeout=180):
    """Run ``lmp_serial`` on the rendered hetero slide and return
    ``(completed_process, combined_output)``.

    ``cwd`` MUST be ``run_dir``: every path baked into ``slide.in`` is
    RUN-DIR-RELATIVE (``hetero.data``, ``lammps/system.in.settings``, the
    settings' ``provenance/potentials/...`` prefix, ``results/...``), so they
    only resolve when LAMMPS is launched from the run dir. ``combined_output``
    is stdout + stderr + the LAMMPS ``log.lammps`` (if written), so callers can
    grep it for errors / "Lost atoms".
    """
    run_dir = Path(run_dir)
    proc = subprocess.run(
        ["lmp_serial", "-in", str(run_dir / "lammps" / slide_in_name)],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    log = run_dir / "log.lammps"
    if log.exists():
        combined += "\n" + log.read_text()
    return proc, combined


def _read_friction_records(friction_path):
    """Parse a ``fix ave/time`` results file into ``(column_names, rows)``.

    The second comment line names the columns
    (``# TimeStep v_xfrict ... v_comx_top v_comy_top ...``); data rows are the
    non-comment lines. Column order is taken from that header (which mirrors the
    rendered ``fix fc_ave ... ave/time`` output list), so the parse survives any
    reordering of the emitted variables. ``column_names`` includes the leading
    ``TimeStep`` and is index-aligned with every data row.
    """
    cols = None
    rows = []
    for line in Path(friction_path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            toks = s.lstrip("#").split()
            if toks and toks[0] == "TimeStep":
                cols = toks
            continue
        rows.append([float(x) for x in s.split()])
    return cols, rows


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_slide_runs_and_drives_top_layer(tmp_path):
    """Capstone D1 smoke: assemble a real MoS2/WS2 2+2 hetero stack and actually
    RUN the slide in LAMMPS for a few thousand steps (NOT mocked). Proves the
    whole pipeline works end to end: run-dir-relative path resolution (data,
    settings, staged potentials, results), the box + z-vacuum, the z-band layer
    groups, and a live virtual_atom drive on the top layer.
    """
    cfg = _hetero_2p2_config()
    # A tiny driven run for a smoke. The friction fix is
    # ``ave/time 1 1000 {{results_freq}}`` (Nrepeat=1000, Nfreq=results_freq=1000)
    # and is defined AFTER the template's fixed 10000-step equilibration, so its
    # first record lands at step 12000 and one more every 1000 steps thereafter.
    # 4000 driven steps -> records at 12000/13000/14000 (>=2, needed for the
    # displacement check). The whole LAMMPS run is ~5 s at this size (~288 atoms).
    cfg.settings.simulation.slide_run_steps = 4000
    cfg.settings.simulation.drive_method = "virtual_atom"
    # The matched supercell must be larger than the hetero cross-LJ cutoff
    # (``pair_style hybrid ... lj/cut 11.0``) or ``comm_style tiled`` aborts with
    # "Communication cutoff ... cannot exceed periodic box length". The 12 A
    # target of _hetero_2p2_config yields an ~11.04 A box edge (too tight); a
    # 14 A target gives a ~12.7 A min edge, comfortably clear of 11.0 + skin.
    for sheet in cfg.sheets:
        sheet.x = 14.0
        sheet.y = 14.0
    # Drive fast enough that the drive translation dominates thermal COM jitter
    # (speed=100 -> ~1 A/ps -> ~3 A of top-layer travel over the driven run).
    cfg.general.scan_speed = 100

    sim = HeteroSheetOnSheetSimulation(cfg, str(tmp_path))
    sim.build()

    slide = next(Path(tmp_path).rglob("slide*.in"))
    # run dir = the dir that holds lammps/, hetero.data, provenance/, results/
    run_dir = slide.parent.parent
    assert (run_dir / "hetero.data").exists()
    assert (run_dir / "lammps" / "system.in.settings").exists()

    proc, out = _run_lammps_slide(run_dir, slide.name)

    # 1) LAMMPS finished cleanly, with no lost atoms.
    assert proc.returncode == 0, f"lmp_serial exited {proc.returncode}:\n{out[-3000:]}"
    assert "Lost atoms" not in out, f"atoms lost during slide:\n{out[-3000:]}"

    # 2) A non-empty friction results file was written into results/.
    friction = next(run_dir.rglob("friction_*"))
    assert friction.stat().st_size > 0, f"empty friction file: {friction}"

    # 3) The drive is LIVE: the top layer's in-plane COM moved a nonzero net
    #    amount between the first and last record. A virtual_atom drive at
    #    nonzero speed MUST translate it (here ~3 A along the drive axis, well
    #    above the ~0.5 A thermal COM jitter of a 48-atom layer).
    cols, rows = _read_friction_records(friction)
    assert cols is not None, f"no column header parsed from {friction}"
    assert len(rows) >= 2, f"need >=2 records to measure displacement, got {len(rows)}"
    ix, iy = cols.index("v_comx_top"), cols.index("v_comy_top")
    dx = rows[-1][ix] - rows[0][ix]
    dy = rows[-1][iy] - rows[0][iy]
    disp = (dx * dx + dy * dy) ** 0.5
    assert disp > 0.5, (
        f"top-layer COM barely moved (dx={dx:.3f}, dy={dy:.3f}, |d|={disp:.3f} A); "
        f"virtual_atom drive appears dead"
    )
