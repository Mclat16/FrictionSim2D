"""Tests for the freestanding (substrate-free) tip PES builder."""
from pathlib import Path

import pytest

from src.builders.afm_freestanding import FreestandingPESTipSimulation
from src.core.config import AFMSimulationConfig, load_settings


def _free_cfg(tmp_path, layers=(4,), eval_mode="minimize"):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    return AFMSimulationConfig(**{
        "general": {"temp": 300.0, "force": [2.0], "scan_speed": 2.0, "finite_sheet": False},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 40.0, "y": 40.0, "layers": list(layers)},
        "tip": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "r": 25.0, "amorph": "c"},
        "pes": {"grid_n": 8, "eval_mode": eval_mode},
        "settings": load_settings().model_dump(),
    })


def test_freestanding_hpc_job_name(tmp_path):
    builder = FreestandingPESTipSimulation(_free_cfg(tmp_path), output_dir=str(tmp_path / "o"))
    assert builder._get_hpc_job_name() == "pes_tip_free_h-MoS2"


def test_freestanding_pes_template_fixes_layer1_no_substrate(tmp_path, monkeypatch):
    """write_inputs must render a slide.in that fixes layer_1 and never reads a substrate."""
    n = 4
    builder = FreestandingPESTipSimulation(_free_cfg(tmp_path, layers=(n,)), output_dir=str(tmp_path / "o"))

    # Stand in for the geometry/potential state build() would populate.
    ctx = {
        "atom_style": "atomic", "neighbor_list": 2.0,
        "neigh_modify_command": "neigh_modify every 1 delay 0 check yes",
        "thermo": 100, "timestep": 0.001, "xlo": 0.0, "xhi": 40.0, "ylo": 0.0, "yhi": 40.0,
        "zhi_box": 120.0, "ngroups": 8, "tip_file": "build/tip.lmp", "sub_file": None,
        "sheet_file": "build/sheet.lmp", "tip_x": 20.0, "tip_y": 20.0, "tip_z": 80.0,
        "sheet_shift_x": 0.0, "sheet_shift_y": 0.0, "sheet_z": 5.0, "offset_2d": 2,
        "potential_file": "lammps/system.in.settings", "min_style": "cg",
        "minimization_command": "minimize 0.0 1.0e-6 10000 100000",
        "contact_gap": 3.0, "ev_a_to_nn": 16.021766,
    }
    monkeypatch.setattr(builder, "build_render_context", lambda n_layers: dict(ctx))
    builder.sheet_unit_cell = {"xlo": 0.0, "xhi": 3.16, "ylo": 0.0, "yhi": 5.47}
    builder.box_dims = {"xlo": 0.0, "xhi": 40.0, "ylo": 0.0, "yhi": 40.0}
    builder.output_dir_layer = {n: tmp_path / "o" / f"L{n}"}
    builder.relative_run_dir_layer = {n: Path(f"pes_tip/h-MoS2/L{n}")}

    builder.write_inputs(n)

    slide = (tmp_path / "o" / f"L{n}" / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "read_data" in slide and "group sub" not in slide      # no substrate
    assert "fix             slab_anchor layer_1 setforce 0.0 0.0 0.0" in slide
    assert "x,y,energy_eV,fx,fy,fz" in slide                      # descriptor columns
    assert "Nx     equal 8" in slide


def test_freestanding_pes_requires_two_layers(tmp_path):
    builder = FreestandingPESTipSimulation(_free_cfg(tmp_path, layers=(1,)), output_dir=str(tmp_path / "o"))
    with pytest.raises(ValueError, match="layers"):
        builder.write_inputs(1)


def test_freestanding_md_mode_emits_langevin_and_run(tmp_path, monkeypatch):
    n = 4
    builder = FreestandingPESTipSimulation(_free_cfg(tmp_path, layers=(n,), eval_mode="md"),
                                           output_dir=str(tmp_path / "o"))
    ctx = {
        "atom_style": "atomic", "neighbor_list": 2.0,
        "neigh_modify_command": "neigh_modify every 1 delay 0 check yes",
        "thermo": 100, "timestep": 0.001, "xlo": 0.0, "xhi": 40.0, "ylo": 0.0, "yhi": 40.0,
        "zhi_box": 120.0, "ngroups": 8, "tip_file": "build/tip.lmp", "sub_file": None,
        "sheet_file": "build/sheet.lmp", "tip_x": 20.0, "tip_y": 20.0, "tip_z": 80.0,
        "sheet_shift_x": 0.0, "sheet_shift_y": 0.0, "sheet_z": 5.0, "offset_2d": 2,
        "potential_file": "lammps/system.in.settings", "min_style": "cg",
        "minimization_command": "minimize 0.0 1.0e-6 10000 100000",
        "contact_gap": 3.0, "ev_a_to_nn": 16.021766, "temp": 300.0,
    }
    monkeypatch.setattr(builder, "build_render_context", lambda n_layers: dict(ctx))
    builder.sheet_unit_cell = {"xlo": 0.0, "xhi": 3.16, "ylo": 0.0, "yhi": 5.47}
    builder.box_dims = {"xlo": 0.0, "xhi": 40.0, "ylo": 0.0, "yhi": 40.0}
    builder.output_dir_layer = {n: tmp_path / "o" / f"L{n}"}
    builder.relative_run_dir_layer = {n: Path(f"pes_tip/h-MoS2/L{n}")}

    builder.write_inputs(n)
    slide = (tmp_path / "o" / f"L{n}" / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "langevin" in slide and "layer_2" in slide
    assert "run             2000" in slide
    assert "fix_ave" in slide or "ave/time" in slide  # per-point time averaging


def test_freestanding_logs_resolved_config(tmp_path, caplog):
    import logging
    builder = FreestandingPESTipSimulation(_free_cfg(tmp_path, layers=(4,)), output_dir=str(tmp_path / "o"))
    with caplog.at_level(logging.INFO):
        builder._log_resolved_config(4)
    assert any(
        "freestanding" in r.message.lower() and "4-layer" in r.message
        and "contact gap" in r.message and "not enforced" in r.message
        for r in caplog.records
    )
