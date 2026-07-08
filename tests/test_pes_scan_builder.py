"""Tests for the static PES-scan builders (src/builders/pes_scan.py)."""

from pathlib import Path

import pytest

from src.builders.pes_scan import (
    PESSheetSimulation,
    PESTipSimulation,
    _cell_period,
    _tile_count,
)
from src.core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
)


def _sheet_cfg(tmp_path: Path, layers, grid_n=12, z_relax=True):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    return SheetOnSheetSimulationConfig(**{
        "general": {"temp": 0.0},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 30.0, "y": 30.0, "layers": layers},
        "pes": {"grid_n": grid_n, "z_relax": z_relax},
        "settings": load_settings().model_dump(),
    })


def _afm_cfg(tmp_path: Path, force):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    return AFMSimulationConfig(**{
        "general": {"temp": 300.0, "force": force, "scan_speed": 2.0, "finite_sheet": False},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 60.0, "y": 60.0, "layers": [1]},
        "tip": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "r": 25.0, "amorph": "c"},
        "sub": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "thickness": 12.0, "amorph": "a"},
        "pes": {"grid_n": 16},
        "settings": load_settings().model_dump(),
    })


# --- helpers ---------------------------------------------------------------

def test_cell_period_and_tile_count():
    uc = {"xlo": 0.0, "xhi": 3.0, "ylo": 0.0, "yhi": 5.0}
    assert _cell_period(uc, "x") == pytest.approx(3.0)
    assert _cell_period(uc, "y") == pytest.approx(5.0)
    full = {"xlo": 0.0, "xhi": 30.0, "ylo": 0.0, "yhi": 25.0}
    assert _tile_count(full, uc) == 10 * 5


def test_cell_period_missing_raises():
    with pytest.raises(ValueError, match="unit-cell"):
        _cell_period({}, "x")


# --- PESSheetSimulation ----------------------------------------------------

def test_pes_sheet_min_layers_is_two():
    assert PESSheetSimulation.MIN_LAYERS == 2


def test_pes_sheet_rejects_non_bilayer(tmp_path: Path):
    builder = PESSheetSimulation(_sheet_cfg(tmp_path, layers=[3]), output_dir=str(tmp_path / "o"))
    with pytest.raises(ValueError, match="exactly 2 layers"):
        builder.build()


def test_pes_sheet_no_virtual_drive_atom(tmp_path: Path):
    """The static scan must not register the virtual driving atom."""
    cfg = _sheet_cfg(tmp_path, layers=[2])
    cfg.settings.simulation.drive_method = "virtual_atom"
    builder = PESSheetSimulation(cfg, output_dir=str(tmp_path / "o"))

    calls = {"virtual": 0}

    class _PM:
        def register_virtual_atom(self):
            calls["virtual"] += 1

    builder._register_drive_atom(_PM())
    assert calls["virtual"] == 0


def test_pes_sheet_build_forces_none_constraint(tmp_path: Path, monkeypatch):
    cfg = _sheet_cfg(tmp_path, layers=[2])
    cfg.settings.simulation.constraint_mode = "atom_bonds"
    builder = PESSheetSimulation(cfg, output_dir=str(tmp_path / "o"))

    seen = {}

    def fake_super_build(self):
        seen["constraint_mode"] = self.config.settings.simulation.constraint_mode

    monkeypatch.setattr("src.builders.sheetonsheet.SheetOnSheetSimulation.build", fake_super_build)
    builder.build()
    assert seen["constraint_mode"] == "none"


def test_pes_sheet_write_inputs_renders_scan(tmp_path: Path):
    """write_inputs should emit a slide.in PES scan (+ refine) with grid context."""
    cfg = _sheet_cfg(tmp_path, layers=[2])
    cfg.pes.grid_n_refine = 20
    out = tmp_path / "o"
    builder = PESSheetSimulation(cfg, output_dir=str(out))
    builder.relative_run_dir = Path("pes_sheet/h-MoS2/30x_30y/K0")

    # Stand in for the geometry/potential state build() would populate.
    class _PM:  # minimal PotentialManager stand-in (unused by write_inputs)
        pass
    builder.pm = _PM()
    builder.structure_paths = {"sheet": tmp_path / "h-MoS2_2.lmp"}
    builder.sheet_dims = {"xlo": 0.0, "xhi": 30.0, "ylo": 0.0, "yhi": 30.0, "zhi": 20.0}
    builder.sheet_unit_cell = {"xlo": 0.0, "xhi": 3.0, "ylo": 0.0, "yhi": 5.0}
    builder.lat_c = 6.5

    builder.write_inputs()

    slide = (out / "lammps" / "slide.in").read_text(encoding="utf-8")
    assert "interlayer PES scan" in slide
    assert "displace_atoms  layer_2 move" in slide
    assert "setforce 0.0 0.0 0.0" in slide            # frozen bottom
    assert "Nx     equal 12" in slide
    assert "$(v_dx-v_prev_dx)" in slide                # snapshotted delta, not lazy
    assert (out / "lammps" / "slide_refine.in").exists()   # convergence grid


# --- PESTipSimulation ------------------------------------------------------

def test_pes_tip_uses_fixed_contact_gap():
    """Tip scan is a fixed firm-contact scan (no repulsive-load setpoint)."""
    from src.builders.pes_scan import PES_TIP_CONTACT_GAP
    assert PES_TIP_CONTACT_GAP > 0


def test_pes_tip_z_relax_from_config(tmp_path: Path):
    cfg = _afm_cfg(tmp_path, force=[2.0])
    cfg.pes.z_relax = False
    builder = PESTipSimulation(cfg, output_dir=str(tmp_path / "o"))
    assert builder._z_relax() is False


def test_pes_tip_hpc_job_name(tmp_path: Path):
    builder = PESTipSimulation(_afm_cfg(tmp_path, force=[2.0]), output_dir=str(tmp_path / "o"))
    assert builder._get_hpc_job_name() == "pes_tip_h-MoS2"
