"""Tests for run-level sweep expansion and orchestration utilities."""

from pathlib import Path

import pytest

from src.core.config import load_settings as real_load_settings
from src.core.run import (
    _validate_runtime_sweep_ordering,
    _build_hpc_manifest_entries,
    collect_hpc_simulation_paths,
    expand_config_sweeps,
    run_simulations,
)


def test_expand_config_sweeps_does_not_expand_2d_layers() -> None:
    """Layer lists are not expanded in run-level sweeps."""
    base_config = {
        "general": {
            "temp": 300.0,
            "pressure": [1.0, 2.0],
        },
        "2D": {
            "mat": "h-MoS2",
            "pot_type": "sw",
            "pot_path": "dummy.sw",
            "cif_path": "dummy.cif",
            "x": 20.0,
            "y": 20.0,
            "layers": [3, 4],
        },
    }

    expanded = expand_config_sweeps(base_config)

    assert len(expanded) == 1
    assert expanded[0]["2D"]["layers"] == [3, 4]


def test_runtime_validation_rejects_unsorted_pressure() -> None:
    """Runtime validation should reject non-ascending pressure lists."""
    with pytest.raises(ValueError, match="pressure must be in ascending order"):
        _validate_runtime_sweep_ordering({"pressure": [1.0, 0.1]})


def test_runtime_validation_rejects_unsorted_scan_angle_force() -> None:
    """Runtime validation should reject non-ascending selector lists."""
    with pytest.raises(ValueError, match="scan_angle_force must be in ascending order"):
        _validate_runtime_sweep_ordering(
            {"pressure": [0.1, 1.0, 10.0], "scan_angle_force": [10.0, 0.1]}
        )


def test_runtime_validation_rejects_selector_not_in_target_order() -> None:
    """Selector values must follow pressure/force order as a subsequence."""
    with pytest.raises(ValueError, match="scan_angle_force must follow the same order as pressure values"):
        _validate_runtime_sweep_ordering(
            {"pressure": [0.1, 1.0, 10.0], "scan_angle_force": [0.1, 5.0]}
        )


def test_expand_config_sweeps_expands_general_non_lammps_lists() -> None:
    """General non-LAMMPS-loop list parameters should expand combinations."""
    base_config = {
        "general": {
            "temp": [300.0, 350.0],
            "scan_speed": [1.0, 2.0],
        },
        "2D": {
            "mat": "h-MoS2",
            "pot_type": "sw",
            "pot_path": "dummy.sw",
            "cif_path": "dummy.cif",
            "x": 20.0,
            "y": 20.0,
            "layers": [3],
        },
    }

    expanded = expand_config_sweeps(base_config)

    # Only temp expands; scan_speed is intentionally handled at LAMMPS level.
    assert len(expanded) == 2
    temps = sorted(conf["general"]["temp"] for conf in expanded)
    assert temps == [300.0, 350.0]
    for conf in expanded:
        assert conf["general"]["scan_speed"] == [1.0, 2.0]


def test_collect_hpc_simulation_paths_detects_valid_inputs(tmp_path: Path) -> None:
    """Collect only simulation dirs with recognized LAMMPS input scripts."""
    (tmp_path / "simA" / "lammps").mkdir(parents=True)
    (tmp_path / "simA" / "lammps" / "system.in").write_text("", encoding="utf-8")

    (tmp_path / "simB" / "lammps").mkdir(parents=True)
    (tmp_path / "simB" / "lammps" / "slide_custom.in").write_text("", encoding="utf-8")

    (tmp_path / "simC" / "lammps").mkdir(parents=True)
    (tmp_path / "simC" / "lammps" / "notes.txt").write_text("", encoding="utf-8")

    paths = collect_hpc_simulation_paths(tmp_path)

    assert paths == ["simA", "simB"]


def test_collect_hpc_simulation_paths_orders_layers_numerically(tmp_path: Path) -> None:
    """Layer-like paths should use numeric order (L2 before L10)."""
    for layer in ("L1", "L2", "L10"):
        (tmp_path / layer / "lammps").mkdir(parents=True)
        (tmp_path / layer / "lammps" / "slide.in").write_text("", encoding="utf-8")

    paths = collect_hpc_simulation_paths(tmp_path)

    assert paths == ["L1", "L2", "L10"]


def test_collect_hpc_simulation_paths_accepts_any_in_when_scripts_empty(tmp_path: Path) -> None:
    """An empty script filter should include dirs containing any .in file."""
    (tmp_path / "simA" / "lammps").mkdir(parents=True)
    (tmp_path / "simA" / "lammps" / "custom_input.in").write_text("", encoding="utf-8")

    (tmp_path / "simB" / "lammps").mkdir(parents=True)
    (tmp_path / "simB" / "lammps" / "notes.txt").write_text("", encoding="utf-8")

    paths = collect_hpc_simulation_paths(tmp_path, lammps_scripts=[])

    assert paths == ["simA"]


def test_build_hpc_manifest_entries_prefers_slide_when_available(tmp_path: Path) -> None:
    """Manifest should include slide scripts when present, else system scripts."""
    (tmp_path / "simA" / "lammps").mkdir(parents=True)
    (tmp_path / "simA" / "lammps" / "system.in").write_text("", encoding="utf-8")
    (tmp_path / "simA" / "lammps" / "slide_p1.in").write_text("", encoding="utf-8")

    (tmp_path / "simB" / "lammps").mkdir(parents=True)
    (tmp_path / "simB" / "lammps" / "system_init.in").write_text("", encoding="utf-8")

    entries, resolved = _build_hpc_manifest_entries(
        tmp_path,
        ["simA", "simB"],
        ["system.in", "slide.in"],
    )

    assert entries == [
        "simA/lammps/slide_p1.in",
        "simB/lammps/system_init.in",
    ]
    assert resolved == ["__MANIFEST_PATH__"]


def test_build_hpc_manifest_entries_falls_back_when_no_scripts(tmp_path: Path) -> None:
    """When no scripts are discovered, preserve original simulation paths/scripts."""
    (tmp_path / "simA").mkdir(parents=True)

    entries, resolved = _build_hpc_manifest_entries(
        tmp_path,
        ["simA"],
        ["slide.in"],
    )

    assert entries == ["simA"]
    assert resolved == ["slide.in"]


def test_run_simulations_dispatches_afm_and_generates_hpc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """run_simulations should instantiate AFM builder and call HPC generation."""
    config_file = tmp_path / "config.ini"
    config_file.write_text("[dummy]\n", encoding="utf-8")

    base_dict = {
        "general": {"temp": 300.0},
        "2D": {"mat": "MoS2", "x": 20.0, "y": 20.0},
        "tip": {"mat": "Si", "amorph": "c", "r": 25},
        "sub": {"mat": "Si", "amorph": "a"},
    }

    settings = real_load_settings()
    settings.hpc.lammps_scripts = ["slide.in"]

    monkeypatch.setattr("src.core.run.parse_config", lambda _p: base_dict)
    monkeypatch.setattr("src.core.run.load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr("src.core.run.expand_config_sweeps", lambda _c: [dict(base_dict)])

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.sub = kwargs.get('sub')

        def model_dump_json(self, indent=2):
            return "{}"

    calls = {"build": 0, "base": 0, "hpc": 0}
    seen_output_dirs: list[Path] = []

    class FakeAFMBuilder:
        def __init__(self, _config, _output_dir, config_path=None):
            seen_output_dirs.append(Path(_output_dir))
            self.config_path = config_path

        def set_base_output_dir(self, _root):
            calls["base"] += 1

        def build(self):
            calls["build"] += 1

    monkeypatch.setattr("src.core.run.AFMSimulationConfig", FakeConfig)
    monkeypatch.setattr("src.core.run.AFMSimulation", FakeAFMBuilder)
    monkeypatch.setattr(
        "src.core.run.generate_hpc_scripts_for_root",
        lambda _root, _settings: calls.__setitem__("hpc", calls["hpc"] + 1),
    )

    created, root, _configs, returned_settings = run_simulations(
        config_file=str(config_file),
        model="afm",
        output_root=tmp_path,
        generate_hpc=True,
        simulation_root_name="sim_run",
    )

    assert len(created) == 1
    assert root == tmp_path / "sim_run"
    assert calls == {"build": 1, "base": 1, "hpc": 1}
    assert seen_output_dirs[0].as_posix().endswith("afm/MoS2/20x_20y/sub_aSi_tip_Si_r25/K300")
    assert returned_settings.hpc.lammps_scripts == ["system.in", "slide.in"]


def test_run_simulations_sheetonsheet_adjusts_default_scripts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Sheet-on-sheet mode should default to slide-only script set."""
    config_file = tmp_path / "config.ini"
    config_file.write_text("[dummy]\n", encoding="utf-8")

    base_dict = {
        "general": {"temp": 300.0},
        "2D": {"mat": "MoS2", "x": 20.0, "y": 20.0},
    }

    settings = real_load_settings()
    settings.hpc.lammps_scripts = ["system.in", "slide.in"]

    monkeypatch.setattr("src.core.run.parse_config", lambda _p: base_dict)
    monkeypatch.setattr("src.core.run.load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr("src.core.run.expand_config_sweeps", lambda _c: [dict(base_dict)])

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def model_dump_json(self, indent=2):
            return "{}"

    class FakeSheetBuilder:
        def __init__(self, _config, _output_dir, config_path=None):
            self.config_path = config_path

        def set_base_output_dir(self, _root):
            pass

        def build(self):
            pass

    monkeypatch.setattr("src.core.run.SheetOnSheetSimulationConfig", FakeConfig)
    monkeypatch.setattr("src.core.run.SheetOnSheetSimulation", FakeSheetBuilder)

    _created, _root, _configs, returned_settings = run_simulations(
        config_file=str(config_file),
        model="sheetonsheet",
        output_root=tmp_path,
        simulation_root_name="sim_run",
    )

    assert returned_settings.hpc.lammps_scripts == ["slide.in"]


def test_run_simulations_continues_after_build_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A failed build should be logged and subsequent configs should still run."""
    config_file = tmp_path / "config.ini"
    config_file.write_text("[dummy]\n", encoding="utf-8")

    base_dict = {
        "general": {"temp": 300.0},
        "2D": {"mat": "MoS2", "x": 20.0, "y": 20.0},
        "tip": {"mat": "Si", "amorph": "c", "r": 25},
        "sub": {"mat": "Si", "amorph": "a"},
    }

    settings = real_load_settings()
    monkeypatch.setattr("src.core.run.parse_config", lambda _p: base_dict)
    monkeypatch.setattr("src.core.run.load_settings", lambda **_kwargs: settings)
    monkeypatch.setattr(
        "src.core.run.expand_config_sweeps",
        lambda _c: [dict(base_dict), dict(base_dict)],
    )

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.sub = kwargs.get('sub')

        def model_dump_json(self, indent=2):
            return "{}"

    call_counter = {"builds": 0}

    class FlakyAFMBuilder:
        def __init__(self, _config, _output_dir, config_path=None):
            self.config_path = config_path

        def set_base_output_dir(self, _root):
            pass

        def build(self):
            call_counter["builds"] += 1
            if call_counter["builds"] == 1:
                raise RuntimeError("boom")

    monkeypatch.setattr("src.core.run.AFMSimulationConfig", FakeConfig)
    monkeypatch.setattr("src.core.run.AFMSimulation", FlakyAFMBuilder)

    created, _root, _configs, _returned_settings = run_simulations(
        config_file=str(config_file),
        model="afm",
        output_root=tmp_path,
        simulation_root_name="sim_run",
    )

    assert call_counter["builds"] == 2
    assert len(created) == 1


def test_run_simulations_raises_for_missing_config_file(tmp_path: Path) -> None:
    """Missing config file should raise immediately."""
    with pytest.raises(FileNotFoundError):
        run_simulations(
            config_file=str(tmp_path / "missing.ini"),
            model="afm",
            output_root=tmp_path,
        )


def test_pes_tip_no_substrate_selects_freestanding_builder():
    from src.core.run import _select_afm_builder_cls
    from src.builders.afm_freestanding import FreestandingPESTipSimulation
    from src.builders.pes_scan import PESTipSimulation

    class _Cfg:  # minimal stand-in with the only attribute the selector reads
        sub = None
    assert _select_afm_builder_cls('pes_tip', _Cfg()) is FreestandingPESTipSimulation

    class _CfgSub:
        sub = object()
    assert _select_afm_builder_cls('pes_tip', _CfgSub()) is PESTipSimulation


def test_afm_no_substrate_raises():
    from src.core.run import _select_afm_builder_cls
    import pytest
    class _Cfg:
        sub = None
    with pytest.raises(ValueError, match="substrate-free"):
        _select_afm_builder_cls('afm', _Cfg())
