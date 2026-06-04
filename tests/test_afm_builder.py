"""Tests for AFMSimulation builder layer sweep behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.builders.afm import AFMSimulation
from src.core.config import AFMSimulationConfig, load_settings
from src.core.utils import normalize_potential_type


def _make_afm_config(tmp_path: Path, layers: list[int]) -> AFMSimulationConfig:
    """Create a minimal valid AFM config for builder tests."""
    pot_path = tmp_path / "dummy.sw"
    cif_path = tmp_path / "dummy.cif"
    pot_path.write_text("# dummy potential", encoding="utf-8")
    cif_path.write_text("# dummy cif", encoding="utf-8")

    settings = load_settings().model_dump()

    return AFMSimulationConfig(
        **{
            "general": {
                "temp": 300.0,
                "force": 10.0,
                "scan_speed": 2.0,
            },
            "tip": {
                "mat": "Si",
                "pot_type": "sw",
                "pot_path": str(pot_path),
                "cif_path": str(cif_path),
                "r": 20.0,
                "dspring": 0.1,
                "s": 1.0,
            },
            "sub": {
                "mat": "Si",
                "pot_type": "sw",
                "pot_path": str(pot_path),
                "cif_path": str(cif_path),
                "thickness": 10.0,
                "amorph": "a",
            },
            "2D": {
                "mat": "h-MoS2",
                "pot_type": "sw",
                "pot_path": str(pot_path),
                "cif_path": str(cif_path),
                "x": 20.0,
                "y": 20.0,
                "layers": layers,
            },
            "lj_override": {},
            "settings": settings,
        }
    )


def test_afm_builder_supports_layer_sweep(tmp_path: Path) -> None:
    """AFM builder should accept multiple layer values (unlike sheet-on-sheet)."""
    config = _make_afm_config(tmp_path, layers=[1, 2, 3])

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    # Should not raise; multiple layers are intentional for AFM
    assert config.sheet.layers == [1, 2, 3]


def test_afm_builder_layer_specific_paths(tmp_path: Path, monkeypatch) -> None:
    """AFM should create L{n} directories for each layer count."""
    # Mock out actual file operations to avoid needing real files
    monkeypatch.setattr(
        "src.builders.components.build_sheet",
        lambda *args, **kwargs: (tmp_path / "sheet.lmp", {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zhi": 15}, 3.0)
    )
    monkeypatch.setattr(
        "src.builders.components.build_tip",
        lambda *args, **kwargs: (tmp_path / "tip.lmp", 20.0)
    )
    monkeypatch.setattr(
        "src.builders.components.build_substrate",
        lambda *args, **kwargs: tmp_path / "sub.lmp"
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._create_directories",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._init_provenance",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation.write_inputs",
        lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        lambda *args, **kwargs: None
    )

    config = _make_afm_config(tmp_path, layers=[2, 3])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    # Verify internal layer storage before build
    assert builder.config.sheet.layers == [2, 3]


def test_normalize_potential_type_consistency() -> None:
    """Potential type normalization should be consistent across builders."""
    test_cases = [
        ("SW", "sw"),
        ("Tersoff", "tersoff"),
        ("ReaxFF", "reaxff"),
        ("REAX/C", "reax/c"),
        ("  sw  ", "sw"),
    ]

    for input_val, expected in test_cases:
        assert normalize_potential_type(input_val) == expected


def test_afm_build_orchestrates_per_layer_steps(tmp_path: Path, monkeypatch) -> None:
    """Build should execute per-layer orchestration for all requested layers."""
    config = _make_afm_config(tmp_path, layers=[2, 4])
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    called = {
        "build_sheet_layers": [],
        "create_dirs": [],
        "build_components": 0,
        "generate_potentials": [],
        "z_positions": [],
        "write_inputs": [],
        "hpc": 0,
    }

    def mock_build_sheet(*_args, **kwargs):
        called["build_sheet_layers"].append(kwargs["n_layers_override"])
        return (
            tmp_path / f"sheet_L{kwargs['n_layers_override']}.lmp",
            {"xlo": 0, "xhi": 100, "ylo": 0, "yhi": 100, "zlo": 0, "zhi": 20},
            3.0,
        )

    monkeypatch.setattr("src.builders.components.build_sheet", mock_build_sheet)
    monkeypatch.setattr("src.builders.afm.AFMSimulation._init_provenance", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._get_substrate_box_dims",
        lambda _self: {"xlo": 0.0, "xhi": 100.0, "ylo": 0.0, "yhi": 100.0, "zlo": -5.0, "zhi": 0.0},
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._create_directories",
        lambda _self, out_dir=None: called["create_dirs"].append(out_dir),
    )

    def mock_build_components(_self, _build_dir):
        called["build_components"] += 1
        return tmp_path / "tip.lmp", 20.0, tmp_path / "sub.lmp"

    monkeypatch.setattr("src.builders.afm.AFMSimulation._build_components", mock_build_components)
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_potentials",
        lambda _self, n: called["generate_potentials"].append(n) or object(),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._calculate_z_positions",
        lambda _self, n, _r: called["z_positions"].append(n),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation.write_inputs",
        lambda _self, n: called["write_inputs"].append(n),
    )
    monkeypatch.setattr(
        "src.builders.afm.AFMSimulation._generate_hpc_scripts",
        lambda *_a, **_k: called.__setitem__("hpc", called["hpc"] + 1),
    )

    builder.build()

    assert called["build_sheet_layers"] == [2, 4]
    assert called["build_components"] == 1
    assert called["generate_potentials"] == [2, 4]
    assert called["z_positions"] == [2, 4]
    assert called["write_inputs"] == [2, 4]
    assert called["hpc"] == 1
    assert sorted(builder.output_dir_layer.keys()) == [2, 4]


def _prepare_builder_for_slide_write(builder: AFMSimulation, tmp_path: Path) -> Path:
    """Seed builder state so write_inputs can render slide.in in isolation."""
    output_dir = tmp_path / "out"
    layer_dir = output_dir / "L1"

    builder.output_dir = output_dir
    builder.relative_run_dir = Path("out")
    builder.output_dir_layer[1] = layer_dir
    builder.relative_run_dir_layer[1] = Path("out") / "L1"

    builder.sheet_paths[1] = tmp_path / "sheet.lmp"
    builder.tip_path = tmp_path / "tip.lmp"
    builder.sub_path = tmp_path / "sub.lmp"

    builder.z_positions[1] = {"sub": 0.0, "sheet": 16.0, "tip": 40.0}
    builder.groups[1] = {
        "sub_types": "1",
        "tip_types": "2",
        "sheet_types": "3",
    }
    builder.pm[1] = SimpleNamespace(types=[1, 2, 3])
    builder.lat_c = 6.0
    sheet_dims = {
        "xlo": 0.0,
        "xhi": 100.0,
        "ylo": 0.0,
        "yhi": 100.0,
        "zlo": 0.0,
        "zhi": 12.0,
    }
    builder.sheet_dims = dict(sheet_dims)
    builder.box_dims = dict(sheet_dims)
    builder.sheet_offset = {"x": 0.0, "y": 0.0}

    for path in (builder.sheet_paths[1], builder.tip_path, builder.sub_path):
        path.write_text("# dummy\n", encoding="utf-8")

    builder.write_inputs(1)
    return layer_dir / "lammps" / "slide.in"


def test_afm_finite_sheet_computes_box_and_center_offsets(tmp_path: Path) -> None:
    """Finite-sheet mode should use substrate x/y as box and center the sheet."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.finite_sheet = True
    config.sub.x = 160.0
    config.sub.y = 140.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    builder.sheet_dims = {
        "xlo": 0.0,
        "xhi": 100.0,
        "ylo": 0.0,
        "yhi": 100.0,
        "zlo": 0.0,
        "zhi": 12.0,
    }
    builder.box_dims = {
        "xlo": 0.0,
        "xhi": 160.0,
        "ylo": 0.0,
        "yhi": 140.0,
        "zlo": -5.0,
        "zhi": 0.0,
    }

    offsets = builder._compute_sheet_offset()

    assert builder.box_dims["xhi"] == 160.0
    assert builder.box_dims["yhi"] == 140.0
    assert offsets == {"x": 30.0, "y": 20.0}


def test_afm_finite_sheet_validates_final_substrate_dims(tmp_path: Path) -> None:
    """Final substrate dims from .lmp should be validated, not only requested config dims."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.finite_sheet = True
    config.sub.x = 160.0
    config.sub.y = 140.0
    config.settings.potential.lj_cutoff = 11.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    builder.sheet_dims = {
        "xlo": 0.0,
        "xhi": 100.0,
        "ylo": 0.0,
        "yhi": 100.0,
        "zlo": 0.0,
        "zhi": 12.0,
    }
    # Simulate post-build substrate dimensions read from sub.lmp.
    builder.box_dims = {
        "xlo": 0.0,
        "xhi": 110.0,
        "ylo": 0.0,
        "yhi": 110.0,
        "zlo": -5.0,
        "zhi": 0.0,
    }

    with pytest.raises(ValueError, match="Final built substrate x-span delta must be greater than LJ cutoff"):
        builder._validate_final_substrate_dims()


def test_afm_scan_angle_list_with_force_gate(tmp_path: Path) -> None:
    """Explicit angle lists should be used directly (no interval expansion)."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0, 20.0, 30.0]
    config.general.scan_angle = [2.0, 10.0, 4.0]
    config.general.scan_angle_force = 30.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert "variable        a index 2.0 10.0 4.0" in slide_script
    assert "variable        scan_angle_force equal 30.0" in slide_script
    assert "abs(v_find-v_scan_angle_force) < 1.0e-12" in slide_script


def test_afm_scan_angle_explicit_list_with_force_gate(tmp_path: Path) -> None:
    """Explicit AFM angle lists should be supported with a separate force gate."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0, 20.0, 30.0]
    config.general.scan_angle = [0.0, 2.0, 6.0, 10.0]
    config.general.scan_angle_force = 30.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert "variable        a index 0.0 2.0 6.0 10.0" in slide_script
    assert "variable        scan_angle_force equal 30.0" in slide_script
    assert "abs(v_find-v_scan_angle_force) < 1.0e-12" in slide_script


def test_afm_scan_angle_force_accepts_multiple_targets(tmp_path: Path) -> None:
    """scan_angle_force should support lists of target force values."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.general.force = [2.0, 5.0, 10.0, 20.0, 30.0]
    config.general.scan_angle = [0.0, 45.0, 90.0]
    config.general.scan_angle_force = [10.0, 30.0]

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    slide_path = _prepare_builder_for_slide_write(builder, tmp_path)
    slide_script = slide_path.read_text(encoding="utf-8")

    assert "variable        scan_angle_force index 10.0 30.0" in slide_script
    assert "abs(v_find-v_scan_angle_force) < 1.0e-12" in slide_script
    assert "then \"next scan_angle_force\"" in slide_script


def test_afm_explicit_tip_x_ignores_tip_x_offset(tmp_path: Path) -> None:
    """Explicit tip_x should be used exactly; tip_x_offset should not be applied."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.tip.tip_x = 33.0
    config.tip.tip_x_offset = 7.5

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    layer_dir = _prepare_builder_for_slide_write(builder, tmp_path).parent
    system_script = (layer_dir / "system.in").read_text(encoding="utf-8")

    assert "shift 33.0" in system_script


def test_afm_explicit_tip_y_ignores_tip_y_offset(tmp_path: Path) -> None:
    """Explicit tip_y should be used exactly; tip_y_offset should not be applied."""
    config = _make_afm_config(tmp_path, layers=[1])
    config.tip.tip_y = 41.0
    config.tip.tip_y_offset = -3.0

    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))
    layer_dir = _prepare_builder_for_slide_write(builder, tmp_path).parent
    system_script = (layer_dir / "system.in").read_text(encoding="utf-8")

    assert " 41.0 " in system_script
