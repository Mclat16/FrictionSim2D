import os
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError
from src.core.config import (
    AFMSimulationConfig, 
    SheetOnSheetSimulationConfig,
    TipConfig, 
    SheetConfig, 
    SubstrateConfig,
    GeneralConfig,
    GlobalSettings,
    parse_config,
    load_settings,
    settings_origin,
)

# Sample data
VALID_TIP = {
    'mat': 'Si', 'pot_type': 'tersoff', 'pot_path': 'potentials/Si.tersoff',
    'cif_path': 'Si.cif', 'r': 10.0, 'amorph': 'c', 'cspring': 10.0, 's': 10.0
}
VALID_SHEET = {
    'mat': 'MoS2', 'pot_type': 'sw', 'pot_path': 'potentials/MoS2.sw',
    'cif_path': 'MoS2.cif', 'x': 50.0, 'y': 50.0, 'layers': [1]
}
VALID_SUB = {
    'mat': 'Si', 'pot_type': 'tersoff', 'pot_path': 'potentials/Si.tersoff',
    'cif_path': 'Si.cif', 'thickness': 10.0, 'amorph': 'a'
}
VALID_GENERAL = {'temp': 300.0}

def test_load_default_settings():
    """Ensure default settings load correctly from package resources."""
    settings = load_settings()
    assert isinstance(settings, GlobalSettings)
    assert settings.geometry.rigid_tip is not None
    assert settings.thermostat.type in ['langevin', 'nose-hoover']

def test_valid_config_creation(tmp_path, monkeypatch):
    """Test creating a full configuration object."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    
    config = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip=VALID_TIP,
        sub=VALID_SUB,
        **{'2D': VALID_SHEET},
        settings=settings
    )
    assert config.tip.r == 10.0
    assert config.sheet.mat == 'MoS2'

def test_validation_error(tmp_path, monkeypatch):
    """Ensure invalid types raise errors."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    invalid_tip = VALID_TIP.copy()
    invalid_tip['r'] = "not_a_number" # Should fail
    
    with pytest.raises(ValidationError):
        AFMSimulationConfig(
            general=VALID_GENERAL,
            tip=invalid_tip,
            sub=VALID_SUB,
            **{'2D': VALID_SHEET},
            settings=settings
        )

def test_parse_config_ini(temp_dir):
    """Test parsing an INI file."""
    ini_content = """
[general]
temp = 300
[tip]
r = 10.0
"""
    ini_file = temp_dir / "test.ini"
    ini_file.write_text(ini_content)
    
    data = parse_config(ini_file)
    assert data['general']['temp'] == 300 # int conversion check
    assert data['tip']['r'] == 10.0 # float conversion check

def test_parse_config_yaml(temp_dir):
    """Test parsing a YAML file."""
    yaml_content = {'general': {'temp': 300}}
    yaml_file = temp_dir / "test.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f)
        
    data = parse_config(yaml_file)
    assert data['general']['temp'] == 300


def test_sheetonsheet_config_accepts_explicit_scan_angle_list(tmp_path, monkeypatch):
    """Sheet-on-sheet config should accept explicit numeric angle lists."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()

    config = SheetOnSheetSimulationConfig(
        general={
            "temp": 300.0,
            "pressure": [0.1, 1.0],
            "scan_angle": [0, 45, 90],
            "scan_speed": 1.0,
            "scan_angle_force": [0.1, 1.0],
        },
        **{
            '2D': {
                'mat': 'MoS2',
                'pot_type': 'sw',
                'pot_path': str(dummy_pot),
                'cif_path': str(dummy_cif),
                'x': 50.0,
                'y': 50.0,
                'layers': [3],
            }
        },
        settings=settings,
    )

    assert config.general.scan_angle == [0.0, 45.0, 90.0]
    assert config.general.scan_angle_force == [0.1, 1.0]


def test_general_config_rejects_invalid_scan_angle_string() -> None:
    """scan_angle now only accepts numeric values."""
    with pytest.raises(ValidationError):
        GeneralConfig(scan_angle=[0, 90, 'all'])


def test_tip_config_allows_explicit_tip_coordinates_with_offsets(tmp_path, monkeypatch) -> None:
    """Tip placement fields should belong to [tip] and allow explicit+offset inputs."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    tip = dict(VALID_TIP)
    tip.update({"tip_x": 10.0, "tip_y": 5.0, "tip_x_offset": 1.0, "tip_y_offset": -0.5})

    cfg = AFMSimulationConfig(
        general=VALID_GENERAL,
        tip=tip,
        sub=VALID_SUB,
        **{'2D': VALID_SHEET},
        settings=settings,
    )
    assert cfg.tip.tip_x == 10.0
    assert cfg.tip.tip_y == 5.0
    assert cfg.tip.tip_x_offset == 1.0
    assert cfg.tip.tip_y_offset == -0.5


def test_finite_sheet_requires_substrate_xy(tmp_path, monkeypatch):
    """Finite-sheet AFM should require explicit substrate x/y in [sub]."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    with pytest.raises(ValidationError, match="requires substrate x and y"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip=VALID_TIP,
            sub={**VALID_SUB, "amorph": "c"},
            **{'2D': VALID_SHEET},
            settings=settings,
        )


def test_finite_sheet_rejects_substrate_smaller_than_sheet(tmp_path, monkeypatch):
    """Finite-sheet AFM should reject substrate sizes smaller than the sheet."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    bad_sub = dict(VALID_SUB)
    bad_sub.update({"x": 40.0, "y": 60.0, "amorph": "c"})
    with pytest.raises(ValidationError, match="cannot be smaller than sheet x"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip=VALID_TIP,
            sub=bad_sub,
            **{'2D': VALID_SHEET},
            settings=settings,
        )


def test_finite_sheet_rejects_margin_not_above_lj_cutoff(tmp_path, monkeypatch):
    """Finite-sheet AFM should enforce strict substrate-sheet delta > LJ cutoff."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings().model_copy(deep=True)
    settings.potential.lj_cutoff = 11.0
    bad_sub = dict(VALID_SUB)
    bad_sub.update({"x": 61.0, "y": 70.0, "amorph": "c"})
    # sheet x is 50, so x-delta is exactly 11.0 -> invalid (must be > cutoff)
    with pytest.raises(ValidationError, match="greater than LJ cutoff"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": True},
            tip=VALID_TIP,
            sub=bad_sub,
            **{'2D': VALID_SHEET},
            settings=settings,
        )


def test_non_finite_sheet_rejects_crystalline_substrate(tmp_path, monkeypatch):
    """Non-finite-sheet AFM should allow only amorphous substrates."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    with pytest.raises(ValidationError, match="requires an amorphous substrate"):
        AFMSimulationConfig(
            general={"temp": 300.0, "finite_sheet": False},
            tip=VALID_TIP,
            sub={**VALID_SUB, "amorph": "c"},
            **{'2D': VALID_SHEET},
            settings=settings,
        )


# ---------------------------------------------------------------------------
# Settings precedence tests
# ---------------------------------------------------------------------------

def _write_settings(path: Path, timestep: float) -> None:
    """Write a minimal settings.yaml with a distinctive timestep value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump({'simulation': {'timestep': timestep}}), encoding='utf-8')


def test_defaults_used_when_no_file_exists(tmp_path, monkeypatch):
    """load_settings returns hardcoded defaults when no settings file exists."""
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: tmp_path / 'nonexistent.yaml')
    settings = load_settings()
    assert isinstance(settings, GlobalSettings)
    # Default timestep is 0.001 as defined in SimulationSettings
    assert settings.simulation.timestep == 0.001


def test_global_settings_used_when_no_explicit_file(tmp_path, monkeypatch):
    """load_settings picks up the global settings file when no explicit file is given."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

    settings = load_settings()
    assert settings.simulation.timestep == 0.005


def test_explicit_file_takes_precedence_over_global(tmp_path, monkeypatch):
    """Explicit settings_file argument overrides the global settings file."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    explicit_path = tmp_path / 'explicit' / 'settings.yaml'
    _write_settings(explicit_path, timestep=0.002)

    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

    settings = load_settings(settings_file=explicit_path)
    assert settings.simulation.timestep == 0.002


def test_env_var_is_not_honoured(tmp_path, monkeypatch):
    """FRICTIONSIM2D_SETTINGS_FILE env var is no longer part of the precedence chain."""
    env_path = tmp_path / 'env' / 'settings.yaml'
    _write_settings(env_path, timestep=0.007)

    # Set the env var but point the global path to a non-existent file
    monkeypatch.setenv('FRICTIONSIM2D_SETTINGS_FILE', str(env_path))
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: tmp_path / 'nonexistent.yaml')

    # Should fall back to hardcoded defaults, NOT pick up the env var file
    settings = load_settings()
    assert settings.simulation.timestep == 0.001


def test_settings_origin_none_when_no_file(tmp_path, monkeypatch):
    """settings_origin returns None when no settings file exists."""
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: tmp_path / 'nonexistent.yaml')
    assert settings_origin() is None


def test_settings_origin_returns_global_path(tmp_path, monkeypatch):
    """settings_origin returns the global path when it exists and no explicit file given."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)
    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

    assert settings_origin() == global_path


def test_settings_origin_returns_explicit_path(tmp_path, monkeypatch):
    """settings_origin returns the explicit path when one is provided."""
    global_path = tmp_path / 'global' / 'settings.yaml'
    _write_settings(global_path, timestep=0.005)

    explicit_path = tmp_path / 'explicit' / 'settings.yaml'
    _write_settings(explicit_path, timestep=0.002)

    monkeypatch.setattr('src.core.config._global_settings_path', lambda: global_path)

    assert settings_origin(settings_file=explicit_path) == explicit_path


def test_lattice_match_section_is_parsed(tmp_path):
    from src.core.config import LatticeMatchConfig
    lm = LatticeMatchConfig()
    assert lm.method == "zur_mcgill"
    assert lm.strain_tol == 0.02
    assert lm.max_supercell == 10
    assert lm.area_tol == 0.10
    # out-of-range values are rejected
    import pytest
    with pytest.raises(Exception):
        LatticeMatchConfig(strain_tol=2.0)   # > 1.0
    with pytest.raises(Exception):
        LatticeMatchConfig(max_supercell=0)  # < 1


def test_lattice_match_area_tol_is_bounded():
    """area_tol must be in [0, 1]; out-of-range values are rejected."""
    from src.core.config import LatticeMatchConfig
    with pytest.raises(ValidationError):
        LatticeMatchConfig(area_tol=2.0)


def test_two_material_sections_fold_into_sheets_list(tmp_path, monkeypatch):
    """[2D-1]/[2D-2] sections fold into cfg.sheets in numeric order."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "alternating"},
        "2D-1": {"mat": "MoS2", "cif_path": str(dummy_cif), "pot_path": str(dummy_pot),
                 "pot_type": "sw", "x": 30, "y": 30, "layers": [1], "stack_type": "AB"},
        "2D-2": {"mat": "WS2", "cif_path": str(dummy_cif), "pot_path": str(dummy_pot),
                 "pot_type": "sw", "x": 30, "y": 30, "layers": [1], "stack_type": "AB"},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=settings)
    assert [s.mat for s in cfg.sheets] == ["MoS2", "WS2"]
    assert cfg.general.hetero_stacking == "alternating"
    # Backward-compatible single-sheet accessor still resolves to the first material.
    assert cfg.sheet.mat == "MoS2"


def test_single_2d_section_still_builds_one_element_sheets_list(tmp_path, monkeypatch):
    """A lone [2D] section (no [2D-N]) still builds and yields len(sheets) == 1."""
    dummy_pot = tmp_path / "dummy.sw"
    dummy_cif = tmp_path / "dummy.cif"
    dummy_pot.write_text("# dummy potential\n", encoding="utf-8")
    dummy_cif.write_text("# dummy cif\n", encoding="utf-8")
    monkeypatch.setattr("src.core.config.get_potential_path", lambda _v: dummy_pot)
    monkeypatch.setattr("src.core.config.get_material_path", lambda _v: dummy_cif)

    settings = load_settings()
    cfg = SheetOnSheetSimulationConfig(
        general=VALID_GENERAL,
        **{'2D': VALID_SHEET},
        settings=settings,
    )
    assert len(cfg.sheets) == 1
    assert cfg.sheets[0].mat == "MoS2"
    assert cfg.sheet.mat == "MoS2"