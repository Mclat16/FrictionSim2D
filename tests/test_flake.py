"""Tests for the shaped-flake AFM use case.

Covers the geometry predicates, flake cutting (build_flake), corner detection +
retyping (apply_flake_corner_types), the FlakeConfig model, and the LAMMPS
template wiring (corner restraint present during indentation, absent in slide).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.builders import components
from src.builders.afm import AFMSimulation
from src.core.config import AFMSimulationConfig, FlakeConfig, load_settings


# ---------------------------------------------------------------------------
# Geometry predicates
# ---------------------------------------------------------------------------

def test_geometry_predicates_membership():
    # Center is always inside; far-away points are outside.
    assert components._in_square(0.0, 0.0, 10.0)
    assert not components._in_square(100.0, 0.0, 10.0)
    assert components._in_equilateral_triangle(0.0, 0.0, 10.0)
    assert not components._in_equilateral_triangle(0.0, 100.0, 10.0)
    assert components._in_hexagon(0.0, 0.0, 10.0)
    assert not components._in_hexagon(0.0, 100.0, 10.0)


def test_square_rotation_is_symmetric():
    # A point on the +x axis just inside an axis-aligned square edge stays inside
    # after a 90° rotation (square maps onto itself).
    half = 5.0
    assert components._in_square(half - 0.1, 0.0, 2 * half, (0.0, 0.0), 0.0)
    assert components._in_square(half - 0.1, 0.0, 2 * half, (0.0, 0.0), 90.0)


# ---------------------------------------------------------------------------
# Synthetic supercell helper
# ---------------------------------------------------------------------------

def _write_supercell(path: Path, span: float = 60.0, spacing: float = 2.0) -> int:
    """Write a single-type square-lattice atomic LAMMPS file; return atom count."""
    coords = []
    n = int(span / spacing) + 1
    for i in range(n):
        for j in range(n):
            coords.append((i * spacing, j * spacing))
    header = [
        "Test supercell", "",
        f"{len(coords)} atoms", "0 bonds", "0 angles", "0 dihedrals", "",
        "1 atom types", "",
        f"0.0 {span} xlo xhi", f"0.0 {span} ylo yhi", "-1.0 1.0 zlo zhi", "",
        "Masses", "", "1 95.94  # Mo", "",
        "Atoms # atomic", "",
    ]
    body = [f"{k+1} 1 {x:.4f} {y:.4f} 0.0" for k, (x, y) in enumerate(coords)]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return len(coords)


@pytest.mark.parametrize("shape,expected_vertices", [
    ("triangle", 3), ("square", 4), ("hexagon", 6),
])
def test_build_flake_cuts_shape(tmp_path: Path, shape: str, expected_vertices: int):
    src = tmp_path / "super.lmp"
    n_full = _write_supercell(src)

    out = tmp_path / f"flake_{shape}.data"
    components.build_flake(src, out, shape=shape, edge_length=20.0)

    types, xs, ys, _zs = components._parse_lmp_atoms(out)
    # The cut keeps a strict subset of atoms and at least a handful.
    assert 0 < len(types) < n_full
    assert int(types.max()) == 1


@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon"])
def test_apply_flake_corner_types_interleaves(tmp_path: Path, shape: str):
    src = tmp_path / "super.lmp"
    _write_supercell(src)
    out = tmp_path / f"flake_{shape}.data"
    components.build_flake(src, out, shape=shape, edge_length=24.0)

    components.apply_flake_corner_types(out, shape=shape, corner_radius=5.0)
    types, _xs, _ys, _zs = components._parse_lmp_atoms(out)

    # One base type -> interleaved core(1)/corner(2); corner atoms are the even type.
    assert int(types.max()) == 2
    n_corner = int((types % 2 == 0).sum())
    n_core = int((types % 2 == 1).sum())
    assert n_corner > 0, "expected at least some corner atoms"
    assert n_core > 0, "expected at least some core atoms"
    # Corners should be the minority for a reasonably sized flake.
    assert n_corner < n_core


def test_detect_corner_atoms_free_in_z_unchanged(tmp_path: Path):
    """Corner detection / retyping must not move atoms (z stays free for indent)."""
    src = tmp_path / "super.lmp"
    _write_supercell(src)
    out = tmp_path / "flake.data"
    components.build_flake(src, out, shape="square", edge_length=24.0)
    before = components._parse_lmp_atoms(out)
    components.apply_flake_corner_types(out, shape="square", corner_radius=5.0)
    after = components._parse_lmp_atoms(out)
    # Same atom count and identical coordinates (only the type column changed).
    assert len(before[0]) == len(after[0])
    assert (before[1] == after[1]).all()  # x
    assert (before[3] == after[3]).all()  # z


# ---------------------------------------------------------------------------
# FlakeConfig
# ---------------------------------------------------------------------------

@pytest.fixture
def real_files(tmp_path: Path):
    pot = tmp_path / "dummy.sw"
    cif = tmp_path / "dummy.cif"
    pot.write_text("# pot\n", encoding="utf-8")
    cif.write_text("# cif\n", encoding="utf-8")
    return str(pot), str(cif)


def test_flake_config_autosizes_supercell(real_files):
    pot, cif = real_files
    fc = FlakeConfig(mat="MoS2", pot_type="sw", pot_path=pot, cif_path=cif,
                     edge_length=30.0, corner_radius=5.0)
    assert fc.shape == "hexagon"
    # span = 2*edge_length + max(2*corner_radius, 10)
    assert fc.x == fc.y == 2 * 30.0 + 10.0


def test_afm_config_without_flake_is_none(real_files):
    pot, cif = real_files
    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif, 'r': 10.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
                'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
               'x': 50.0, 'y': 50.0, 'layers': [1]},
        'settings': load_settings().model_dump(),
    }
    cfg = AFMSimulationConfig(**data)
    assert cfg.flake is None


def test_afm_config_parses_flake_section(real_files):
    pot, cif = real_files
    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif, 'r': 10.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
                'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
               'x': 50.0, 'y': 50.0, 'layers': [1]},
        'flake': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': pot, 'cif_path': cif,
                  'shape': 'square', 'edge_length': 25.0, 'corner_spring_k': 12.0},
        'settings': load_settings().model_dump(),
    }
    cfg = AFMSimulationConfig(**data)
    assert cfg.flake is not None
    assert cfg.flake.shape == 'square'
    assert cfg.flake.edge_length == 25.0
    assert cfg.flake.corner_spring_k == 12.0


# ---------------------------------------------------------------------------
# Template wiring (corner restraint present in indentation, absent in slide)
# ---------------------------------------------------------------------------

def _make_flake_builder(tmp_path: Path) -> AFMSimulation:
    pot = tmp_path / "dummy.sw"
    cif = tmp_path / "dummy.cif"
    pot.write_text("# pot\n", encoding="utf-8")
    cif.write_text("# cif\n", encoding="utf-8")
    data = {
        'general': {'temp': 300.0, 'force': 10.0, 'scan_speed': 2.0},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif), 'r': 10.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif),
                'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif),
               'x': 50.0, 'y': 50.0, 'layers': [1]},
        'flake': {'mat': 'h-MoS2', 'pot_type': 'sw', 'pot_path': str(pot), 'cif_path': str(cif),
                  'shape': 'hexagon', 'edge_length': 20.0, 'corner_spring_k': 9.0},
        'settings': load_settings().model_dump(),
    }
    config = AFMSimulationConfig(**data)
    builder = AFMSimulation(config, output_dir=str(tmp_path / "out"))

    # Hand-populate the state write_inputs() consumes (bypassing real builds).
    builder.flake_path = Path("flake_h-MoS2_hexagon.data")
    builder.flake_dims = {'xlo': 0.0, 'xhi': 50.0, 'ylo': 0.0, 'yhi': 50.0, 'zlo': -1.0, 'zhi': 4.0}
    builder.sheet_dims = {'xlo': 0.0, 'xhi': 50.0, 'ylo': 0.0, 'yhi': 50.0, 'zlo': 0.0, 'zhi': 3.0}
    builder.box_dims = {'xlo': 0.0, 'xhi': 70.0, 'ylo': 0.0, 'yhi': 70.0, 'zlo': -5.0, 'zhi': 0.0}
    builder.sheet_offset = {'x': 0.0, 'y': 0.0}
    builder.lat_c = 6.0
    builder.sheet_paths[1] = Path("sheet.lmp")
    builder.tip_path = Path("tip.lmp")
    builder.sub_path = Path("sub.lmp")
    builder.output_dir_layer[1] = builder.output_dir / "L1"
    builder.relative_run_dir_layer[1] = builder.relative_run_dir / "L1"
    (builder.output_dir_layer[1] / "lammps").mkdir(parents=True, exist_ok=True)

    builder.z_positions[1] = {'sub': 0.0, 'sheet': 12.0, 'flake': 18.0, 'tip': 40.0, 'tip_contact_gap': 3.5}
    builder.groups[1] = {
        'sub_types': '1 2',
        'tip_types': '3',
        'sheet_types': '4 5',
        'flake_types': '6 7 8 9',
        'flake_corner_types': '7 9',
    }
    fake_pm = MagicMock()
    fake_pm.types.__len__.return_value = 9
    builder.pm[1] = fake_pm
    return builder


def test_system_init_has_corner_restraint(tmp_path: Path):
    builder = _make_flake_builder(tmp_path)
    builder.write_inputs(1)

    system_in = (builder.output_dir_layer[1] / "lammps" / "system.in").read_text()
    # Flake is read into the indentation system.
    assert "group flake offset" in system_in
    assert "flake_h-MoS2_hexagon.data" in system_in
    # Corner atoms tethered in xy (free in z) with the configured stiffness.
    assert "group           flake_corner type 7 9" in system_in
    assert "fix             corner_hold flake_corner spring/self 9.0 xy" in system_in


def test_slide_has_no_corner_restraint(tmp_path: Path):
    builder = _make_flake_builder(tmp_path)
    builder.write_inputs(1)

    slide_in = (builder.output_dir_layer[1] / "lammps" / "slide.in").read_text()
    # The flake rides into the load data file; sliding must not re-tether corners.
    assert "corner_hold" not in slide_in
    assert "flake_corner" not in slide_in
