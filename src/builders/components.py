"""Component builders for FrictionSim2D.

This module contains the logic to construct the physical components of the simulation:
the AFM tip, the substrate, and the 2D material sheet. It handles the calculation
of duplication factors based on requested dimensions and unit cell properties.

Function Organization:
1. Low-level utilities (run_lammps_commands, calculate_layer_shifts)
2. Slab creation (create_orthogonal_slab)
3. Amorphous material helpers (get_amorphous_path, make_amorphous)
4. Sheet stacking helpers (stack_multilayer_sheet)
5. Main builders (build_tip, build_substrate, build_sheet)
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
from importlib import resources
import re
import math
import subprocess
import numpy as np

from ase import io as ase_io
from jinja2 import Environment

from ..core.config import (
    GlobalSettings, SheetConfig, SubstrateConfig, TipConfig, ComponentConfig, FlakeConfig
)
from ..core.potential_manager import PotentialManager
from ..core.utils import (
    cifread, get_material_path, get_model_dimensions,
    renumber_atom_types, check_potential_cif_compatibility, get_num_atom_types,
    atomic2charge, charge2atom, shift_atoms_to_z_zero, get_potential_element_order
)
from ..interfaces.atomsk import AtomskWrapper
from ..interfaces.jinja import PackageLoader
from ..interfaces.lammps import run_lammps_commands

logger = logging.getLogger(__name__)


def _create_lammps_instance():
    from lammps import lammps  # pylint: disable=import-outside-toplevel

    return lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])


# =============================================================================
# Flake Geometry and Building
# =============================================================================

def _require_box_dims(
    dims: Dict[str, Optional[float]],
    *,
    context: str,
) -> Dict[str, float]:
    """Validate and narrow parsed box dimensions to floats.

    Args:
        dims: Mapping returned by get_model_dimensions.
        context: Text included in ValueError when a key is missing.

    Returns:
        Dictionary with fully-typed float values for x/y/z bounds.
    """
    keys = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
    missing = [k for k in keys if dims.get(k) is None]
    if missing:
        raise ValueError(f"Could not parse box dimensions ({missing}) from {context}")
    out: Dict[str, float] = {}
    for key in keys:
        value = dims[key]
        if value is None:
            raise ValueError(f"Missing required box dimension '{key}' in {context}")
        out[key] = float(value)
    return out


def _parse_lmp_atoms(
    lmp_path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Parse atom types and Cartesian positions from a LAMMPS data file.

    Supports both 'atomic' and 'charge' atom styles (auto-detected from the
    ``Atoms`` header comment).

    Returns:
        Tuple of (types, xs, ys, zs) numpy arrays, or None if no atoms found.
    """
    lines = lmp_path.read_text(encoding='utf-8').splitlines()

    atom_style = 'atomic'
    atoms_start: Optional[int] = None
    for i, line in enumerate(lines):
        if re.match(r'^\s*Atoms\s*(#|$)', line):
            if 'charge' in line.lower():
                atom_style = 'charge'
            atoms_start = i + 2
            break
    if atoms_start is None:
        return None

    types: List[int] = []
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    for line in lines[atoms_start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            if types:
                break
            continue
        parts = stripped.split()
        try:
            if atom_style == 'charge':
                t = int(parts[1])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            else:
                t = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            types.append(t)
            xs.append(x)
            ys.append(y)
            zs.append(z)
        except (ValueError, IndexError):
            break

    if not types:
        return None
    return np.array(types), np.array(xs), np.array(ys), np.array(zs)


def _in_equilateral_triangle(
    x: float,
    y: float,
    edge_length: float,
    center: Tuple[float, float] = (0.0, 0.0),
) -> bool:
    """Check if point (x, y) is inside an equilateral triangle (apex pointing up)."""
    cx, cy = center
    x_rel = x - cx
    y_rel = y - cy

    height = edge_length * math.sqrt(3) / 2.0
    half_base = edge_length / 2.0

    y_apex = height / 3.0
    y_base = -2.0 * height / 3.0

    if y_rel < y_base or y_rel > y_apex:
        return False

    left_bound = -half_base * (1.0 - (y_rel - y_base) / height)
    right_bound = half_base * (1.0 - (y_rel - y_base) / height)

    return left_bound <= x_rel <= right_bound


def _in_square(
    x: float,
    y: float,
    edge_length: float,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> bool:
    """Check if point (x, y) is inside a (optionally rotated) square."""
    cx, cy = center
    x_rel = x - cx
    y_rel = y - cy

    if rotation_deg != 0.0:
        angle_rad = math.radians(-rotation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x_rot = cos_a * x_rel - sin_a * y_rel
        y_rot = sin_a * x_rel + cos_a * y_rel
    else:
        x_rot, y_rot = x_rel, y_rel

    half = edge_length / 2.0
    return -half <= x_rot <= half and -half <= y_rot <= half


def _in_hexagon(
    x: float,
    y: float,
    edge_length: float,
    center: Tuple[float, float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> bool:
    """Check if point (x, y) is inside a (optionally rotated) regular hexagon.

    ``edge_length`` is the distance from the center to a vertex (circumradius).
    """
    cx, cy = center
    x_rel = x - cx
    y_rel = y - cy

    if rotation_deg != 0.0:
        angle_rad = math.radians(-rotation_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        x_rot = cos_a * x_rel - sin_a * y_rel
        y_rot = sin_a * x_rel + cos_a * y_rel
    else:
        x_rot, y_rot = x_rel, y_rel

    apothem = edge_length * math.sqrt(3) / 2.0

    thresholds = [
        abs(y_rot) <= apothem,
        abs(x_rot) <= edge_length,
        abs(x_rot / 2.0 + math.sqrt(3) * y_rot / 2.0) <= apothem,
        abs(x_rot / 2.0 - math.sqrt(3) * y_rot / 2.0) <= apothem,
    ]

    return all(thresholds)


def build_flake(
    base_lmp_path: Path,
    output_path: Path,
    shape: str = "triangle",
    edge_length: float = 40.0,
    rotation_deg: float = 0.0,
    center_xy: Optional[Tuple[float, float]] = None,
) -> Path:
    """Extract a natural-shaped flake from a supercell LAMMPS file.

    Creates a finite patch (triangle/square/hexagon) by selecting atoms within
    the shape boundary and writing them to a new LAMMPS data file. The flake is
    centered at the box center (or ``center_xy``) and can be rotated.

    Args:
        base_lmp_path: Path to source LAMMPS data file (single layer).
        output_path: Path for output flake file.
        shape: Flake shape ('triangle', 'square', 'hexagon').
        edge_length: Characteristic edge length (Å). For triangle/square: side;
            for hexagon: circumradius (vertex distance from center).
        rotation_deg: Rotation angle (degrees, counter-clockwise).
        center_xy: Override flake center (default: box center).

    Returns:
        Path to the output flake file.

    Raises:
        ValueError: If shape is not recognized or if base file has no atoms.
    """
    shape_lower = shape.lower()
    if shape_lower not in ("triangle", "square", "hexagon"):
        raise ValueError(
            f"Unknown flake shape: {shape}. Choose from: triangle, square, hexagon."
        )

    dims = _require_box_dims(get_model_dimensions(base_lmp_path), context=str(base_lmp_path))
    xlo, xhi = dims["xlo"], dims["xhi"]
    ylo, yhi = dims["ylo"], dims["yhi"]

    if center_xy is None:
        cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    else:
        cx, cy = center_xy

    parsed = _parse_lmp_atoms(base_lmp_path)
    if parsed is None or len(parsed[0]) == 0:
        raise ValueError(f"No atoms found in {base_lmp_path}")

    types, xs, ys, zs = parsed
    natoms = len(types)

    boundary_check = {
        "triangle": lambda x, y: _in_equilateral_triangle(x, y, edge_length, (cx, cy)),
        "square": lambda x, y: _in_square(x, y, edge_length, (cx, cy), rotation_deg),
        "hexagon": lambda x, y: _in_hexagon(x, y, edge_length, (cx, cy), rotation_deg),
    }[shape_lower]

    mask = np.array([boundary_check(xs[i], ys[i]) for i in range(natoms)], dtype=bool)
    if not mask.any():
        raise ValueError(
            f"No atoms fell within the {shape_lower} flake boundary "
            f"(edge_length={edge_length:.1f} Å). Is the supercell large enough?"
        )

    selected_indices = np.where(mask)[0]
    lines = base_lmp_path.read_text(encoding="utf-8").splitlines()

    atom_style = "atomic"
    atoms_start: Optional[int] = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*Atoms\s*(#|$)", line):
            if "charge" in line.lower():
                atom_style = "charge"
            atoms_start = i + 2
            break
    if atoms_start is None:
        raise ValueError(f"Could not find Atoms section in {base_lmp_path}")

    atom_lines: List[str] = []
    new_id = 1
    for idx in selected_indices:
        raw_line = lines[atoms_start + idx]
        parts = raw_line.strip().split()
        if not parts or not parts[0].isdigit():
            raise ValueError(
                f"Malformed Atoms entry while building flake at line "
                f"{atoms_start + idx + 1}: '{raw_line}'"
            )
        parts[0] = str(new_id)
        atom_lines.append("  ".join(parts))
        new_id += 1

    selected_z = zs[selected_indices]
    z_min, z_max = float(np.min(selected_z)), float(np.max(selected_z))
    num_types = int(np.max(types[selected_indices]))

    output_lines: List[str] = []
    output_lines.append(f"Flake ({shape_lower}, edge={edge_length:.1f} A)")
    output_lines.append("")
    output_lines.append(f"{len(selected_indices)} atoms")
    output_lines.append("0 bonds")
    output_lines.append("0 angles")
    output_lines.append("0 dihedrals")
    output_lines.append("")
    output_lines.append(f"{num_types} atom types")
    output_lines.append("")
    output_lines.append(f"{xlo:.6f} {xhi:.6f} xlo xhi")
    output_lines.append(f"{ylo:.6f} {yhi:.6f} ylo yhi")
    output_lines.append(f"{z_min - 1.0:.6f} {z_max + 1.0:.6f} zlo zhi")
    output_lines.append("")

    # Copy the Masses section verbatim from the source file.
    masses_section: List[str] = []
    in_masses = False
    seen_mass_values = False
    for line in lines:
        if line.strip() == "Masses":
            in_masses = True
            masses_section.append(line)
            continue
        if in_masses:
            if not line.strip() and not seen_mass_values:
                masses_section.append(line)
                continue
            if not line.strip() and seen_mass_values:
                masses_section.append(line)
                break
            if line.strip() and line.strip()[0].isalpha():
                break
            seen_mass_values = True
            masses_section.append(line)

    if masses_section:
        output_lines.extend(masses_section)
        output_lines.append("")

    output_lines.append(f"Atoms # {atom_style}")
    output_lines.append("")
    output_lines.extend(atom_lines)
    output_lines.append("")

    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    logger.info(
        "[flake] Created %s flake at (%.1f, %.1f) with %d atoms",
        shape_lower, cx, cy, len(selected_indices),
    )

    return output_path


_SHAPE_N_VERTICES = {"triangle": 3, "square": 4, "hexagon": 6}


def _detect_corner_atom_indices(
    xs: np.ndarray,
    ys: np.ndarray,
    n_vertices: int,
    corner_radius: float,
) -> np.ndarray:
    """Identify the atom indices that sit at the polygon's corners.

    Robust to shape orientation/rotation: the flake centroid is computed, atoms
    are sorted into ``n_vertices`` angular sectors aligned with the farthest
    atom, the farthest atom in each sector is taken as the vertex, and every
    atom within ``corner_radius`` of any vertex is flagged as a corner atom.

    Args:
        xs, ys: Atom coordinates.
        n_vertices: Number of polygon corners (3, 4 or 6).
        corner_radius: Capture radius (Å) around each vertex.

    Returns:
        Sorted numpy array of corner atom indices.
    """
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    dx = xs - cx
    dy = ys - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    # Align sector boundaries so one sector is centered on the farthest atom.
    theta0 = float(theta[int(np.argmax(r))])
    sector = 2.0 * math.pi / n_vertices

    corner_mask = np.zeros(len(xs), dtype=bool)
    for k in range(n_vertices):
        center_angle = theta0 + k * sector
        # Wrap angular difference into [-pi, pi].
        d_angle = np.angle(np.exp(1j * (theta - center_angle)))
        in_sector = np.abs(d_angle) <= sector / 2.0
        if not in_sector.any():
            continue
        sector_idx = np.where(in_sector)[0]
        vertex_idx = sector_idx[int(np.argmax(r[sector_idx]))]
        vx, vy = xs[vertex_idx], ys[vertex_idx]
        near_vertex = np.hypot(xs - vx, ys - vy) <= corner_radius
        corner_mask |= near_vertex

    return np.where(corner_mask)[0]


def apply_flake_corner_types(
    flake_path: Path,
    shape: str,
    corner_radius: float,
) -> Path:
    """Re-type the flake's corner atoms into distinct LAMMPS atom types.

    Mirrors :func:`apply_sheet_edge_types`: every original type ``t`` is split
    into an interleaved core type ``2t-1`` and corner type ``2t`` so the result
    matches ``PotentialManager.register_component(..., regions=[None, 'corner'])``.
    The corner atoms can then be selected in LAMMPS via ``group ... type``.

    The flake data file is modified in-place. Returns the same path.

    Args:
        flake_path: LAMMPS data file produced by :func:`build_flake`.
        shape: Flake shape ('triangle', 'square', 'hexagon') — sets vertex count.
        corner_radius: Capture radius (Å) around each detected vertex.

    Raises:
        ValueError: If the shape is unknown or the file has no atoms.
    """
    shape_lower = shape.lower()
    if shape_lower not in _SHAPE_N_VERTICES:
        raise ValueError(
            f"Unknown flake shape: {shape}. Choose from: triangle, square, hexagon."
        )
    n_vertices = _SHAPE_N_VERTICES[shape_lower]

    parsed = _parse_lmp_atoms(flake_path)
    if parsed is None or len(parsed[0]) == 0:
        raise ValueError(f"No atoms found in {flake_path}")
    types, xs, ys, _zs = parsed

    corner_indices = set(
        int(i) for i in _detect_corner_atom_indices(xs, ys, n_vertices, corner_radius)
    )
    logger.info(
        "[flake] Marked %d corner atoms (of %d) for %s flake",
        len(corner_indices), len(types), shape_lower,
    )

    lines = flake_path.read_text(encoding="utf-8").splitlines()

    # Locate sections.
    atoms_start: Optional[int] = None
    masses_start: Optional[int] = None
    for i, line in enumerate(lines):
        if line.strip() == "Masses":
            masses_start = i
        if re.match(r"^\s*Atoms\s*(#|$)", line):
            atoms_start = i + 2
            break
    if atoms_start is None:
        raise ValueError(f"Could not find Atoms section in {flake_path}")

    num_types = int(np.max(types))

    # Rewrite the atom type column: core -> 2t-1, corner -> 2t.
    atom_seq = 0
    for li in range(atoms_start, len(lines)):
        stripped = lines[li].strip()
        if not stripped or stripped.startswith("#"):
            if atom_seq > 0:
                break
            continue
        parts = stripped.split()
        if not parts[0].isdigit():
            break
        old_type = int(parts[1])
        is_corner = atom_seq in corner_indices
        parts[1] = str(2 * old_type if is_corner else 2 * old_type - 1)
        lines[li] = "  ".join(parts)
        atom_seq += 1

    # Update the atom-types header.
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d+\s+atom types\s*$", line.strip()):
            lines[i] = f"{2 * num_types} atom types"
            break

    # Expand the Masses section: each original type -> core + corner with same mass.
    if masses_start is not None:
        new_mass_lines: List[str] = []
        mi = masses_start + 1
        # Skip the blank line after 'Masses'.
        while mi < len(lines) and not lines[mi].strip():
            mi += 1
        mass_end = mi
        old_masses: Dict[int, str] = {}
        while mass_end < len(lines):
            s = lines[mass_end].strip()
            if not s:
                break
            if s[0].isalpha():
                break
            mparts = s.split()
            old_masses[int(mparts[0])] = mparts[1]
            mass_end += 1
        for t in range(1, num_types + 1):
            mass_val = old_masses.get(t, "1.0")
            new_mass_lines.append(f"{2 * t - 1} {mass_val}")
            new_mass_lines.append(f"{2 * t} {mass_val}")
        lines[mi:mass_end] = new_mass_lines

    flake_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flake_path


def build_flake_component(
    config: "FlakeConfig",
    build_dir: Path,
) -> Tuple[Path, dict]:
    """Build a shaped flake (monolayer supercell → cut shape → mark corners).

    Reuses :func:`build_monolayer` to make a supercell of the flake's own
    material, :func:`build_flake` to carve the requested outline, and
    :func:`apply_flake_corner_types` to split corner atoms into their own
    interleaved types (matching ``regions=[None, 'corner']``).

    Args:
        config: Flake configuration (``mat``/``cif``/``pot`` + ``shape``,
            ``edge_length``, ``rotation_deg``, ``corner_radius``, supercell
            ``x``/``y``).
        build_dir: Directory to write the final ``flake_*.data`` file into.

    Returns:
        Tuple of (flake_path, flake_box_dims).
    """
    base_path, _dims, _pot_counts, _total_types, _supercell = build_monolayer(
        config, edge_mode='none', edge_width=0.0
    )

    center_xy: Optional[Tuple[float, float]] = None
    if config.center_x is not None and config.center_y is not None:
        center_xy = (float(config.center_x), float(config.center_y))

    flake_path = build_dir / f"flake_{config.mat}_{config.shape}.data"
    build_flake(
        base_path,
        flake_path,
        shape=config.shape,
        edge_length=float(config.edge_length),
        rotation_deg=float(config.rotation_deg),
        center_xy=center_xy,
    )
    base_path.unlink(missing_ok=True)

    apply_flake_corner_types(
        flake_path,
        shape=config.shape,
        corner_radius=float(config.corner_radius),
    )

    flake_dims = _require_box_dims(
        get_model_dimensions(flake_path), context=str(flake_path)
    )
    logger.info("Built flake component: %s", flake_path.name)
    return flake_path, flake_dims


def _write_cif_as_lammps_atomic(
    cif_path: Path,
    output_path: Path,
    specorder: Optional[List[str]] = None
) -> None:
    """Write a CIF structure as LAMMPS data in atomic style.

    ASE's LAMMPS writer omits the Masses section, which is required by
    renumber_atom_types for element identification.  This function injects
    the missing Masses block after the atom-type count header.

    Args:
        cif_path: Input CIF path.
        output_path: Output LAMMPS data path.
        specorder: Optional element order for deterministic type assignment.
    """
    from ase.data import atomic_masses, atomic_numbers  # pylint: disable=import-outside-toplevel

    atoms = ase_io.read(str(cif_path))
    if isinstance(atoms, list):
        atoms = atoms[0]

    if output_path.exists():
        output_path.unlink()

    write_kwargs = {
        'filename': str(output_path),
        'images': atoms,
        'format': 'lammps-data',
        'atom_style': 'atomic'
    }
    if specorder:
        write_kwargs['specorder'] = specorder

    ase_io.write(**write_kwargs)

    # Inject Masses section (ASE omits it); use specorder when available.
    # Insert just before the Atoms section so box-bounds lines aren't mistaken for masses.
    elements = specorder if specorder else sorted(set(atoms.get_chemical_symbols()))
    masses_block = "Masses\n\n"
    for idx, element in enumerate(elements, start=1):
        mass = atomic_masses[atomic_numbers[element]]
        masses_block += f"{idx} {mass:.6f}  # {element}\n"
    masses_block += "\n"

    content = output_path.read_text(encoding='utf-8')
    import re  # pylint: disable=import-outside-toplevel
    # Insert the Masses block immediately before the "Atoms" section header
    content = re.sub(
        r'(Atoms\s)',
        masses_block + r'\1',
        content,
        count=1
    )
    output_path.write_text(content, encoding='utf-8')


def _orthogonalize_lammps_data(input_path: Path, output_path: Path) -> None:
    """Orthogonalize a LAMMPS data file.

    For orthogonal (no-tilt) boxes, uses LAMMPS ``change_box all ortho``.
    For triclinic (tilted) boxes, uses ASE ``make_supercell`` to build the
    minimum orthogonal supercell, then writes the result as LAMMPS data and
    re-injects the Masses section preserved from the original file.

    Args:
        input_path: Input LAMMPS data path.
        output_path: Orthogonalized output path.
    """
    import re as _re  # pylint: disable=import-outside-toplevel
    from ase.build import make_supercell as _make_supercell  # pylint: disable=import-outside-toplevel

    if output_path.exists():
        output_path.unlink()

    # Check whether the box is already orthogonal (no xy xz yz line with non-zero values)
    content = input_path.read_text(encoding='utf-8')
    tilt_match = _re.search(
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+'
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+'
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+xy xz yz',
        content
    )
    has_tilt = tilt_match is not None and any(
        abs(float(tilt_match.group(i))) > 1e-10 for i in (1, 2, 3)
    )

    if not has_tilt:
        # Orthogonal box: LAMMPS change_box approach
        lmp = _create_lammps_instance()
        try:
            lmp.command("clear")
            lmp.command("units metal")
            lmp.command("atom_style atomic")
            lmp.command(f'read_data "{input_path}"')
            lmp.command("change_box all ortho")
            lmp.command(f'write_data "{output_path}"')
        finally:
            lmp.close()
        return

    # Triclinic box: build a minimum orthogonal supercell with ASE, preserving
    # each atom's explicit LAMMPS type so that sublattice types that share a
    # chemical element (e.g. Se1/Se2, Al1/Al2) survive the supercell tiling.
    # (Mapping types->elements, as ASE does by default, would collapse them.)
    # Each type is carried through make_supercell/write as a *distinct placeholder
    # element*; the real masses/elements are restored afterward. Writing via
    # ase_io.write keeps the box in LAMMPS restricted-triclinic orientation.
    # The non-tilt LAMMPS change_box path above already preserves types.
    from ase import Atoms as _Atoms  # pylint: disable=import-outside-toplevel
    from ase.data import chemical_symbols as _csym  # pylint: disable=import-outside-toplevel

    cell, atom_rows, masses = _parse_lammps_cell_atoms_masses(content)
    types = [t for (t, *_r) in atom_rows]
    positions = [(x, y, z) for (_t, x, y, z) in atom_rows]
    n_types = max(masses) if masses else (max(types) if types else 1)
    placeholders = [_csym[i] for i in range(1, n_types + 1)]  # H, He, Li, ... per type
    symbols = [placeholders[t - 1] for t in types]

    atoms = _Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    transform_matrix = _find_ortho_transform(atoms.get_cell())
    ortho_atoms = _make_supercell(atoms, transform_matrix)
    ortho_atoms.wrap()

    ase_io.write(str(output_path), ortho_atoms, format='lammps-data',
                 atom_style='atomic', specorder=placeholders)

    # Restore the real Masses section (ASE writes placeholder masses / omits it).
    real_masses = {t: masses.get(t, (1.0, 'X')) for t in range(1, n_types + 1)}
    _restore_masses_section(output_path, real_masses)
    _snap_small_tilts(output_path)


def _element_z(symbol: str) -> int:
    """Return atomic number for element symbol."""
    from ase.data import atomic_numbers  # pylint: disable=import-outside-toplevel
    return atomic_numbers.get(symbol, 1)


def _find_ortho_transform(cell) -> "np.ndarray":
    """Find a transformation matrix P such that P @ cell is (nearly) orthogonal.

    For hexagonal cells, the canonical 2× transformation [[1,0,0],[1,2,0],[0,0,1]]
    produces an orthorhombic cell.  For cells already orthogonal, returns the
    identity matrix.

    Args:
        cell: ASE Cell or 3×3 array of cell vectors (rows).

    Returns:
        3×3 integer numpy array P.
    """
    _np = np

    c = _np.array(cell)
    a1, a2 = c[0], c[1]

    # If already orthogonal, return identity
    if abs(_np.dot(a1, a2)) < 1e-6 * _np.linalg.norm(a1) * _np.linalg.norm(a2):
        return _np.eye(3, dtype=int)

    # Search for n1, n2 integers (smallest cell) such that n1*a1 + n2*a2 is ~along y.
    # A relative x-tolerance accepts slightly-distorted (near-hexagonal) cells whose
    # a/b ratio is not exactly commensurate (e.g. GaO's a2/a1 = -0.50002); the tiny
    # residual tilt is snapped to zero after the supercell is built.
    best_transform = None
    best_key = None
    x_tol = max(1e-6, 5e-3 * float(_np.linalg.norm(a1)))
    search_range = range(-12, 13)
    for n1 in search_range:
        for n2 in search_range:
            if n1 == 0 and n2 == 0:
                continue
            trial = n1 * a1 + n2 * a2
            if abs(trial[0]) < x_tol and abs(trial[1]) > 1e-3:
                key = (abs(n2), abs(n1), abs(trial[0]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_transform = _np.array([[1, 0, 0], [n1, n2, 0], [0, 0, 1]], dtype=int)

    if best_transform is not None:
        return best_transform

    # Fallback: identity (leave as-is)
    return _np.eye(3, dtype=int)



def _parse_lammps_cell_atoms_masses(content: str):
    """Parse a LAMMPS ``atomic``-style data file string.

    Returns:
        Tuple of (cell 3x3 list, atoms [(type, x, y, z)], masses {type: (mass, element)}).
        Atoms are returned in file order; the cell includes any xy/xz/yz tilt.
    """
    bounds = {}
    tilt = {'xy': 0.0, 'xz': 0.0, 'yz': 0.0}
    masses: Dict[int, tuple] = {}
    atoms = []
    section = None
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.split('#', 1)[0].strip()
        if low.endswith(('xlo xhi', 'ylo yhi', 'zlo zhi')):
            parts = low.split()
            key = parts[2][0]  # 'x' | 'y' | 'z'
            bounds[f'{key}lo'] = float(parts[0])
            bounds[f'{key}hi'] = float(parts[1])
            continue
        if low.endswith('xy xz yz'):
            parts = low.split()
            tilt['xy'], tilt['xz'], tilt['yz'] = (float(parts[0]), float(parts[1]), float(parts[2]))
            continue
        if line in ('Masses', 'Atoms') or line.startswith('Atoms'):
            section = 'Masses' if line == 'Masses' else 'Atoms'
            continue
        if line in ('Velocities', 'Bonds', 'Angles'):
            section = None
            continue
        if section == 'Masses':
            parts = line.split()
            if parts[0].lstrip('-').isdigit():
                elem = line.split('#')[-1].strip().split()[0] if '#' in line else 'X'
                masses[int(parts[0])] = (float(parts[1]), elem)
        elif section == 'Atoms':
            parts = low.split()
            if len(parts) >= 5 and parts[0].lstrip('-').isdigit():
                atoms.append((int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

    lx = bounds['xhi'] - bounds['xlo']
    ly = bounds['yhi'] - bounds['ylo']
    lz = bounds['zhi'] - bounds['zlo']
    cell = [[lx, 0.0, 0.0], [tilt['xy'], ly, 0.0], [tilt['xz'], tilt['yz'], lz]]
    return cell, atoms, masses


def _write_lammps_typed(output_path: Path, cell, positions, types,
                        masses: Dict[int, tuple]) -> None:
    """Write a LAMMPS ``atomic`` data file preserving explicit per-atom types.

    Args:
        output_path: Destination path.
        cell: 3x3 lower-triangular cell (rows are a, b, c vectors).
        positions: Nx3 atom coordinates.
        types: length-N per-atom integer type ids.
        masses: mapping type-id -> (mass, element) for the Masses section.
    """
    n_types = max(masses) if masses else max(types)
    lx, ly, lz = cell[0][0], cell[1][1], cell[2][2]
    xy, xz, yz = cell[1][0], cell[2][0], cell[2][1]
    lines = [
        "LAMMPS data file via FrictionSim2D (type-preserving orthogonalization)\n\n",
        f"{len(positions)} atoms\n",
        f"{n_types} atom types\n\n",
        f"0.0 {lx:.12f} xlo xhi\n",
        f"0.0 {ly:.12f} ylo yhi\n",
        f"0.0 {lz:.12f} zlo zhi\n",
    ]
    if any(abs(v) > 1e-10 for v in (xy, xz, yz)):
        lines.append(f"{xy:.12f} {xz:.12f} {yz:.12f} xy xz yz\n")
    lines.append("\nMasses\n\n")
    for t in range(1, n_types + 1):
        mass, elem = masses.get(t, (1.0, 'X'))
        lines.append(f"{t} {mass:.6f}  # {elem}\n")
    lines.append("\nAtoms # atomic\n\n")
    for i, (pos, t) in enumerate(zip(positions, types), start=1):
        lines.append(f"{i} {int(t)} {pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}\n")
    output_path.write_text(''.join(lines), encoding='utf-8')


def _snap_small_tilts(path: Path, tol: float = 0.05) -> None:
    """Drop a negligible xy/xz/yz tilt line so the box is treated as orthogonal.

    Near-hexagonal cells orthogonalize to a box with a sub-0.01 Å residual tilt
    (float non-commensurability). LAMMPS then refuses to stack such a box onto an
    orthogonal one ("cannot switch to restricted triclinic"), so we snap tiny
    tilts to zero (atom displacement over the box is far below tol).
    """
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    out = []
    for line in lines:
        low = line.split('#', 1)[0].strip()
        if low.endswith('xy xz yz'):
            parts = low.split()
            if all(abs(float(parts[i])) < tol for i in range(3)):
                continue  # drop the tilt line -> orthogonal box
        out.append(line)
    path.write_text(''.join(out), encoding='utf-8')


def _restore_masses_section(path: Path, masses: Dict[int, tuple]) -> None:
    """Insert/replace the Masses section of a LAMMPS data file with real values.

    ``ase_io.write`` for ``lammps-data`` omits (or writes placeholder) masses;
    this writes ``type mass  # element`` lines so downstream element parsing works.

    Args:
        path: LAMMPS data file (modified in-place).
        masses: mapping type-id -> (mass, element).
    """
    import re as _re  # pylint: disable=import-outside-toplevel

    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    out = []
    skip_masses = False
    for line in lines:
        stripped = line.strip()
        if stripped == 'Masses':
            skip_masses = True
            continue
        if skip_masses:
            # Consume the existing (placeholder) Masses block until a blank-then-section.
            if stripped and not stripped[0].isdigit():
                skip_masses = False
                # fall through to emit this line (likely 'Atoms ...')
            else:
                continue
        out.append(line)

    block = "Masses\n\n"
    for t in range(1, (max(masses) if masses else 0) + 1):
        mass, elem = masses.get(t, (1.0, 'X'))
        block += f"{t} {mass:.6f}  # {elem}\n"
    block += "\n"

    text = ''.join(out)
    text = _re.sub(r'(^|\n)(Atoms)', r'\1' + block + r'\2', text, count=1)
    path.write_text(text, encoding='utf-8')


def _assign_pot_types_by_order(path: Path, order: List[str],
                               counts: Dict[str, int]) -> None:
    """Assign potential sublattice types to a primitive cell, in-place.

    Used when the number of atoms equals the number of potential types
    (``type_multiplier == 1`` with sublattice types, e.g. t-PtSe2's Pt/Se1/Se2).
    Each element's atoms are assigned that element's block of sublattice type ids
    in file order (cycling if there are more atoms than sublattice types), giving
    exactly ``sum(counts)`` types. The subsequent orthogonalization tiles these
    types across the supercell.

    Args:
        path: Primitive LAMMPS data file (modified in-place).
        order: Element order from the potential (e.g. ['Pt', 'Se']).
        counts: Sublattice-type count per element (e.g. {'Pt': 1, 'Se': 2}).
    """
    from ase.data import atomic_masses as _am, atomic_numbers as _an  # pylint: disable=import-outside-toplevel

    cell, atoms, masses = _parse_lammps_cell_atoms_masses(path.read_text(encoding='utf-8'))

    # Map each current type-id to an element. Prefer the Masses ``# element``
    # comment, but fall back to the nearest atomic mass among the potential's
    # elements when it is missing: LAMMPS ``replicate``/``write_data`` (used by
    # _duplicate_lammps_data just before this call) strips those comments. Without
    # the fallback every atom is left unmatched, its type is never reassigned, and
    # the potential's sublattice types collapse to one type per element.
    order_masses = {el: float(_am[_an[el]]) for el in order}

    def _element_for(cur_type: int) -> str:
        mass, elem = masses.get(cur_type, (None, 'X'))
        if elem in order_masses:
            return elem
        if mass is not None and order_masses:
            return min(order_masses, key=lambda el: abs(order_masses[el] - mass))
        return order[0]

    cur_type_elem = {t: _element_for(t) for t in masses}

    # Contiguous type-id block per element, in potential order.
    elem_type_ids: Dict[str, List[int]] = {}
    next_id = 1
    for el in order:
        n = int(counts[el])
        elem_type_ids[el] = list(range(next_id, next_id + n))
        next_id += n

    counters = {el: 0 for el in order}
    new_types = []
    for (cur_type, *_xyz) in atoms:
        el = cur_type_elem.get(cur_type, order[0])
        ids = elem_type_ids.get(el)
        if not ids:  # element not in potential order: leave as-is
            new_types.append(cur_type)
            continue
        new_types.append(ids[counters[el] % len(ids)])
        counters[el] += 1

    new_masses = {
        tid: (float(_am[_an[el]]), el)
        for el, ids in elem_type_ids.items() for tid in ids
    }
    positions = [(x, y, z) for (_t, x, y, z) in atoms]
    _write_lammps_typed(path, cell, positions, new_types, new_masses)


def _duplicate_lammps_data(
    input_path: Path,
    output_path: Path,
    nx: int,
    ny: int,
    nz: int
) -> None:
    """Duplicate a LAMMPS data file via native LAMMPS replicate.

    This preserves existing atom type IDs, which is required after renumbering.

    Args:
        input_path: Input LAMMPS data path.
        output_path: Duplicated output path.
        nx: X multiplier.
        ny: Y multiplier.
        nz: Z multiplier.
    """
    if output_path.exists():
        output_path.unlink()

    lmp = _create_lammps_instance()
    try:
        lmp.command("clear")
        lmp.command("units metal")
        lmp.command("atom_style atomic")
        lmp.command(f'read_data "{input_path}"')
        lmp.command(f"replicate {nx} {ny} {nz}")
        lmp.command(f'write_data "{output_path}"')
    finally:
        lmp.close()


def _annotate_masses(path: Path, type_element_map: Dict[int, str]) -> None:
    """Re-add ``# element`` comments to the Masses section of a LAMMPS data file.

    LAMMPS ``write_data`` strips element comments; this restores them so that
    ``renumber_atom_types`` can read the element names in its PASS 1.

    Args:
        path: Path to LAMMPS data file (modified in-place).
        type_element_map: Mapping of atom-type int -> element symbol.
    """
    content = path.read_text(encoding='utf-8')
    lines = content.splitlines(keepends=True)
    in_masses = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == 'Masses':
            in_masses = True
            new_lines.append(line)
            continue
        if in_masses:
            if not stripped:
                new_lines.append(line)
                continue
            if not stripped[0].isdigit():
                in_masses = False
                new_lines.append(line)
                continue
            parts = stripped.split()
            type_id = int(parts[0])
            if type_id in type_element_map and '#' not in stripped:
                line = line.rstrip('\n').rstrip() + f'  # {type_element_map[type_id]}\n'
            new_lines.append(line)
        else:
            new_lines.append(line)
    path.write_text(''.join(new_lines), encoding='utf-8')


def _strip_velocities(path: Path) -> None:
    """Remove the Velocities section (and any extra 0 0 0 velocities) from a LAMMPS data file.

    LAMMPS ``write_data`` writes a Velocities section; ``renumber_atom_types``
    appends atom lines at the end of the file, so the Velocities section must
    be removed first to keep Atoms as the last section.

    Args:
        path: Path to LAMMPS data file (modified in-place).
    """
    content = path.read_text(encoding='utf-8')
    lines = content.splitlines(keepends=True)
    new_lines = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped == 'Velocities':
            skip = True
            continue
        if skip:
            # Skip until next section header or end of file
            if stripped and not stripped[0].isdigit() and not stripped[0] == '-':
                # New section header
                skip = False
                new_lines.append(line)
            # else skip this line (velocity data or blank)
        else:
            new_lines.append(line)
    path.write_text(''.join(new_lines), encoding='utf-8')

def calculate_layer_shifts(
    mat_name: str,
    dims: Dict[str, float],
    n_layers: int = 1,
    use_pair_bonding: bool = False,
    stacking_type: str = 'AB'
) -> List[Tuple[float, float]]:
    """Calculate in-plane stacking shifts for each layer.
    
    Args:
        mat_name: Material name.
        dims: Box dimensions.
        n_layers: Number of layers.
        use_pair_bonding: If True, first and last pairs are bonded with no relative shift.
        stacking_type: Stacking type ('AA' or 'AB'). AA has no shifts, AB has shifts.
        
    Returns:
        List of (shift_x, shift_y) tuples per layer.
    """
    if stacking_type.upper() == 'AA':
        return [(0.0, 0.0) for _ in range(n_layers)]

    if mat_name.startswith('p-') or mat_name == 'black_phosphorus':
        base_shift_x = (dims['xhi'] - dims['xlo']) / 2
        base_shift_y = (dims['yhi'] - dims['ylo']) / 2
    else:
        base_shift_x = 0.0
        base_shift_y = (dims['yhi'] - dims['ylo']) / 3

    layer_shifts = []

    if use_pair_bonding and n_layers >= 3:
        for i in range(n_layers):
            if n_layers == 3:
                if i == 0:
                    layer_shifts.append((0.0, 0.0))
                else:
                    layer_shifts.append((base_shift_x, base_shift_y))
            else:
                if i <= 1:
                    layer_shifts.append((0.0, 0.0))
                elif i >= n_layers - 2:
                    num_shifts = n_layers - 3
                    layer_shifts.append((base_shift_x * num_shifts, base_shift_y * num_shifts))
                else:
                    num_shifts = i - 1
                    layer_shifts.append((base_shift_x * num_shifts, base_shift_y * num_shifts))
    else:
        for i in range(n_layers):
            layer_shifts.append((base_shift_x * i, base_shift_y * i))

    return layer_shifts

def create_orthogonal_slab(
    cif_path: Path,
    output_path: Path,
    target_x: float,
    target_y: float,
    target_z: Optional[float] = None,
    atomsk: Optional[AtomskWrapper] = None
) -> Tuple[Path, Dict[str, float], List[int]]:
    """Create orthogonalized slab from CIF file sized to target dimensions.
    
    Args:
        cif_path: Input CIF file path.
        output_path: Output LAMMPS data file path.
        target_x: Target X dimension (Angstroms).
        target_y: Target Y dimension (Angstroms).
        target_z: Target Z dimension (Angstroms), or None for single layer.
        atomsk: Retained for API compatibility (unused).
        
    Returns:
        Tuple of (output_path, box_dimensions, duplication_factors).
    """
    # Keep `atomsk` in signature for backward compatibility with existing callers.
    _ = atomsk
    cif_path = Path(cif_path).absolute()

    unit_cell = Path(tempfile.gettempdir()) / f"{cif_path.stem}_{uuid.uuid4().hex[:8]}_unit.lmp"
    predup_cell = Path(tempfile.gettempdir()) / f"{cif_path.stem}_{uuid.uuid4().hex[:8]}_predup.lmp"
    ortho_cell = Path(tempfile.gettempdir()) / f"{cif_path.stem}_{uuid.uuid4().hex[:8]}_ortho.lmp"

    cif_data = cifread(cif_path)
    _write_cif_as_lammps_atomic(cif_path, unit_cell, specorder=cif_data['elements'])
    _duplicate_lammps_data(unit_cell, predup_cell, 2, 2, 1)
    _orthogonalize_lammps_data(predup_cell, ortho_cell)
    unit_cell.unlink()
    predup_cell.unlink()

    dims = get_model_dimensions(ortho_cell)
    assert all(dims[k] is not None for k in ['xhi', 'xlo', 'yhi', 'ylo', 'zhi', 'zlo'])

    unit_x = (cast(float, dims['xhi']) - cast(float, dims['xlo'])) / 2.0
    unit_y = (cast(float, dims['yhi']) - cast(float, dims['ylo'])) / 2.0
    unit_z = cast(float, dims['zhi']) - cast(float, dims['zlo'])

    dup_x = max(1, round(target_x / unit_x / 2.0))
    dup_y = max(1, round(target_y / unit_y / 2.0))
    dup_z = 1 if target_z is None else max(1, round(target_z / unit_z))

    duplication = [dup_x, dup_y, dup_z]

    if dup_x == 1 and dup_y == 1 and dup_z == 1:
        shutil.copy(ortho_cell, output_path)
    else:
        _duplicate_lammps_data(ortho_cell, output_path, dup_x, dup_y, dup_z)

    ortho_cell.unlink()

    final_dims = get_model_dimensions(output_path)
    assert all(final_dims[k] is not None for k in ['xhi', 'xlo', 'yhi', 'ylo', 'zhi', 'zlo'])
    final_dims_typed: Dict[str, float] = {k: cast(float, v) for k, v in final_dims.items()}
    return output_path, final_dims_typed, duplication

def get_amorphous_path(mat_name: str) -> Optional[Path]:
    """Get path to pre-generated amorphous material file if it exists.
    
    Args:
        mat_name: Material name.
        
    Returns:
        Path to amorphous file or None.
    """
    mat_dir = resources.files('src.data.materials')
    amor_file = mat_dir.joinpath(f"amor_{mat_name}.lmp")
    return Path(str(amor_file)) if amor_file.is_file() else None

def make_amorphous(
    mat_name: str,
    cif_path: Path,
    target_x: float,
    target_y: float,
    target_z: float,
    pot_path: Path,
    pot_type: str,
    output_dir: Path,
    settings: GlobalSettings
) -> Path:
    """Get or generate amorphous structure via melt-quench procedure.
    
    Args:
        mat_name: Material name.
        cif_path: Crystalline CIF file path.
        target_x: Required X dimension (Angstroms).
        target_y: Required Y dimension (Angstroms).
        target_z: Required Z dimension (Angstroms).
        pot_path: Potential file path.
        pot_type: Potential type (sw, tersoff, etc.).
        output_dir: Directory for temporary files.
        settings: Global settings with quench parameters.
        
    Returns:
        Path to amorphous structure file with sufficient dimensions.
    """
    amor_path = get_amorphous_path(mat_name)
    if amor_path:
        amor_dims = get_model_dimensions(amor_path)
        assert all(amor_dims[k] is not None for k in ['xhi', 'xlo', 'yhi', 'ylo', 'zhi', 'zlo'])

        amor_x = cast(float, amor_dims['xhi']) - cast(float, amor_dims['xlo'])
        amor_y = cast(float, amor_dims['yhi']) - cast(float, amor_dims['ylo'])
        amor_z = cast(float, amor_dims['zhi']) - cast(float, amor_dims['zlo'])

        if amor_x >= target_x and amor_y >= target_y and amor_z >= target_z:
            logger.info("Using existing amorphous file: %s", amor_path)
            return amor_path
        logger.info(
            "Existing amorphous file too small (%.1fx%.1fx%.1f vs %.1fx%.1fx%.1f), regenerating...",
            amor_x, amor_y, amor_z, target_x, target_y, target_z
        )

    logger.info("Generating amorphous %s via melt-quench...", mat_name)

    quench = settings.quench
    slab_x = max(quench.quench_slab_dims[0], target_x)
    slab_y = max(quench.quench_slab_dims[1], target_y)
    slab_z = max(quench.quench_slab_dims[2], target_z)

    if (slab_x > quench.quench_slab_dims[0] or
        slab_y > quench.quench_slab_dims[1] or
        slab_z > quench.quench_slab_dims[2]):
        logger.info(
            "Increasing slab dimensions from (%dx%dx%d) to required (%.1fx%.1fx%.1f)",
            quench.quench_slab_dims[0], quench.quench_slab_dims[1],
            quench.quench_slab_dims[2], slab_x, slab_y, slab_z
        )

    base_block = output_dir / f"{mat_name}_crystal_block.lmp"
    create_orthogonal_slab(
        cif_path, base_block,
        target_x=slab_x, target_y=slab_y, target_z=slab_z
    )

    mat_dir = resources.files('src.data.materials')
    materials_path = Path(str(mat_dir))
    materials_path.mkdir(parents=True, exist_ok=True)

    amor_file = materials_path / f"amor_{mat_name}.lmp"

    cif_data = cifread(cif_path)
    elements = cif_data['elements']

    temp_config = ComponentConfig(
        mat=mat_name,
        pot_type=pot_type,
        pot_path=str(pot_path),
        cif_path=str(cif_path)
    )

    pm = PotentialManager(settings, use_langevin=False)
    pot_commands = pm.get_single_component_commands(temp_config, elements)

    quench_rate = float(quench.quench_rate) * 1e-12 / float(quench.timestep)
    quench_nsteps = max(int((quench.melt_temp - quench.quench_temp) * quench_rate), 1)

    lmp_input_path = output_dir / f"lmp_input_amorphous_{mat_name}.in"

    jinja_env = Environment(
        loader=PackageLoader('src.templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    context = {
        'neighbor_list': settings.simulation.neighbor_list,
        'neigh_modify_command': settings.simulation.neigh_modify_command,
        'base_block': base_block,
        'pot_commands': pot_commands,
        'min_style': settings.simulation.min_style,
        'minimization_command': settings.simulation.minimization_command,
        'timestep': quench.timestep,
        'melt_temp': quench.melt_temp,
        'melt_steps': quench.melt_steps,
        'quench_temp': quench.quench_temp,
        'quench_nsteps': quench_nsteps,
        'equilibrate_steps': quench.equilibrate_steps,
        'amor_file': amor_file
    }

    template = jinja_env.get_template('common/make_amorphous.lmp')
    script_content = template.render(context)
    lmp_input_path.write_text(script_content, encoding='utf-8')

    if quench.run_local:
        logger.info("Running melt-quench with %d processors...", quench.n_procs)
        try:
            subprocess.run(
                f"mpiexec -np {quench.n_procs} lmp -in {lmp_input_path}",
                shell=True, check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error("Melt-quench LAMMPS run failed: %s", e)
            raise
    else:
        local_lmp_input = output_dir / f"make_amorphous_{mat_name}.in"
        shutil.copy(lmp_input_path, local_lmp_input)
        logger.info(
            "LAMMPS input file for amorphous structure generation saved to: %s",
            local_lmp_input
        )
        logger.info("Please run the following command to generate the amorphous structure:")
        logger.info("mpiexec -np %d lmp -in %s", quench.n_procs,
                    local_lmp_input)
        logger.info("After the simulation is complete, please rerun the program.")
        raise RuntimeError(
            f"Amorphous structure generation requires manual LAMMPS run. "
            f"See {local_lmp_input} for input script."
        )

    if base_block.exists():
        base_block.unlink()
    if lmp_input_path.exists():
        lmp_input_path.unlink()

    logger.info("Created amorphous structure in materials folder: %s", amor_file)
    return amor_file

def _create_base_slab(
    config,
    cif_path: Path,
    target_x: float,
    target_y: float,
    target_z: float,
    output_path: Path,
    build_dir: Path,
    settings: GlobalSettings
) -> None:
    """Create base slab (amorphous or crystalline).
    
    Args:
        config: TipConfig or SubstrateConfig.
        cif_path: CIF file path.
        target_x: Target X dimension (Angstroms).
        target_y: Target Y dimension (Angstroms).
        target_z: Target Z dimension (Angstroms).
        output_path: Output slab file path.
        build_dir: Directory for temporary files.
        settings: Global simulation settings.
    """
    if config.amorph == 'a':
        pot_path = get_material_path(config.pot_path)
        amor_path = make_amorphous(
            config.mat, cif_path,
            target_x=target_x, target_y=target_y, target_z=target_z,
            pot_path=pot_path, pot_type=config.pot_type,
            output_dir=build_dir, settings=settings
        )
        shutil.copy(amor_path, output_path)
    else:
        create_orthogonal_slab(
            cif_path, output_path,
            target_x=target_x, target_y=target_y, target_z=target_z
        )

def stack_multilayer_sheet(
    base_layer_path: Path,
    config: SheetConfig,
    output_path: Path,
    box_dims: Dict[str, float],
    n_layers: int,
    types_per_layer: int,
    pot_counts: Dict[str, int],
    supercell_dims: Dict[str, float],
    lat_c: Optional[float] = None,
    settings: Optional[GlobalSettings] = None,
    use_pair_bonding: bool = False,
    stacking_type: str = 'AB',
    edge_mode: str = 'none'
) -> float:
    """Stack multiple 2D material layers with proper atom type renumbering.
    
    Args:
        base_layer_path: Single-layer LAMMPS data file path.
        config: Sheet configuration.
        output_path: Stacked output file path.
        box_dims: Box dimensions from base layer.
        n_layers: Number of layers to stack.
        types_per_layer: Atom types per layer before renumbering.
        pot_counts: Element to type count mapping.
        supercell_dims: Original supercell dimensions for calculating shifts.
        lat_c: Interlayer distance (calculated if None).
        settings: Global settings (required if lat_c is None).
        use_pair_bonding: If True, layers are bonded in pairs (L1-L2, L3-L4)
            with no relative shift within pairs. Used for sheetonsheet simulations.
        stacking_type: Stacking type ('AA' or 'AB'). AA has no shifts, AB has shifts.
    Returns:
        float: Interlayer distance (lat_c).
    """
    if n_layers < 2:
        raise ValueError("stack_multilayer_sheet requires at least 2 layers")

    calculate_lat_c = lat_c is None

    if calculate_lat_c:
        assert settings is not None
        logger.info("Running minimization to find optimal interlayer distance")
        initial_guess = settings.geometry.lat_c_default
    else:
        assert settings is not None
        initial_guess = lat_c

    layers_for_setup = 2 if calculate_lat_c else n_layers
    pot_file = Path(tempfile.gettempdir()) / f"sheet_{layers_for_setup}_{uuid.uuid4().hex[:8]}.in.settings"

    pm = PotentialManager(settings)
    regions = [None, 'edge'] if edge_mode != 'none' else None
    pm.register_component("sheet", config, n_layers=layers_for_setup, regions=regions)
    pm.add_self_interaction("sheet")

    if pm.is_sheet_lj(config.pot_type):
        pm.add_interlayer_interaction("sheet")

    pm.write_file(pot_file)

    total_types = len(pm.types)
    max_z = initial_guess * (layers_for_setup - 1) + 20.0

    jinja_env = Environment(
        loader=PackageLoader('src.templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    layer_shifts = [initial_guess * l for l in range(layers_for_setup)]

    per_layer_shifts = calculate_layer_shifts(
        config.mat, supercell_dims, n_layers=layers_for_setup,
        use_pair_bonding=use_pair_bonding, stacking_type=stacking_type
    )
    layer_shifts_x = [shift[0] for shift in per_layer_shifts]
    layer_shifts_y = [shift[1] for shift in per_layer_shifts]

    atom_style = 'charge' if config.pot_type in ('reaxff', 'reax/c') else 'atomic'

    context = {
        'box_xlo': box_dims['xlo'],
        'box_xhi': box_dims['xhi'],
        'box_ylo': box_dims['ylo'],
        'box_yhi': box_dims['yhi'],
        'box_zhi': max_z,
        'total_types': total_types,
        'atom_style': atom_style,
        'base_layer_path': base_layer_path,
        'n_layers': layers_for_setup,
        'layer_shifts': layer_shifts,
        'layer_shifts_x': layer_shifts_x,
        'layer_shifts_y': layer_shifts_y,
        'types_per_layer': types_per_layer,
        'pot_counts': pot_counts,
        'pot_file': pot_file,
        'pot_type': config.pot_type.lower(),
        'calculate_lat_c': calculate_lat_c,
        'min_style': settings.simulation.min_style if calculate_lat_c else None,
        'minimization_command': settings.simulation.minimization_command if calculate_lat_c else None,
        'timestep': settings.simulation.timestep if calculate_lat_c else None,
        'output_path': output_path
    }

    template = jinja_env.get_template('common/stack_layers.lmp')
    commands = template.render(context).strip().split('\n')
    commands = [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith('#')]

    lmp = _create_lammps_instance()
    try:
        for cmd in commands:
            lmp.command(cmd)

        if calculate_lat_c:
            extracted = lmp.extract_variable("lat_c", None, 0)
            lat_c = float(extracted) if extracted is not None else initial_guess
            logger.info("Found optimal lat_c: %.4f", lat_c)

            if n_layers > 2:
                lmp.close()
                return stack_multilayer_sheet(
                    base_layer_path=base_layer_path,
                    config=config,
                    output_path=output_path,
                    box_dims=box_dims,
                    n_layers=n_layers,
                    types_per_layer=types_per_layer,
                    pot_counts=pot_counts,
                    supercell_dims=supercell_dims,
                    lat_c=lat_c,
                    settings=settings,
                    use_pair_bonding=use_pair_bonding,
                    stacking_type=stacking_type,
                    edge_mode=edge_mode
                )

        lat_c = lat_c or initial_guess

    except Exception as e:
        logger.error("Failed to stack layers: %s", e)
        if calculate_lat_c:
            lat_c = settings.geometry.lat_c_default
        raise
    finally:
        lmp.close()
        if pot_file.exists():
            pot_file.unlink()

    # Re-anchor the stacked sheet so its lowest atom is at z=0.
    # Without this, some stacking/minimization outcomes can leave atoms below
    # zero, which reduces the intended sheet-substrate separation in AFM setup.
    if output_path.exists():
        shift_atoms_to_z_zero(output_path)

    return lat_c

def build_tip(
    config: TipConfig,
    build_dir: Path,
    settings: GlobalSettings
) -> Tuple[Path, float]:
    """Build AFM tip structure (spherical geometry).
    
    Args:
        config: Tip configuration.
        build_dir: Output directory.
        settings: Global simulation settings.
        
    Returns:
        Tuple of (tip_path, actual_radius).
    """
    cif_path = get_material_path(config.cif_path)

    base_lmp = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_base.lmp"
    final_lmp = build_dir / "tip.lmp"
    radius = config.r
    box_size = radius * 2.0

    _create_base_slab(
        config, cif_path,
        target_x=box_size, target_y=box_size, target_z=box_size,
        output_path=base_lmp, build_dir=build_dir,
        settings=settings
    )

    if config.amorph != 'a':
        dim = get_model_dimensions(base_lmp)
        assert all(dim[k] is not None for k in ['xhi', 'xlo', 'yhi', 'ylo'])
        radius = min(cast(float, dim['xhi']) - cast(float, dim['xlo']),
                    cast(float, dim['yhi']) - cast(float, dim['ylo'])) / 2.0

    h = radius / settings.geometry.tip_reduction_factor

    jinja_env = Environment(
        loader=PackageLoader('src.templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    context = {
        'base_lmp': base_lmp,
        'radius': radius,
        'tip_height': h,
        'output_path': final_lmp
    }

    template = jinja_env.get_template('common/build_tip.lmp')
    commands = template.render(context).strip().split('\n')
    commands = [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith('#')]

    run_lammps_commands(commands)
    base_lmp.unlink()
    return final_lmp, radius

def build_substrate(
    config: SubstrateConfig,
    build_dir: Path,
    box_dims: dict,
    settings: Optional[GlobalSettings] = None,
    target_x: Optional[float] = None,
    target_y: Optional[float] = None,
    ) -> Path:
    """Build substrate slab.
    
    Args:
        config: Substrate configuration.
        build_dir: Output directory.
        box_dims: Reference box dimensions from sheet.
        settings: Global settings (required for amorphous).
        target_x: Optional explicit substrate size in x.
        target_y: Optional explicit substrate size in y.
        
    Returns:
        Path to substrate file.
    """

    cif_path = get_material_path(config.cif_path)
    base_lmp = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_base.lmp"
    final_lmp = build_dir / "sub.lmp"
    target_x = float(target_x) if target_x is not None else box_dims['xhi'] - box_dims['xlo']
    target_y = float(target_y) if target_y is not None else box_dims['yhi'] - box_dims['ylo']
    target_z = config.thickness

    _create_base_slab(
        config, cif_path,
        target_x=target_x, target_y=target_y, target_z=target_z,
        output_path=base_lmp, build_dir=build_dir,
        settings=settings or GlobalSettings()
    )

    # Preserve periodic tiling for crystalline substrates by using the
    # commensurate replicated slab span in x/y instead of forcing a non-
    # commensurate crop to the requested target.
    if config.amorph == 'a':
        box_x_span = target_x
        box_y_span = target_y
    else:
        base_dims = get_model_dimensions(base_lmp)
        assert all(base_dims[k] is not None for k in ['xlo', 'xhi', 'ylo', 'yhi'])
        box_x_span = cast(float, base_dims['xhi']) - cast(float, base_dims['xlo'])
        box_y_span = cast(float, base_dims['yhi']) - cast(float, base_dims['ylo'])

    jinja_env = Environment(
        loader=PackageLoader('src.templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    context = {
        'base_lmp': base_lmp,
        'box_xlo': box_dims['xlo'],
        'box_xhi': box_dims['xlo'] + box_x_span,
        'box_ylo': box_dims['ylo'],
        'box_yhi': box_dims['ylo'] + box_y_span,
        'box_zlo': box_dims['zlo'],
        'target_z': target_z,
        'output_path': final_lmp
    }

    template = jinja_env.get_template('common/build_substrate.lmp')
    commands = template.render(context).strip().split('\n')
    commands = [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith('#')]

    run_lammps_commands(commands)
    base_lmp.unlink()
    return final_lmp

def apply_langevin_regions(
    component_name: str,
    component_path: Path,
    config: ComponentConfig,
    settings: GlobalSettings,
    component_height: float,
    substrate_thickness: Optional[float] = None
) -> Path:
    """Apply Langevin thermostat region splitting to a component.
    
    Splits a component (tip or substrate) into three spatial regions:
    - fix: Bottom region (fixed atoms)
    - thermo: Middle region (thermalized atoms)
    - normal: Top region (mobile atoms)
    
    Each original atom type gets expanded to 3 types (normal, fix, thermo).
    Generates appropriate mass and potential commands.
    
    Args:
        component_name: Name of component ('tip' or 'sub').
        component_path: Path to the component LAMMPS data file.
        config: Component configuration with potential info.
        settings: Global simulation settings with langevin_boundaries.
        component_height: Height of component (tip_height or substrate thickness).
        substrate_thickness: Required if component_name='sub' to calculate boundaries.
        
    Returns:
        Path to the modified component file with 3-region types.
    """
    num_types = get_num_atom_types(component_path)

    boundaries = {}
    if component_name == 'tip':
        h = component_height
        bounds_config = settings.thermostat.langevin_boundaries['tip']
        boundaries['f_zlo'] = h - bounds_config['fix'][0]
        boundaries['f_zhi'] = h - bounds_config['fix'][1]
        boundaries['t_zlo'] = h - bounds_config['thermo'][0]
        boundaries['t_zhi'] = h - bounds_config['thermo'][1]
        boundaries['n_zhi'] = h
    elif component_name == 'sub':
        if substrate_thickness is None:
            raise ValueError("substrate_thickness required for substrate component")
        bounds_config = settings.thermostat.langevin_boundaries['sub']
        boundaries['f_zlo'] = bounds_config['fix'][0] * substrate_thickness
        boundaries['f_zhi'] = bounds_config['fix'][1] * substrate_thickness
        boundaries['t_zlo'] = bounds_config['thermo'][0] * substrate_thickness
        boundaries['t_zhi'] = bounds_config['thermo'][1] * substrate_thickness
        boundaries['n_zhi'] = substrate_thickness
    else:
        raise ValueError(f"Unknown component name: {component_name}")

    pm = PotentialManager(settings, use_langevin=True)
    pm.register_component(component_name, config)
    base_elements = pm.components[component_name]['elements']
    expanded_elements = [elem for elem in base_elements for _ in range(3)]
    potential_commands = pm.get_single_component_commands(config, expanded_elements)

    jinja_env = Environment(
        loader=PackageLoader('src.templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    context = {
        'component_name': component_name,
        'input_path': component_path,
        'num_types': num_types,
        'boundaries': boundaries,
        'potential_commands': potential_commands
    }

    template = jinja_env.get_template('common/apply_langevin_regions.lmp')
    commands = template.render(context).strip().split('\n')
    commands = [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith('#')]

    run_lammps_commands(commands)

    return component_path


def apply_sheet_edge_types(component_path: Path, edge_width: float) -> Path:
    """Split a built sheet into core and edge atom types."""
    if edge_width <= 0:
        return component_path

    num_types = get_num_atom_types(component_path)
    if num_types < 1:
        return component_path

    jinja_env = Environment(
        loader=PackageLoader('src.templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    context = {
        'input_path': component_path,
        'num_types': num_types,
        'edge_width': edge_width,
    }

    template = jinja_env.get_template('common/apply_sheet_edge_types.lmp')
    commands = template.render(context).strip().split('\n')
    commands = [cmd.strip() for cmd in commands if cmd.strip() and not cmd.strip().startswith('#')]

    run_lammps_commands(commands)

    return component_path

def build_monolayer(
    config: SheetConfig,
    edge_mode: str = 'none',
    edge_width: float = 0.0,
) -> Tuple[Path, dict, dict, int, Dict[str, float]]:
    """Build single-layer 2D material sheet.
    
    Workflow:
    1. Read CIF structure and validate against potential.
    2. Write as LAMMPS data with types matching CIF element order.
    3. Renumber types to match potential file expectations (if multi-type).
    4. Duplicate and orthogonalize as needed.
    
    Args:
        config: Sheet configuration.
        
    Returns:
        Tuple of (path, box_dims, pot_counts, total_types, supercell_dims).
        
    Raises:
        ValueError: If CIF elements do not match potential file expectations.
    """
    cif_path = get_material_path(config.cif_path)
    pot_path = get_material_path(config.pot_path)
    base_path = Path(tempfile.gettempdir()) / f"{config.mat}_1_{uuid.uuid4().hex[:8]}.lmp"

    cif_data = cifread(cif_path)
    cif_elements = cif_data['elements']

    pot_element_order, pot_element_counts = get_potential_element_order(
        pot_path, config.pot_type, cif_elements
    )
    pot_counts = pot_element_counts
    total_pot_types = sum(pot_counts.values())

    logger.debug(
        "Potential mapping: CIF elements %s -> %d types per element %s",
        cif_elements, total_pot_types, pot_counts
    )

    type_multiplier = 1 if config.pot_type in ['rebo', 'rebomos', 'airebo', 'meam', 'reaxff'] else \
        check_potential_cif_compatibility(cif_path, pot_path)

    temp_unit_cell = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_unit.lmp"
    _write_cif_as_lammps_atomic(cif_path, temp_unit_cell, specorder=cif_elements)

    has_sublattice_types = any(v != 1 for v in pot_counts.values())

    # When the primitive already holds exactly `total_pot_types` atoms
    # (type_multiplier == 1, e.g. t-PtSe2's Pt/Se1/Se2), assign the potential
    # sublattice types on the PRIMITIVE, then let the (type-preserving)
    # orthogonalization tile them across the supercell. Assigning them *after*
    # orthogonalization instead produced one unique type per atom, because
    # orthogonalization multiplies the atom count (the pre-fix failure mode).
    assign_on_primitive = has_sublattice_types and type_multiplier == 1
    if assign_on_primitive:
        _assign_pot_types_by_order(temp_unit_cell, pot_element_order, pot_counts)

    # Orthogonalize. _orthogonalize_lammps_data preserves each atom's explicit
    # LAMMPS type, so sublattice types assigned above survive the tiling.
    ortho_cell = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_ortho.lmp"
    _orthogonalize_lammps_data(temp_unit_cell, ortho_cell)
    temp_unit_cell.unlink()

    # Types are assigned either on the primitive (type_multiplier == 1, above) or
    # on the duplicated potential cell (type_multiplier > 1, below), so the
    # orthogonalized element-typed cell needs no post-ortho renumber here.

    dims = get_model_dimensions(ortho_cell)

    if type_multiplier != 1:
        # The potential defines `total_pot_types` types for a k× supercell of the
        # primitive (e.g. p-SnS: 8 types on a 2× cell; h-MoS2: 12 on a 2×2 cell).
        # Duplicate the orthogonal cell to exactly `total_pot_types` atoms, then
        # assign the sublattice types by element order (each element's atoms take
        # that element's block of type ids). Counting atoms via a manual parse
        # avoids ase.io.read's StopIteration on LAMMPS write_data output
        # (Velocities section + image flags).
        num_atoms = len(_parse_lammps_cell_atoms_masses(ortho_cell.read_text(encoding='utf-8'))[1])
        if num_atoms and total_pot_types % num_atoms == 0:
            factor = total_pot_types // num_atoms
            dup_factor_x, dup_factor_y = 1, 1
            for factor_candidate in range(int(np.sqrt(factor)) + 1, 0, -1):
                if factor % factor_candidate == 0:
                    dup_factor_x = factor_candidate
                    dup_factor_y = factor // factor_candidate
                    break
            dup_ortho = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_dup.lmp"
            _duplicate_lammps_data(ortho_cell, dup_ortho, dup_factor_x, dup_factor_y, 1)
            ortho_cell.unlink()
            ortho_cell = dup_ortho
            _assign_pot_types_by_order(ortho_cell, pot_element_order, pot_counts)
            dims = get_model_dimensions(ortho_cell)

    dims_calc = get_model_dimensions(ortho_cell)
    dims_calc_typed: Dict[str, float] = {k: cast(float, v) for k, v in dims_calc.items()}

    supercell_dims = dims_calc_typed.copy()

    unit_x = cast(float, dims_calc_typed['xhi']) - cast(float, dims_calc_typed['xlo'])
    unit_y = cast(float, dims_calc_typed['yhi']) - cast(float, dims_calc_typed['ylo'])

    target_x = config.x[0] if isinstance(config.x, list) else config.x
    target_y = config.y[0] if isinstance(config.y, list) else config.y
    dup_x = max(1, round(target_x / unit_x))
    dup_y = max(1, round(target_y / unit_y))

    if dup_x == 1 and dup_y == 1:
        shutil.copy(ortho_cell, base_path)
    else:
        _duplicate_lammps_data(ortho_cell, base_path, dup_x, dup_y, 1)
    ortho_cell.unlink()

    dims = get_model_dimensions(base_path)

    shift_atoms_to_z_zero(base_path)
    dims = get_model_dimensions(base_path)

    edge_enabled = edge_mode != 'none'
    if edge_enabled:
        apply_sheet_edge_types(base_path, edge_width=edge_width)

    if config.pot_type in ['tersoff', 'sw', 'rebo', 'airebo']:
        charge2atom(base_path)
    elif config.pot_type in ['reaxff', 'reax/c']:
        atomic2charge(base_path)

    if edge_enabled:
        total_pot_types *= 2

    observed_types = get_num_atom_types(base_path)
    if observed_types != total_pot_types:
        raise ValueError(
            f"Type count mismatch after build_monolayer: expected {total_pot_types} "
            f"(from potential) but observed {observed_types} in final structure. "
            f"Potential mapping: {pot_counts}"
        )

    logger.info(
        "Built %s monolayer: %d atom types, mapping %s",
        config.mat, total_pot_types, pot_counts
    )

    return base_path, dims, pot_counts, total_pot_types, supercell_dims

def build_sheet(
    config: SheetConfig,
    build_dir: Path,
    stack_if_multi: bool = False,
    settings: Optional[GlobalSettings] = None,
    n_layers_override: Optional[int] = None,
    use_pair_bonding: bool = False,
    stacking_type: str = 'AB',
    edge_mode: str = 'none',
    edge_width: float = 0.0,
    unit_cell_out: Optional[Dict[str, float]] = None,
) -> Tuple[Path, dict, Optional[float]]:
    """Build 2D material sheet (single or multi-layer).

    Args:
        config: Sheet configuration.
        build_dir: Output directory.
        stack_if_multi: If True, stack multiple layers.
        settings: Global settings (required for multi-layer).
        n_layers_override: Override layer count.
        use_pair_bonding: If True, use pair bonding stacking (for sheetonsheet).
        stacking_type: Stacking type ('AA' or 'AB'). AA has no shifts, AB has shifts.
        unit_cell_out: Optional mutable dict populated in-place with the
            orthogonalized single-unit-cell bounds (xlo/xhi/ylo/yhi/...). Used by
            PES scans to size the lateral grid to one surface lattice period.

    Returns:
        Tuple of (path, box_dims, lat_c).
    """
    base_path, dims, pot_counts, total_pot_types, supercell_dims = build_monolayer(
        config,
        edge_mode=edge_mode,
        edge_width=edge_width,
    )
    if unit_cell_out is not None:
        unit_cell_out.update({k: float(v) for k, v in supercell_dims.items()})

    n_layers = n_layers_override or (max(config.layers) if config.layers else 1)
    stacked_path = build_dir / f"{config.mat}_{n_layers}.lmp"
    lat_c = config.lat_c
    if not stack_if_multi:
        shutil.copy(base_path, stacked_path)
        return stacked_path, dims, lat_c

    if stack_if_multi and n_layers > 1:
        lat_c = stack_multilayer_sheet(
            base_layer_path=base_path,
            config=config,
            output_path=stacked_path,
            box_dims=dims,
            n_layers=n_layers,
            types_per_layer=total_pot_types,
            pot_counts=pot_counts,
            supercell_dims=supercell_dims,
            lat_c=lat_c,
            settings=settings,
            use_pair_bonding=use_pair_bonding,
            stacking_type=stacking_type,
            edge_mode=edge_mode,
        )
        return stacked_path, dims, lat_c

    if n_layers == 1:
        shutil.copy(base_path, stacked_path)
    return stacked_path, dims, lat_c
