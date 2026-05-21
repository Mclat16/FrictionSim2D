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
import subprocess
import numpy as np

from ase import io as ase_io
from jinja2 import Environment

from ..core.config import GlobalSettings, SheetConfig, SubstrateConfig, TipConfig, ComponentConfig
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

    # Triclinic box: use ASE to build a minimum orthogonal supercell.
    # Read the structure; preserve specorder from the Masses section.
    masses_lines = []
    in_masses = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == 'Masses':
            in_masses = True
            continue
        if in_masses:
            if stripped == '' or stripped.startswith('Atoms'):
                if stripped.startswith('Atoms'):
                    in_masses = False
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[0].isdigit():
                masses_lines.append(stripped)

    specorder = None
    if masses_lines:
        specorder = []
        for mline in masses_lines:
            parts = mline.split()
            if '#' in mline:
                specorder.append(mline.split('#')[-1].strip())

    atoms = ase_io.read(str(input_path), format='lammps-data',
                        atom_style='atomic', units='metal',
                        Z_of_type={i + 1: _element_z(el)
                                   for i, el in enumerate(specorder or [])})
    if isinstance(atoms, list):
        atoms = atoms[0]

    # Build a minimum orthogonal (no-tilt) supercell.
    # For hexagonal cells, [[1,0,0],[1,2,0],[0,0,1]] gives an orthorhombic 2× cell.
    cell = atoms.get_cell()
    transform_matrix = _find_ortho_transform(cell)
    ortho_atoms = _make_supercell(atoms, transform_matrix)

    write_kwargs: dict = {'filename': str(output_path), 'images': ortho_atoms,
                          'format': 'lammps-data', 'atom_style': 'atomic'}
    if specorder:
        write_kwargs['specorder'] = specorder
    ase_io.write(**write_kwargs)

    # Re-inject Masses section (ASE omits it)
    if specorder:
        from ase.data import atomic_masses as _am, atomic_numbers as _an  # pylint: disable=import-outside-toplevel
        masses_block = "Masses\n\n"
        for idx, element in enumerate(specorder, start=1):
            mass = _am[_an[element]]
            masses_block += f"{idx} {mass:.6f}  # {element}\n"
        masses_block += "\n"
        out_content = output_path.read_text(encoding='utf-8')
        out_content = _re.sub(r'(Atoms\s)', masses_block + r'\1', out_content, count=1)
        output_path.write_text(out_content, encoding='utf-8')


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

    # Search for n1, n2 integers (smallest |det|) such that n1*a1 + n2*a2 is along y.
    best_transform = None
    best_det = None
    search_range = range(-6, 7)
    for n1 in search_range:
        for n2 in search_range:
            if n1 == 0 and n2 == 0:
                continue
            trial = n1 * a1 + n2 * a2
            if abs(trial[0]) < 1e-6 and abs(trial[1]) > 1e-3:
                if best_det is None or abs(n2) < best_det:
                    best_det = abs(n2)
                    best_transform = _np.array([[1, 0, 0], [n1, n2, 0], [0, 0, 1]], dtype=int)

    if best_transform is not None:
        return best_transform

    # Fallback: identity (leave as-is)
    return _np.eye(3, dtype=int)



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
    stacking_type: str = 'AB'
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
    pm.register_component("sheet", config, n_layers=layers_for_setup)
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
                    stacking_type=stacking_type
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
    settings: Optional[GlobalSettings] = None
    ) -> Path:
    """Build substrate slab.
    
    Args:
        config: Substrate configuration.
        build_dir: Output directory.
        box_dims: Target box dimensions from sheet.
        settings: Global settings (required for amorphous).
        
    Returns:
        Path to substrate file.
    """

    cif_path = get_material_path(config.cif_path)
    base_lmp = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_base.lmp"
    final_lmp = build_dir / "sub.lmp"
    target_x = box_dims['xhi'] - box_dims['xlo']
    target_y = box_dims['yhi'] - box_dims['ylo']
    target_z = config.thickness

    _create_base_slab(
        config, cif_path,
        target_x=target_x, target_y=target_y, target_z=target_z,
        output_path=base_lmp, build_dir=build_dir,
        settings=settings or GlobalSettings()
    )

    jinja_env = Environment(
        loader=PackageLoader('src.templates'),
        trim_blocks=True,
        lstrip_blocks=True
    )

    context = {
        'base_lmp': base_lmp,
        'box_xlo': box_dims['xlo'],
        'box_xhi': box_dims['xhi'],
        'box_ylo': box_dims['ylo'],
        'box_yhi': box_dims['yhi'],
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

def build_monolayer(
    config: SheetConfig,
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

    # Orthogonalize FIRST (while the file has only the simple CIF element types),
    # then renumber.  Renumbering after orthogonalization avoids confusion in the
    # ASE-based tilt removal when types outnumber distinct chemical elements.
    ortho_cell = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_ortho.lmp"
    _orthogonalize_lammps_data(temp_unit_cell, ortho_cell)
    temp_unit_cell.unlink()

    if any(v != 1 for v in pot_counts.values()) or type_multiplier != 1:
        renumber_atom_types(ortho_cell, pot=pot_element_order)

    dims = get_model_dimensions(ortho_cell)

    if type_multiplier != 1:
        atoms = ase_io.read(str(ortho_cell), format="lammps-data")
        num_atoms = len(atoms)
        dup_factor_x = 1
        dup_factor_y = 1
        if total_pot_types % num_atoms == 0:
            for factor_candidate in range(int(np.sqrt(total_pot_types / num_atoms)) + 1, 0, -1):
                if (total_pot_types / num_atoms) % factor_candidate == 0:
                    dup_factor_x = int(factor_candidate)
                    dup_factor_y = int(total_pot_types / num_atoms / factor_candidate)
                    break
            # Capture type→element mapping BEFORE duplication (# comments still present)
            type_elem_map: Dict[int, str] = {}
            for raw_line in ortho_cell.read_text(encoding='utf-8').splitlines():
                stripped_line = raw_line.strip()
                if '#' in stripped_line and stripped_line and stripped_line[0].isdigit():
                    line_parts = stripped_line.split()
                    try:
                        type_id = int(line_parts[0])
                        element_symbol = stripped_line.split('#')[-1].strip().split()[0]
                        type_elem_map[type_id] = element_symbol
                    except (ValueError, IndexError):
                        pass
            dup_ortho = Path(tempfile.gettempdir()) / f"{config.mat}_{uuid.uuid4().hex[:8]}_dup.lmp"
            _duplicate_lammps_data(ortho_cell, dup_ortho, dup_factor_x, dup_factor_y, 1)
            ortho_cell.unlink()
            ortho_cell = dup_ortho
            # Strip velocities and renumber using explicit type->element map.
            # This avoids depending on ``# element`` Masses comments after
            # LAMMPS write_data strips them.
            _strip_velocities(ortho_cell)
            renumber_atom_types(
                ortho_cell,
                pot=list(pot_counts.keys()),
                type_to_element_map=type_elem_map
            )
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

    if config.pot_type in ['tersoff', 'sw', 'rebo', 'airebo']:
        charge2atom(base_path)
    elif config.pot_type in ['reaxff', 'reax/c']:
        atomic2charge(base_path)

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
    stacking_type: str = 'AB'
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
        
    Returns:
        Tuple of (path, box_dims, lat_c).
    """
    base_path, dims, pot_counts, total_pot_types, supercell_dims = build_monolayer(
        config
    )

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
            stacking_type=stacking_type
        )
        return stacked_path, dims, lat_c

    if n_layers == 1:
        shutil.copy(base_path, stacked_path)
    return stacked_path, dims, lat_c
