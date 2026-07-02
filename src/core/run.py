"""Shared simulation runner utilities.

Provides config expansion, shared simulation-root layout, and HPC script
generation used by both the CLI and Python scripts.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import itertools
import re

from .config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_settings,
    parse_config,
)
from .path_utils import format_dimension_token
from ..builders.afm import AFMSimulation
from ..builders.sheetonsheet import SheetOnSheetSimulation
from ..hpc import HPCScriptGenerator, HPCConfig

logger = logging.getLogger(__name__)


def layer_aware_path_sort_key(path_str: str) -> Tuple[Tuple[int, Union[int, str]], ...]:
    """Sort paths naturally for layer segments like L1, L2, L10."""
    parts = Path(path_str).parts
    key_parts: List[Tuple[int, Union[int, str]]] = []
    for part in parts:
        match = re.fullmatch(r"L(\d+)", part)
        if match:
            key_parts.append((0, int(match.group(1))))
        else:
            key_parts.append((1, part))
    return tuple(key_parts)


def _as_float_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _is_ascending(values: List[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def _is_subsequence(subseq: List[float], seq: List[float], tol: float = 1.0e-12) -> bool:
    if not subseq:
        return True
    idx = 0
    for value in seq:
        if abs(value - subseq[idx]) <= tol:
            idx += 1
            if idx == len(subseq):
                return True
    return False


def _validate_runtime_sweep_ordering(general_cfg: Dict[str, Any]) -> None:
    """Validate sweep ordering in runtime path (CLI/script flow)."""
    for field_name in ('force', 'pressure', 'scan_angle', 'scan_angle_force'):
        values = _as_float_list(general_cfg.get(field_name))
        if len(values) > 1 and not _is_ascending(values):
            raise ValueError(f"{field_name} must be in ascending order")

    selector = _as_float_list(general_cfg.get('scan_angle_force'))
    if len(selector) <= 1:
        return

    force_values = _as_float_list(general_cfg.get('force'))
    pressure_values = _as_float_list(general_cfg.get('pressure'))
    target_field = 'force' if force_values else 'pressure'
    target_values = force_values if force_values else pressure_values

    if not target_values:
        raise ValueError("scan_angle_force list requires force or pressure values")

    if not _is_subsequence(selector, target_values):
        raise ValueError(
            f"scan_angle_force must follow the same order as {target_field} values"
        )


def expand_config_sweeps(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand config with material lists and parameter sweeps."""
    sweep_params = {}

    def resolve_materials_list(section_config: Dict[str, Any]) -> List[str]:
        if 'materials_list' not in section_config:
            return []

        mat_list_path = section_config.get('materials_list', '')
        if isinstance(mat_list_path, str) and mat_list_path.endswith(".txt"):
            path = Path(mat_list_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            logger.warning("Material list file %s not found.", path)
        return []

    def expand_mat_template(section_config: Dict[str, Any], mat_value: str) -> Dict[str, Any]:
        expanded = {}
        for key, val in section_config.items():
            if key == 'materials_list':
                continue
            if isinstance(val, str) and '{mat}' in val:
                expanded[key] = val.replace('{mat}', mat_value)
            else:
                expanded[key] = val
        return expanded

    if '2D' in base_config:
        materials = resolve_materials_list(base_config['2D'])
        if materials:
            sweep_params[('2D', '_material_expand')] = materials

    lammps_loop_params = {'force', 'scan_angle', 'pressure', 'scan_speed'}

    if 'general' in base_config:
        for key, val in base_config['general'].items():
            if isinstance(val, list) and key not in lammps_loop_params:
                sweep_params[('general', key)] = val

    if not sweep_params:
        return [base_config]

    keys, values = zip(*sweep_params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    expanded_configs = []
    for perm in permutations:
        new_conf = deepcopy(base_config)

        mat_value = perm.pop(('2D', '_material_expand'), None)
        if mat_value and '2D' in new_conf:
            new_conf['2D'] = expand_mat_template(new_conf['2D'], mat_value)

        for (section, key), val in perm.items():
            new_conf[section][key] = val

        expanded_configs.append(new_conf)

    return expanded_configs


def collect_hpc_simulation_paths(
    simulation_root: Path,
    lammps_scripts: Optional[List[str]] = None,
) -> List[str]:
    """Collect simulation paths relative to a shared simulation root."""
    scripts = ['system.in', 'slide.in'] if lammps_scripts is None else lammps_scripts

    def script_exists(lammps_dir: Path, script_name: str) -> bool:
        if '*' in script_name or '?' in script_name:
            return any(lammps_dir.glob(script_name))
        if script_name == 'system.in':
            return (
                (lammps_dir / 'system.in').exists()
                or any(lammps_dir.glob('*system*.in'))
            )
        if script_name == 'slide.in':
            return (
                (lammps_dir / 'slide.in').exists()
                or any(lammps_dir.glob('*slide*.in'))
            )
        return (lammps_dir / script_name).exists()

    simulation_paths = []
    for lammps_dir in simulation_root.rglob('lammps'):
        if not lammps_dir.is_dir():
            continue

        if scripts:
            if not any(script_exists(lammps_dir, script) for script in scripts):
                continue
        elif not list(lammps_dir.glob('*.in')):
            continue

        rel_path = lammps_dir.parent.relative_to(simulation_root)
        simulation_paths.append(str(rel_path))
    return sorted(set(simulation_paths), key=layer_aware_path_sort_key)


def _build_hpc_manifest_entries(
    simulation_root: Path,
    simulation_paths: List[str],
    configured_scripts: List[str],
) -> Tuple[List[str], List[str]]:
    """Build manifest rows as full script paths discovered from built outputs.

    Each row is a full script path relative to ``simulation_root`` such as
    ``sheetonsheet/.../lammps/slide_p1gpa.in``.

    Discovery strategy per simulation directory:
        - If any ``*slide*.in`` files exist, add all of them (array-per-slide).
        - Otherwise add all ``*system*.in`` files.
    """
    manifest_entries: List[str] = []

    for sim_rel in simulation_paths:
        lammps_dir = simulation_root / sim_rel / 'lammps'
        if not lammps_dir.exists():
            continue

        system_scripts = sorted(
            p.relative_to(simulation_root).as_posix()
            for p in lammps_dir.glob('*.in')
            if 'system' in p.name
        )
        slide_scripts = sorted(
            p.relative_to(simulation_root).as_posix()
            for p in lammps_dir.glob('*.in')
            if 'slide' in p.name
        )

        if slide_scripts:
            manifest_entries.extend(slide_scripts)
        elif system_scripts:
            manifest_entries.extend(system_scripts)

    if manifest_entries:
        return manifest_entries, ['__MANIFEST_PATH__']

    return simulation_paths, configured_scripts


def dispatch_hpc_generation(
    generator: 'HPCScriptGenerator',
    simulation_root: Path,
    output_dir: Path,
    scheduler: str,
    base_dir: str,
    manifest_entries: List[str],
):
    """Generate HPC scripts, choosing combined, two-phase, or single-phase layout.

    AFM simulations have one system.in and one slide.in per simulation
    directory. When both phases are present in equal numbers (1:1 per sim dir)
    we emit a single combined job that runs system then slide sequentially —
    no PBS dependency needed. When counts differ (unusual) we fall back to two
    dependent array jobs. Sheet-on-sheet simulations have only slide scripts
    and use a single array job.

    Returns whatever the chosen generator method returns.
    """
    from ..hpc.manifest import JobManifest

    manifest = JobManifest.from_simulation_directory(simulation_root)
    if manifest.has_system_jobs():
        system_jobs = manifest.get_system_jobs()
        slide_jobs = manifest.get_slide_jobs()
        if len(system_jobs) == len(slide_jobs):
            logger.info(
                "System+slide jobs detected (1:1) — generating combined HPC script."
            )
            return generator.generate_combined_scripts(
                manifest, output_dir, scheduler=scheduler, base_dir=base_dir
            )
        logger.info("System/slide counts differ — generating two-phase HPC scripts.")
        return generator.generate_two_phase_scripts(
            manifest, output_dir, scheduler=scheduler, base_dir=base_dir
        )
    return generator.generate_scripts(
        simulation_paths=manifest_entries,
        output_dir=output_dir,
        scheduler=scheduler,
        base_dir=base_dir,
    )


def generate_hpc_scripts_for_root(simulation_root: Path, settings) -> None:
    """Generate shared HPC scripts for a simulation root directory."""

    hpc_config = HPCConfig.from_settings(settings.hpc, job_name=simulation_root.name)
    # Skip only when required HPC settings were not provided.
    if not hpc_config.modules:
        logger.warning(
            "Skipping HPC script generation: no HPC modules configured in settings.hpc.modules."
        )
        return

    simulation_paths = collect_hpc_simulation_paths(
        simulation_root,
        lammps_scripts=hpc_config.lammps_scripts,
    )
    if not simulation_paths:
        logger.warning("No simulations found for HPC script generation.")
        return

    manifest_entries, resolved_scripts = _build_hpc_manifest_entries(
        simulation_root,
        simulation_paths,
        hpc_config.lammps_scripts,
    )
    hpc_config.lammps_scripts = resolved_scripts

    hpc_dir = simulation_root / 'hpc'
    hpc_dir.mkdir(parents=True, exist_ok=True)

    if hpc_config.hpc_home:
        base_dir = hpc_config.hpc_home.rstrip('/') + '/' + simulation_root.name
    elif hpc_config.scheduler_type == 'pbs':
        base_dir = '$PBS_O_WORKDIR'
    else:
        base_dir = '$SLURM_SUBMIT_DIR'
    generator = HPCScriptGenerator(hpc_config)
    dispatch_hpc_generation(
        generator,
        simulation_root,
        hpc_dir,
        hpc_config.scheduler_type,
        base_dir,
        manifest_entries,
    )


def run_simulations(
    config_file: str,
    model: str,
    settings_file: Optional[Union[str, Path]] = None,
    output_root: Optional[Union[str, Path]] = None,
    ensure_hpc_settings=None,
    use_aiida: bool = False,
    generate_hpc: bool = False,
    simulation_root_name: Optional[str] = None,
) -> Tuple[List[Path], Path, List[Dict[str, Any]], Any]:
    """Run all simulations defined in a config file.

    Returns a tuple of (created_simulations, simulation_root, configs_to_run, settings).
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dict = parse_config(config_path)
    defaults = load_settings(settings_file=settings_file)

    if use_aiida:
        defaults.aiida.enabled = True
        defaults.aiida.create_provenance = True

    if model == 'sheetonsheet':
        if defaults.hpc.lammps_scripts == ['system.in', 'slide.in']:
            defaults.hpc.lammps_scripts = ['slide.in']
    elif model == 'afm':
        if defaults.hpc.lammps_scripts == ['slide.in']:
            defaults.hpc.lammps_scripts = ['system.in', 'slide.in']

    if ensure_hpc_settings is not None and generate_hpc:
        ensure_hpc_settings(defaults.hpc)

    configs_to_run = expand_config_sweeps(base_dict)

    for run_dict in configs_to_run:
        _validate_runtime_sweep_ordering(run_dict.get('general', {}))

    root_name = simulation_root_name or datetime.now().strftime("simulation_%Y%m%d_%H%M%S")
    base_root = Path(output_root) if output_root is not None else Path.cwd()
    simulation_root = base_root / root_name
    simulation_root.mkdir(parents=True, exist_ok=True)
    (simulation_root / 'logs').mkdir(parents=True, exist_ok=True)

    created_simulations: List[Path] = []

    for run_dict in configs_to_run:
        run_dict['settings'] = defaults.model_dump()

        mat = run_dict['2D'].get('mat', 'unknown')
        x = run_dict['2D'].get('x', 100)
        y = run_dict['2D'].get('y', 100)
        temp = run_dict['general'].get('temp', 300)
        size_token = f"{format_dimension_token(x)}x_{format_dimension_token(y)}y"

        if model == "afm":
            tip_mat = run_dict.get('tip', {}).get('mat', 'Si')
            tip_amorph = run_dict.get('tip', {}).get('amorph', 'c')
            tip_r = run_dict.get('tip', {}).get('r', 25)
            sub_mat = run_dict.get('sub', {}).get('mat', 'Si')
            sub_amorph = run_dict.get('sub', {}).get('amorph', 'a')

            sub_str = f"{sub_amorph}{sub_mat}" if sub_amorph == 'a' else sub_mat
            tip_str = f"{tip_amorph}{tip_mat}" if tip_amorph == 'a' else tip_mat

            output_dir = (
                simulation_root / "afm" / mat / size_token /
                f"sub_{sub_str}_tip_{tip_str}_r{int(tip_r)}" /
                f"K{int(temp)}"
            )
        else:
            output_dir = (
                simulation_root / "sheetonsheet" / mat / size_token /
                f"K{int(temp)}"
            )

        prov_dir = output_dir / 'provenance'
        prov_dir.mkdir(parents=True, exist_ok=True)

        try:
            if model == 'afm':
                config_obj = AFMSimulationConfig(**run_dict)
                config_json_path = prov_dir / 'config.json'
                config_json_path.write_text(config_obj.model_dump_json(indent=2), encoding='utf-8')
                builder = AFMSimulation(config_obj, output_dir, config_path=str(config_json_path))
            else:
                config_obj = SheetOnSheetSimulationConfig(**run_dict)
                config_json_path = prov_dir / 'config.json'
                config_json_path.write_text(config_obj.model_dump_json(indent=2), encoding='utf-8')
                builder = SheetOnSheetSimulation(config_obj, output_dir, config_path=str(config_json_path))

            builder.set_base_output_dir(simulation_root)
            builder.build()
            created_simulations.append(output_dir)

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Build failed for %s: %s", output_dir, exc)
            continue

    if generate_hpc and created_simulations:
        generate_hpc_scripts_for_root(simulation_root, defaults)

    return created_simulations, simulation_root, configs_to_run, defaults
