"""HPC script generation for PBS and SLURM schedulers."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import shutil
import json

from jinja2 import Environment, FileSystemLoader

from .manifest import JobManifest
from ..core.config import GlobalSettings

TEMPLATES_DIR = Path(__file__).parent.parent / 'templates' / 'hpc'


def _default_hpc_settings():
    """Return default validated HPC settings from GlobalSettings."""
    return GlobalSettings().hpc


@dataclass
class HPCConfig:
    """HPC job submission configuration.
    
    Can be created from GlobalSettings.hpc or instantiated directly.
    
    Attributes:
        scheduler_type: 'pbs' or 'slurm'.
        nodes: Number of compute nodes.
        cpus_per_node: CPUs per node.
        memory_gb: Memory per node in GB.
        walltime_hours: Maximum runtime in hours.
        job_name: Job name prefix.
        queue: PBS queue name.
        partition: SLURM partition name.
        account: Account/project code.
        hpc_home: HPC home directory path.
        log_dir: Directory for job logs.
        scratch_dir: Scratch directory path.
        modules: Environment modules to load.
        mpi_command: MPI launcher command.
        lmp_flags: Additional LAMMPS flags.
        use_tmpdir: Whether to use scratch directory.
        lammps_scripts: Script names to execute.
        max_array_size: Maximum array job size.
    """
    hpc_settings: Any = field(default_factory=_default_hpc_settings)
    job_name: str = "friction2d"
    lmp_flags: str = "-l none"

    @classmethod
    def from_settings(cls, hpc_settings, job_name: str = "friction2d") -> 'HPCConfig':
        """Create HPCConfig from GlobalSettings.hpc.
        
        Args:
            hpc_settings: HPCSettings instance from GlobalSettings.
            job_name: Job name prefix.
            
        Returns:
            New HPCConfig instance.
        """
        return cls(hpc_settings=hpc_settings, job_name=job_name)

    @property
    def scheduler_type(self) -> Literal['pbs', 'slurm']:
        return self.hpc_settings.scheduler_type

    @property
    def nodes(self) -> int:
        return self.hpc_settings.num_nodes

    @property
    def cpus_per_node(self) -> int:
        return self.hpc_settings.num_cpus

    @property
    def memory_gb(self) -> int:
        return self.hpc_settings.memory_gb

    @property
    def walltime_hours(self) -> int:
        return self.hpc_settings.walltime_hours

    @property
    def queue(self) -> Optional[str]:
        return self.hpc_settings.queue or None

    @property
    def partition(self) -> Optional[str]:
        return self.hpc_settings.partition

    @property
    def account(self) -> Optional[str]:
        return self.hpc_settings.account or None

    @property
    def hpc_host(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'hpc_host', None)

    @property
    def hpc_home(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'hpc_home', None)

    @property
    def log_dir(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'log_dir', None)

    @property
    def scratch_dir(self) -> Optional[str]:
        return getattr(self.hpc_settings, 'scratch_dir', None)

    @property
    def modules(self) -> List[str]:
        return list(self.hpc_settings.modules or [])

    @property
    def mpi_command(self) -> str:
        return self.hpc_settings.mpi_command

    @property
    def use_tmpdir(self) -> bool:
        return self.hpc_settings.use_tmpdir

    @property
    def lammps_scripts(self) -> List[str]:
        return list(getattr(self.hpc_settings, 'lammps_scripts', None) or ['system.in', 'slide.in'])

    @lammps_scripts.setter
    def lammps_scripts(self, scripts: List[str]) -> None:
        self.hpc_settings.lammps_scripts = list(scripts)

    @property
    def max_array_size(self) -> int:
        return self.hpc_settings.max_array_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering.
        
        Returns:
            Dictionary with all config values and computed fields.
        """
        return {
            'nodes': self.nodes,
            'cpus_per_node': self.cpus_per_node,
            'memory_gb': self.memory_gb,
            'walltime': f"{self.walltime_hours}:00:00",
            'job_name': self.job_name,
            'queue': self.queue,
            'partition': self.partition,
            'account': self.account,
            'hpc_host': self.hpc_host,
            'hpc_home': self.hpc_home,
            'log_dir': self.log_dir,
            'scratch_dir': self.scratch_dir,
            'modules': self.modules,
            'mpi_command': self.mpi_command,
            'lmp_flags': self.lmp_flags,
            'use_tmpdir': self.use_tmpdir,
            'lammps_scripts': self.lammps_scripts,
            'select_multi': f"1:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb:mpiprocs={self.cpus_per_node}",
            'select_single': f"1:ncpus={self.cpus_per_node}:mem={self.memory_gb}gb:mpiprocs={self.cpus_per_node}",
            'ntasks_per_node': self.cpus_per_node,
            'cpus_per_task': 1,
            'mem': f"{self.memory_gb}G",
        }


class HPCScriptGenerator:
    """HPC batch script generator for LAMMPS simulations.
    
    Generates PBS and SLURM array job scripts with automatic chunking
    for large simulation sets.
    """

    def __init__(self, config: Optional[HPCConfig] = None):
        """Initialize generator.
        
        Args:
            config: HPC configuration (uses defaults if None).
        """
        self.config = config or HPCConfig.from_settings(_default_hpc_settings())
        self.jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATES_DIR),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _validate_config(self):
        """Validate required config fields.
        
        Raises:
            ValueError: If required fields are missing.
        """
        if not self.config.modules:
            raise ValueError("HPC modules list is empty.")
        if self.config.use_tmpdir and not self.config.scratch_dir:
            raise ValueError("HPC scratch_dir required when use_tmpdir is true.")

    def _resolve_log_dir(self, base_dir: str) -> str:
        """Resolve log directory for job scripts.

        Priority:
        1. Explicit log_dir in config.
        2. hpc_home/logs — a single shared directory for all jobs when hpc_home
           is set (avoids per-sim-root log subdirectories).
        3. base_dir/logs — fallback when no hpc_home (e.g. $PBS_O_WORKDIR/logs).
        """
        if self.config.log_dir:
            return self.config.log_dir
        if self.config.hpc_home:
            return self.config.hpc_home.rstrip('/') + '/logs'
        return f"{base_dir}/logs"

    def _ensure_local_log_dir(self, output_dir: Path) -> None:
        """Create the job log directory locally so it exists after transfer.

        Schedulers open the ``-o``/``-e`` log files *before* the job script
        runs, so an in-script ``mkdir`` is too late. The default log path is
        ``<sim_root>/logs`` (relative to the submit dir), and the submit dir is
        the parent of the ``hpc/`` output directory. Creating it here means the
        empty ``logs/`` folder ships with the package via rsync and exists when
        the job starts.
        """
        sim_root = Path(output_dir).parent
        (sim_root / 'logs').mkdir(parents=True, exist_ok=True)
        # If an explicit, submit-dir-relative log_dir was configured, create it
        # too (skip absolute paths and shell-variable paths we cannot resolve).
        log_dir = self.config.log_dir
        if log_dir and not log_dir.startswith(('$', '/')):
            (sim_root / log_dir).mkdir(parents=True, exist_ok=True)

    def _generate_array_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str, scheduler: Literal['pbs', 'slurm']) -> List[Path]:
        """Generate array job scripts with automatic chunking.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable for HPC.
            scheduler: Scheduler type.
            
        Returns:
            List of generated script paths.
        """
        self._validate_config()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        n_sims = len(simulation_paths)
        n_scripts = math.ceil(n_sims / self.config.max_array_size)
        template_name = f"{scheduler}_array.j2"
        script_ext = ".pbs" if scheduler == 'pbs' else ".sh"
        scripts = []
        template = self.jinja_env.get_template(template_name)

        for i in range(n_scripts):
            start_idx = i * self.config.max_array_size
            end_idx = min((i + 1) * self.config.max_array_size, n_sims)
            chunk = simulation_paths[start_idx:end_idx]

            manifest_name = f"manifest_{i+1}.txt"
            (output_dir / manifest_name).write_text('\n'.join(chunk))

            context = self.config.to_dict()
            context.update({
                'array_size': len(chunk),
                'manifest_file': f"{base_dir}/{output_dir.name}/{manifest_name}",
                'manifest_filename': manifest_name,
                'base_dir': base_dir,
                'log_dir': self._resolve_log_dir(base_dir),
            })
            if n_scripts > 1:
                context['job_name'] = f"{self.config.job_name}_{i+1}"

            script_name = f"run_{i+1}{script_ext}" if n_scripts > 1 else f"run{script_ext}"
            script_path = output_dir / script_name
            script_path.write_text(template.render(context))
            scripts.append(script_path)

        self._write_master_script(output_dir, scripts, scheduler)
        return scripts

    def generate_pbs_scripts(self, simulation_paths: List[str], output_dir: Path,
                            base_dir: str = "$PBS_O_WORKDIR") -> List[Path]:
        """Generate PBS array job scripts.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            
        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'pbs'
        )

    def generate_slurm_scripts(self, simulation_paths: List[str], output_dir: Path,
                                base_dir: str = "$SLURM_SUBMIT_DIR") -> List[Path]:
        """Generate SLURM array job scripts.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            base_dir: Base directory variable.
            
        Returns:
            List of generated script paths.
        """
        return self._generate_array_scripts(
            simulation_paths, output_dir, base_dir, 'slurm'
        )

    def _write_master_script(self, output_dir: Path, scripts: List[Path],
                            scheduler: Literal['pbs', 'slurm']) -> Path:
        """Write master submission instructions.
        
        Args:
            output_dir: Directory to write script.
            scripts: List of generated script paths.
            scheduler: Scheduler type.
            
        Returns:
            Path to master script.
        """
        submit_cmd = 'qsub' if scheduler == 'pbs' else 'sbatch'
        sim_root = output_dir.parent
        hpc_host = self.config.hpc_host or "<HPC_HOST>"
        hpc_home = self.config.hpc_home or "<HPC_HOME>/"
        rsync_target = f"{hpc_host}:{hpc_home}"
        lines = [
            '# Submit all job arrays',
            '',
            '# Optional: upload simulations to HPC',
            f'# rsync -rvltoD {sim_root.name} {rsync_target}',
            '',
            '# Optional: download results from HPC',
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --exclude="*" {rsync_target}{sim_root.name} ./',
            '',
            f'cd {sim_root}',
            '',
        ]
        lines.extend(f'{submit_cmd} {output_dir.name}/{s.name}' for s in scripts)
        master_path = output_dir / 'submit_all.txt'
        master_path.write_text('\n'.join(lines))
        return master_path

    def generate_combined_scripts(self, manifest: 'JobManifest', output_dir: Path,
                                   scheduler: Literal['pbs', 'slurm'] = 'pbs',
                                   base_dir: Optional[str] = None) -> List[Path]:
        """Generate a single combined system+slide script for AFM simulations.

        Each array task runs system.in then slide.in sequentially within the
        same job, eliminating the need for PBS job dependencies. Use when each
        simulation directory has exactly one system script and one slide script.

        Args:
            manifest: JobManifest containing all jobs.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type.
            base_dir: Base directory variable (auto-detected if None).

        Returns:
            List containing the single generated script path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        base_dir = base_dir or ('$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR')

        system_jobs = manifest.get_system_jobs()
        # One entry per unique simulation directory, preserving order.
        seen: dict = {}
        for j in system_jobs:
            seen.setdefault(j.simulation_path, None)
        sim_dirs = list(seen.keys())

        manifest_name = 'manifest_combined.txt'
        (output_dir / manifest_name).write_text('\n'.join(sim_dirs) + '\n', encoding='utf-8')

        template_name = f"{scheduler}_combined.j2"
        template = self.jinja_env.get_template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'

        context = self.config.to_dict()
        context.update({
            'array_size': len(sim_dirs),
            'manifest_file': f"{base_dir}/hpc/{manifest_name}",
            'manifest_filename': manifest_name,
            'base_dir': base_dir,
            'log_dir': self._resolve_log_dir(base_dir),
        })

        script_path = output_dir / f"run{script_ext}"
        script_path.write_text(template.render(context))

        self._write_combined_submission(output_dir, script_path, scheduler)
        return [script_path]

    def _write_combined_submission(self, output_dir: Path, script: Path,
                                    scheduler: Literal['pbs', 'slurm']) -> Path:
        """Write submission helper for a combined system+slide script.

        Args:
            output_dir: Directory to write script.
            script: The combined script path.
            scheduler: Scheduler type.

        Returns:
            Path to submission script.
        """
        submit_cmd = 'qsub' if scheduler == 'pbs' else 'sbatch'
        sim_root = output_dir.parent
        hpc_host = self.config.hpc_host or "<HPC_HOST>"
        hpc_home = self.config.hpc_home or "<HPC_HOME>/"
        rsync_target = f"{hpc_host}:{hpc_home}"
        lines = [
            '#!/bin/bash',
            '# Combined system+slide submission (single job, no dependency needed)',
            '',
            '# Optional: upload to HPC',
            f'# rsync -rvltoD {sim_root.name} {rsync_target}',
            '',
            f'cd {sim_root}',
            '',
            f'{submit_cmd} hpc/{script.name}',
            '',
            f'# Monitor with: {"qstat" if scheduler == "pbs" else "squeue"} -u $USER',
            '',
            '# Optional: download results after completion',
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --exclude="*" {rsync_target}{sim_root.name} ./',
        ]
        submit_path = output_dir / 'submit_jobs.sh'
        submit_path.write_text('\n'.join(lines))
        submit_path.chmod(0o755)
        return submit_path

    def generate_scripts(self, simulation_paths: List[str], output_dir: Path,
                        scheduler: Literal['pbs', 'slurm'] = 'pbs',
                        **kwargs) -> List[Path]:
        """Generate HPC scripts for specified scheduler.
        
        Args:
            simulation_paths: List of relative simulation paths.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type ('pbs' or 'slurm').
            **kwargs: Additional arguments for specific generator.
            
        Returns:
            List of generated script paths.
            
        Raises:
            ValueError: If scheduler is unknown.
        """
        if scheduler == 'pbs':
            return self.generate_pbs_scripts(simulation_paths, output_dir, **kwargs)
        if scheduler == 'slurm':
            return self.generate_slurm_scripts(simulation_paths, output_dir, **kwargs)
        raise ValueError(f"Unknown scheduler: {scheduler}")

    def generate_two_phase_scripts(self, manifest: 'JobManifest', output_dir: Path,
                                    scheduler: Literal['pbs', 'slurm'] = 'pbs',
                                    base_dir: Optional[str] = None) -> Dict[str, List[Path]]:
        """Generate separate system and slide job scripts with dependencies.
        
        Creates two array jobs: system initialization and slide simulations.
        Slide jobs depend on system jobs completing first (for AFM simulations).
        
        Args:
            manifest: JobManifest containing all jobs.
            output_dir: Directory to write scripts.
            scheduler: Scheduler type.
            base_dir: Base directory variable (auto-detected if None).
            
        Returns:
            Dictionary with 'system' and 'slide' script lists.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_local_log_dir(output_dir)

        base_dir = base_dir or ('$PBS_O_WORKDIR' if scheduler == 'pbs' else '$SLURM_SUBMIT_DIR')

        system_jobs = manifest.get_system_jobs()
        slide_jobs = manifest.get_slide_jobs()
        result = {'system': [], 'slide': []}
        template_name = f"{scheduler}_array.j2"
        template = self.jinja_env.get_template(template_name)
        script_ext = '.pbs' if scheduler == 'pbs' else '.sh'

        if system_jobs:
            manifest.save_script_list(output_dir / 'manifest_system.txt', system_jobs)
            context = self.config.to_dict()
            context.update({
                'array_size': len(system_jobs),
                'manifest_file': f"{base_dir}/hpc/manifest_system.txt",
                'manifest_filename': 'manifest_system.txt',
                'base_dir': base_dir,
                'log_dir': self._resolve_log_dir(base_dir),
                'job_name': f"{self.config.job_name}_system",
            })
            script_path = output_dir / f"run_system{script_ext}"
            script_path.write_text(template.render(context))
            result['system'].append(script_path)

        if slide_jobs:
            manifest.save_script_list(output_dir / 'manifest_slide.txt', slide_jobs)
            context = self.config.to_dict()
            context.update({
                'array_size': len(slide_jobs),
                'manifest_file': f"{base_dir}/hpc/manifest_slide.txt",
                'manifest_filename': 'manifest_slide.txt',
                'base_dir': base_dir,
                'log_dir': self._resolve_log_dir(base_dir),
                'job_name': f"{self.config.job_name}_slide",
            })
            script_path = output_dir / f"run_slide{script_ext}"
            script_path.write_text(template.render(context))
            result['slide'].append(script_path)

        self._write_two_phase_submission(output_dir, result, scheduler,
                                            has_system=len(system_jobs) > 0)
        return result

    def _write_two_phase_submission(self, output_dir: Path,
                                    scripts: Dict[str, List[Path]],
                                    scheduler: Literal['pbs', 'slurm'],
                                    has_system: bool) -> Path:
        """Write two-phase submission script with job dependencies.
        
        Args:
            output_dir: Directory to write script.
            scripts: Dictionary with 'system' and 'slide' script lists.
            scheduler: Scheduler type.
            has_system: Whether system jobs exist.
            
        Returns:
            Path to submission script.
        """
        submit_cmd = 'qsub' if scheduler == 'pbs' else 'sbatch'
        sim_root = output_dir.parent
        hpc_host = self.config.hpc_host or "<HPC_HOST>"
        hpc_home = self.config.hpc_home or "<HPC_HOME>/"
        rsync_target = f"{hpc_host}:{hpc_home}"
        lines = [
            '#!/bin/bash',
            '# Two-phase submission: system jobs → slide jobs',
            '',
            '# Optional: upload to HPC',
            f'# rsync -rvltoD {sim_root.name} {rsync_target}',
            '',
        ]

        if has_system:
            if scheduler == 'slurm':
                lines.extend([
                    'echo "Submitting system initialization jobs..."',
                    f'SYSTEM_JOB_ID=$(sbatch --parsable hpc/{scripts["system"][0].name})',
                    'echo "System job ID: $SYSTEM_JOB_ID"',
                    '',
                    'echo "Submitting slide jobs (waiting for system jobs)..."',
                    f'SLIDE_JOB_ID=$(sbatch --parsable --dependency=afterok:$SYSTEM_JOB_ID hpc/{scripts["slide"][0].name})',
                    'echo "Slide job ID: $SLIDE_JOB_ID"',
                    '',
                    'echo "Monitor with: squeue -u $USER"',
                ])
            else:
                lines.extend([
                    'echo "Submitting system initialization jobs..."',
                    f'SYSTEM_JOB_ID=$({submit_cmd} hpc/{scripts["system"][0].name})',
                    'echo "System job ID: $SYSTEM_JOB_ID"',
                    '',
                    'echo "Submitting slide jobs (waiting for system jobs)..."',
                    f'SLIDE_JOB_ID=$({submit_cmd} -W depend=afterok:$SYSTEM_JOB_ID hpc/{scripts["slide"][0].name})',
                    'echo "Slide job ID: $SLIDE_JOB_ID"',
                    '',
                    'echo "Monitor with: qstat -u $USER"',
                ])
        else:
            monitor_cmd = 'squeue' if scheduler == 'slurm' else 'qstat'
            lines.extend([
                'echo "Submitting slide jobs..."',
                f'{submit_cmd} hpc/{scripts["slide"][0].name}',
                '',
                f'echo "Monitor with: {monitor_cmd} -u $USER"',
            ])

        lines.extend([
            '',
            '# Optional: download results after completion',
            f'# rsync -avzhP --include="*/" --include="results/***"'
            f' --include="visuals/***" --exclude="*" {rsync_target}{sim_root.name} ./',
        ])

        submit_path = output_dir / 'submit_jobs.sh'
        submit_path.write_text('\n'.join(lines))
        submit_path.chmod(0o755)
        return submit_path


def create_hpc_package(simulation_dir: Path, output_dir: Path,
                        scheduler: Literal['pbs', 'slurm'] = 'pbs',
                        config: Optional[HPCConfig] = None) -> Path:
    """Package simulations for HPC transfer.
    
    Creates self-contained directory with simulations, scripts, and metadata
    ready for transfer to HPC cluster.
    
    Args:
        simulation_dir: Directory containing FrictionSim2D outputs.
        output_dir: Directory to create package in.
        scheduler: HPC scheduler type.
        config: HPC configuration (uses defaults if None).
        
    Returns:
        Path to created package directory.
        
    Raises:
        ValueError: If no simulations found.
    """
    simulation_dir = Path(simulation_dir)
    output_dir = Path(output_dir)
    package_dir = output_dir / "friction2d_package"
    package_dir.mkdir(parents=True, exist_ok=True)

    simulation_paths = [
        str(d.parent.relative_to(simulation_dir))
        for d in simulation_dir.rglob('lammps') if d.is_dir()
    ]
    if not simulation_paths:
        raise ValueError(f"No simulations found in {simulation_dir}")

    shutil.copytree(simulation_dir, package_dir / 'simulations',
                    ignore=shutil.ignore_patterns('*.lammpstrj'))

    # base_dir '../simulations' means logs resolve under the simulations tree;
    # create it so the scheduler can open its log files at job start.
    (package_dir / 'simulations' / 'logs').mkdir(parents=True, exist_ok=True)

    generator = HPCScriptGenerator(config)
    generator.generate_scripts(simulation_paths, package_dir / 'hpc',
                                scheduler=scheduler, base_dir='../simulations')

    info = {'n_simulations': len(simulation_paths), 'scheduler': scheduler,
            'created_from': str(simulation_dir)}
    (package_dir / 'package_info.json').write_text(json.dumps(info, indent=2))

    monitor = 'qstat' if scheduler == 'pbs' else 'squeue'
    readme = f"""# FrictionSim2D HPC Package

{len(simulation_paths)} simulations for {scheduler.upper()}

## Usage
1. Transfer to HPC: `rsync -avz friction2d_package/ <hpc>:~/`
2. Submit jobs: `cd hpc && bash submit_all.txt`
3. Monitor: `{monitor} -u $USER`
4. Download results: `rsync -avz <hpc>:~/friction2d_package/simulations/*/results ./`
"""
    (package_dir / 'README.md').write_text(readme)
    return package_dir
