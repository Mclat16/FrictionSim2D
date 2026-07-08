"""Job manifest for offline HPC workflow tracking."""

import json
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


class JobStatus(Enum):
    """Simulation job lifecycle states."""
    PREPARED = "prepared"
    EXPORTED = "exported"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    IMPORTED = "imported"


@dataclass
class JobEntry:
    """Single simulation job with status tracking.
    
    Attributes:
        job_id: Unique identifier for the job.
        simulation_path: Relative path to simulation directory.
        lammps_script: Optional LAMMPS script name (e.g., 'slide_20ms.in').
        material: Material identifier.
        layers: Number of layers in the material.
        force: Applied force in nN.
        angle: Scan angle in degrees.
        speed: Sliding speed in m/s.
        status: Current job status.
        hpc_job_id: PBS/SLURM job ID when submitted.
        created_at: ISO timestamp of creation.
        submitted_at: ISO timestamp of submission.
        completed_at: ISO timestamp of completion.
        simulation_node_uuid: AiiDA simulation node UUID.
        results_node_uuid: AiiDA results node UUID.
        error_message: Error description if failed.
    """
    job_id: str
    simulation_path: str
    lammps_script: Optional[str] = None
    material: str = ""
    layers: int = 1
    force: float = 0.0
    angle: float = 0.0
    speed: float = 2.0
    status: str = JobStatus.PREPARED.value
    hpc_job_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    simulation_node_uuid: Optional[str] = None
    results_node_uuid: Optional[str] = None
    error_message: Optional[str] = None

    def update_status(self, new_status: JobStatus, **kwargs):
        """Update job status with automatic timestamp management.
        
        Args:
            new_status: New status to set.
            **kwargs: Additional fields to update (e.g., hpc_job_id, error_message).
        """
        self.status = new_status.value
        now = datetime.now().isoformat()
        if new_status == JobStatus.SUBMITTED:
            self.submitted_at = now
        elif new_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.IMPORTED):
            self.completed_at = now
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobEntry':
        """Create JobEntry from dictionary.
        
        Args:
            data: Dictionary with job entry fields.
            
        Returns:
            New JobEntry instance.
        """
        return cls(**data)


@dataclass
class JobManifest:
    """Batch job tracker for offline HPC workflows with AiiDA integration.
    
    Central tracking system for simulations in disconnected HPC workflows where
    AiiDA cannot directly monitor jobs. Stores job metadata, status, and enables
    manual HPC job management while maintaining provenance.
    
    Attributes:
        name: Manifest identifier.
        created_at: ISO timestamp of creation.
        last_updated: ISO timestamp of last modification.
        config_file: Path to source config file.
        source_directory: Path to simulation root directory.
        config_node_uuid: AiiDA config node UUID.
        provenance_node_uuid: AiiDA provenance node UUID.
        jobs: List of job entries.
        scheduler: HPC scheduler type ('pbs' or 'slurm').
        package_directory: Path to HPC package if exported.
    """
    name: str = "friction2d_manifest"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    config_file: Optional[str] = None
    source_directory: Optional[str] = None
    config_node_uuid: Optional[str] = None
    provenance_node_uuid: Optional[str] = None
    jobs: List[JobEntry] = field(default_factory=list)
    scheduler: str = "pbs"
    package_directory: Optional[str] = None

    def __post_init__(self):
        if self.jobs is None:
            self.jobs = []

    @property
    def n_jobs(self) -> int:
        """Total number of jobs in manifest."""
        return len(self.jobs)

    def _count_status(self, *statuses: JobStatus) -> int:
        status_values = {s.value for s in statuses}
        return sum(1 for j in self.jobs if j.status in status_values)

    @property
    def n_prepared(self) -> int:
        """Number of prepared jobs."""
        return self._count_status(JobStatus.PREPARED)

    @property
    def n_submitted(self) -> int:
        """Number of submitted or running jobs."""
        return self._count_status(JobStatus.SUBMITTED, JobStatus.RUNNING)

    @property
    def n_completed(self) -> int:
        """Number of completed jobs."""
        return self._count_status(JobStatus.COMPLETED)

    @property
    def n_failed(self) -> int:
        """Number of failed jobs."""
        return self._count_status(JobStatus.FAILED)

    @property
    def n_imported(self) -> int:
        """Number of jobs with imported results."""
        return self._count_status(JobStatus.IMPORTED)

    def add_job(self, entry: JobEntry) -> None:
        """Add job entry to manifest.
        
        Args:
            entry: JobEntry to add.
        """
        self.jobs.append(entry)
        self._update_timestamp()

    def get_job(self, job_id: str) -> Optional[JobEntry]:
        """Get job by ID.
        
        Args:
            job_id: Job identifier.
            
        Returns:
            JobEntry if found, None otherwise.
        """
        return next((j for j in self.jobs if j.job_id == job_id), None)

    def get_jobs_by_status(self, status: JobStatus) -> List[JobEntry]:
        """Get all jobs with specific status.
        
        Args:
            status: Status to filter by.
            
        Returns:
            List of matching JobEntry objects.
        """
        return [j for j in self.jobs if j.status == status.value]

    def get_jobs_by_material(self, material: str) -> List[JobEntry]:
        """Get all jobs for specific material.
        
        Args:
            material: Material name.
            
        Returns:
            List of matching JobEntry objects.
        """
        return [j for j in self.jobs if j.material == material]

    def get_system_jobs(self) -> List[JobEntry]:
        """Get all system initialization jobs.
        
        Returns:
            List of jobs with system*.in scripts.
        """
        return [j for j in self.jobs
                if j.lammps_script and j.lammps_script.startswith('system')]

    def get_slide_jobs(self) -> List[JobEntry]:
        """Get all slide simulation jobs.
        
        Returns:
            List of jobs with slide*.in scripts.
        """
        return [j for j in self.jobs
                if j.lammps_script and j.lammps_script.startswith('slide')]

    def has_system_jobs(self) -> bool:
        """Check if manifest contains system initialization jobs.
        
        Returns:
            True if system jobs exist.
        """
        return bool(self.get_system_jobs())

    def update_job_status(self, job_id: str, new_status: JobStatus,
                         **kwargs) -> bool:
        """Update status of specific job.
        
        Args:
            job_id: Job identifier.
            new_status: New status to set.
            **kwargs: Additional fields to update.
            
        Returns:
            True if job was found and updated, False otherwise.
        """
        job = self.get_job(job_id)
        if job:
            job.update_status(new_status, **kwargs)
            self._update_timestamp()
            return True
        return False

    def mark_all_submitted(self, hpc_job_prefix: Optional[str] = None) -> int:
        """Mark all prepared jobs as submitted.
        
        Args:
            hpc_job_prefix: Optional prefix for HPC job IDs.
            
        Returns:
            Number of jobs updated.
        """
        count = 0
        for job in self.jobs:
            if job.status == JobStatus.PREPARED.value:
                hpc_id = f"{hpc_job_prefix}_{count}" if hpc_job_prefix else None
                job.update_status(JobStatus.SUBMITTED, hpc_job_id=hpc_id)
                count += 1
        if count > 0:
            self._update_timestamp()
        return count

    def mark_completed_from_results(self, results_dir: Path) -> Dict[str, str]:
        """Mark jobs as completed based on presence of result files.
        
        Args:
            results_dir: Directory containing simulation results.
            
        Returns:
            Dictionary of job_id -> status for updated jobs.
        """
        results_dir = Path(results_dir)
        updated = {}
        for job in self.jobs:
            if job.status in (JobStatus.SUBMITTED.value, JobStatus.RUNNING.value):
                sim_path = results_dir / job.simulation_path / 'results'
                if sim_path.exists() and any(sim_path.iterdir()):
                    job.update_status(JobStatus.COMPLETED)
                    updated[job.job_id] = JobStatus.COMPLETED.value
        if updated:
            self._update_timestamp()
        return updated

    def get_summary(self) -> Dict[str, Any]:
        """Get manifest status summary.
        
        Returns:
            Dictionary with status counts and metadata.
        """
        return {
            'name': self.name,
            'total_jobs': self.n_jobs,
            'prepared': self.n_prepared,
            'submitted': self.n_submitted,
            'completed': self.n_completed,
            'failed': self.n_failed,
            'imported': self.n_imported,
            'scheduler': self.scheduler,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
        }

    def _update_timestamp(self) -> None:
        self.last_updated = datetime.now().isoformat()

    def save_script_list(self, filepath: Path,
                        jobs: Optional[List[JobEntry]] = None) -> int:
        """Write script paths to text file for HPC array jobs.
        
        Args:
            filepath: Path to save script list.
            jobs: Optional list of jobs to save. If None, uses all jobs.
            
        Returns:
            Number of scripts written.
        """
        jobs_to_save = jobs if jobs is not None else self.jobs
        script_paths = [
            f"{j.simulation_path}/lammps/{j.lammps_script or 'slide.in'}"
            for j in jobs_to_save
        ]
        Path(filepath).write_text('\n'.join(script_paths) + '\n', encoding='utf-8')
        return len(script_paths)

    def save(self, filepath: Path) -> None:
        """Save manifest to JSON file.
        
        Args:
            filepath: Path to save manifest.
        """
        data = {
            'name': self.name,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'config_file': self.config_file,
            'source_directory': self.source_directory,
            'config_node_uuid': self.config_node_uuid,
            'provenance_node_uuid': self.provenance_node_uuid,
            'scheduler': self.scheduler,
            'package_directory': self.package_directory,
            'jobs': [job.to_dict() for job in self.jobs],
        }
        Path(filepath).write_text(json.dumps(data, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, filepath: Path) -> 'JobManifest':
        """Load manifest from JSON file.
        
        Args:
            filepath: Path to manifest file.
            
        Returns:
            Loaded JobManifest instance.
        """
        data = json.loads(Path(filepath).read_text(encoding='utf-8'))
        jobs = [JobEntry.from_dict(j) for j in data.pop('jobs', [])]
        manifest = cls(**data)
        manifest.jobs = jobs
        return manifest

    @classmethod
    def from_simulation_directory(cls, sim_dir: Path,
                                    name: Optional[str] = None) -> 'JobManifest':
        """Create manifest by discovering jobs in simulation directory.
        
        Scans directory tree for lammps/ subdirectories and creates job entries
        for all discovered LAMMPS scripts. Processes system*.in scripts first,
        then slide*.in scripts.
        
        Args:
            sim_dir: Directory containing simulation outputs.
            name: Optional manifest name (defaults to directory name).
            
        Returns:
            New JobManifest with entries for all discovered simulations.
        """
        sim_dir = Path(sim_dir)
        manifest = cls(name=name or sim_dir.name, source_directory=str(sim_dir))
        layer_pattern = re.compile(r'l[_]?(\d+)', re.IGNORECASE)

        sim_directories = []
        for lammps_dir in sim_dir.rglob('lammps'):
            if not lammps_dir.is_dir():
                continue
            sim_path = lammps_dir.parent
            rel_path = sim_path.relative_to(sim_dir)
            path_str = rel_path.as_posix()
            parts = rel_path.parts
            material = parts[1] if len(parts) >= 2 and parts[0] in ('afm', 'sheetonsheet', 'pes_sheet', 'pes_tip') else (parts[0] if parts else "")
            layer_match = layer_pattern.search(path_str)
            layers = int(layer_match.group(1)) if layer_match else 1
            sim_directories.append({
                'lammps_dir': lammps_dir,
                'rel_path': rel_path,
                'path_str': path_str,
                'material': material,
                'layers': layers,
            })

        for sim_info in sim_directories:
            lammps_dir = sim_info['lammps_dir']
            for script_name in sorted(p.name for p in lammps_dir.glob('system*.in')):
                script_tag = Path(script_name).stem.replace('-', '_')
                job_id = f"{sim_info['path_str']}_{script_tag}".replace('/', '_').replace('-', '_')
                manifest.add_job(JobEntry(
                    job_id=job_id,
                    simulation_path=str(sim_info['rel_path']),
                    lammps_script=script_name,
                    material=sim_info['material'],
                    layers=sim_info['layers'],
                ))

        for sim_info in sim_directories:
            lammps_dir = sim_info['lammps_dir']
            slide_scripts = sorted(p.name for p in lammps_dir.glob('slide*.in'))
            if not slide_scripts:
                if not list(lammps_dir.glob('system*.in')):
                    job_id = sim_info['path_str'].replace('/', '_').replace('-', '_')
                    manifest.add_job(JobEntry(
                        job_id=job_id,
                        simulation_path=str(sim_info['rel_path']),
                        material=sim_info['material'],
                        layers=sim_info['layers'],
                    ))
                continue

            for script_name in slide_scripts:
                stem = Path(script_name).stem
                script_tag = stem.replace('-', '_')
                job_id = f"{sim_info['path_str']}_{script_tag}".replace('/', '_').replace('-', '_')
                speed = 2.0
                speed_match = re.match(r"slide_([0-9]+(?:p[0-9]+)?)ms", stem)
                if speed_match:
                    speed = float(speed_match.group(1).replace('p', '.'))
                # Per-load slide scripts (AFM outer_loop=force): slide_f<load>N.in
                force = 0.0
                force_match = re.match(r"slide_f([0-9]+(?:p[0-9]+)?)N$", stem)
                if force_match:
                    force = float(force_match.group(1).replace('p', '.'))
                manifest.add_job(JobEntry(
                    job_id=job_id,
                    simulation_path=str(sim_info['rel_path']),
                    lammps_script=script_name,
                    material=sim_info['material'],
                    layers=sim_info['layers'],
                    force=force,
                    speed=speed,
                ))
        return manifest

    def __repr__(self) -> str:
        return (f"<JobManifest: {self.name} "
                f"({self.n_jobs} jobs, {self.n_completed} completed, "
                f"{self.n_failed} failed)>")


def create_manifest_from_package(package_dir: Path) -> JobManifest:
    """Create manifest from exported HPC package directory.
    
    Args:
        package_dir: Path to HPC package directory.
        
    Returns:
        JobManifest instance with all jobs marked as exported.
    """
    package_dir = Path(package_dir)
    info_file = package_dir / 'package_info.json'
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    sim_dir = package_dir / 'simulations'
    manifest = (JobManifest.from_simulation_directory(sim_dir, name=package_dir.name)
                if sim_dir.exists() else JobManifest(name=package_dir.name))
    manifest.scheduler = info.get('scheduler', 'pbs')
    manifest.package_directory = str(package_dir)
    for job in manifest.jobs:
        job.update_status(JobStatus.EXPORTED)
    return manifest
