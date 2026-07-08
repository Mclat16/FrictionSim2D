"""PES-scan descriptor extraction.

Reduces the raw ``results/pes_scan.csv`` energy/force grids produced by the
static PES scans (:mod:`src.builders.pes_scan`) to a handful of per-material
scalar descriptors — the corrugation-barrier quantities the friction literature
relies on (Prandtl–Tomlinson) — and writes:

    - ``<data_dir>/pes_scan_sheet.csv`` / ``pes_scan_tip.csv`` — one row per
      material, merge key ``material`` (underscored id, e.g. ``h_MoS2``), ready
      to plug into the ML feature pipeline exactly like ``sw_features.csv``.
    - ``<grid_dir>/{sheet,tip}/<material>.csv`` — the tidied per-material grids.

Descriptors (per surface unit cell for the sheet scan; per fixed tip for the
tip scan, where absolute energies share a common offset that cancels in
max−min):

    ============= =========================================================
    pes_barrier_U  max(E) − min(E) — the Prandtl–Tomlinson corrugation barrier
    pes_barrier_mep barrier along the easiest straight sliding channel (min over
                    transverse offset of the peak) — the easy-sliding barrier
    pes_corr_rms   RMS of E − mean(E) — overall corrugation strength
    pes_barrier_aniso barrier along a vs b (directional / friction-anisotropy)
    pes_curv_min   mean axial curvature of E at the minimum — lateral stiffness K
    pes_eta        2π²·U / (K·a²) — dimensionless PT friction parameter
    pes_fmax       (tip only) max |lateral force| — peak static-friction force
    ============= =========================================================
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-12

#: Top-level run directory name -> scan kind.
KIND_BY_TOPDIR = {"pes_sheet": "sheet", "pes_tip": "tip"}


def material_id(name: str) -> str:
    """Map a sheet file-stem/dir name to the ML merge id (``h-MoS2`` -> ``h_MoS2``)."""
    return name.replace("-", "_")


def _grid_from_frame(df: pd.DataFrame) -> Tuple[np.ndarray, float, float]:
    """Pivot a (dx|x, dy|y, energy) frame into an (Ny, Nx) grid + axis steps.

    Missing grid points are filled with the grid mean so barrier/curvature stay
    finite; the count of usable points is validated by the caller.
    """
    xcol = "dx" if "dx" in df.columns else "x"
    ycol = "dy" if "dy" in df.columns else "y"
    piv = df.pivot_table(index=ycol, columns=xcol, values="energy_eV", aggfunc="mean")
    grid = piv.to_numpy(dtype=float)
    if np.isnan(grid).any():
        grid = np.where(np.isnan(grid), np.nanmean(grid), grid)
    xs = np.asarray(piv.columns, dtype=float)
    ys = np.asarray(piv.index, dtype=float)
    dx = float(np.median(np.diff(xs))) if xs.size > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if ys.size > 1 else 1.0
    return grid, dx, dy


def _channel_barrier(grid: np.ndarray, axis: int) -> float:
    """Easiest straight-line sliding barrier along ``axis`` (peak above the min).

    Sliding straight along an axis at fixed transverse offset traverses a 1D
    profile whose peak is the barrier for that channel; the easy-sliding barrier
    is the lowest such peak over all transverse offsets. ``axis=1`` slides along
    x (a), ``axis=0`` along y (b).
    """
    peak_per_line = grid.max(axis=axis)          # peak along each straight line
    easiest_peak = float(peak_per_line.min())    # the lowest-ridge channel
    return easiest_peak - float(grid.min())


def _curvature_at_min(grid: np.ndarray, dx: float, dy: float) -> float:
    """Mean of the two axial second derivatives of E at the global minimum.

    Uses periodic central differences at the minimum cell → effective lateral
    stiffness K (eV/Å²).
    """
    ny, nx = grid.shape
    iy, ix = np.unravel_index(int(np.argmin(grid)), grid.shape)
    e0 = grid[iy, ix]
    cxx = (grid[iy, (ix + 1) % nx] - 2 * e0 + grid[iy, (ix - 1) % nx]) / (dx * dx + EPS)
    cyy = (grid[(iy + 1) % ny, ix] - 2 * e0 + grid[(iy - 1) % ny, ix]) / (dy * dy + EPS)
    return 0.5 * (cxx + cyy)


def grid_descriptors(df: pd.DataFrame, kind: str) -> Dict[str, float]:
    """Compute the scalar PES descriptors for one material's grid frame."""
    grid, dx, dy = _grid_from_frame(df)
    e = grid.ravel()
    barrier_u = float(e.max() - e.min())
    barrier_a = _channel_barrier(grid, axis=1)   # slide along x (a)
    barrier_b = _channel_barrier(grid, axis=0)   # slide along y (b)
    curv = _curvature_at_min(grid, dx, dy)
    # PT friction parameter eta = 2 pi^2 U / (K a^2); a = shorter surface period.
    a = min(dx * grid.shape[1], dy * grid.shape[0])
    eta = 2.0 * np.pi ** 2 * barrier_u / (abs(curv) * a * a + EPS)

    out: Dict[str, float] = {
        "pes_barrier_U": barrier_u,
        "pes_barrier_mep": float(min(barrier_a, barrier_b)),
        "pes_corr_rms": float(np.sqrt(np.mean((e - e.mean()) ** 2))),
        "pes_barrier_aniso": float(barrier_a / (barrier_b + EPS)),
        "pes_curv_min": float(curv),
        "pes_eta": float(eta),
        "pes_grid_n": int(min(grid.shape)),
    }
    if kind == "tip" and {"fx", "fy"} <= set(df.columns):
        flat = np.hypot(df["fx"].to_numpy(float), df["fy"].to_numpy(float))
        out["pes_fmax"] = float(np.nanmax(flat))
        out["pes_flat_rms"] = float(np.sqrt(np.nanmean(flat ** 2)))
    return out


def _material_from_run(csv_path: Path, run_root: Path, kind_topdir: str) -> Optional[str]:
    """Extract the material id from a ``.../<kind>/<material>/.../results/pes_scan.csv``."""
    try:
        parts = csv_path.relative_to(run_root).parts
    except ValueError:
        parts = csv_path.parts
    if kind_topdir in parts:
        i = parts.index(kind_topdir)
        if i + 1 < len(parts):
            return material_id(parts[i + 1])
    return None


def discover_grids(run_root: Path, kind: str) -> List[Tuple[str, Path]]:
    """Find (material, csv_path) for every PES ``pes_scan.csv`` of the given kind.

    ``pes_scan_refine.csv`` grids are ignored here (they are a convergence check;
    the base grid is the descriptor source).
    """
    topdir = next(k for k, v in KIND_BY_TOPDIR.items() if v == kind)
    found: List[Tuple[str, Path]] = []
    for csv_path in sorted(run_root.rglob("results/pes_scan.csv")):
        if topdir not in csv_path.parts:
            continue
        mat = _material_from_run(csv_path, run_root, topdir)
        if mat:
            found.append((mat, csv_path))
    return found


def extract_all(run_root: Path, kind: str,
                grid_dir: Optional[Path] = None) -> pd.DataFrame:
    """Reduce every ``kind`` PES grid under ``run_root`` to one row per material.

    When ``grid_dir`` is given, each tidied grid is also copied to
    ``grid_dir/<kind>/<material>.csv`` (the recipe hand-back layout).
    """
    rows: List[Dict[str, float]] = []
    for mat, csv_path in discover_grids(run_root, kind):
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError) as exc:
            logger.warning("Skipping unreadable PES grid %s: %s", csv_path, exc)
            continue
        if len(df) < 4 or "energy_eV" not in df.columns:
            logger.warning("Skipping degenerate PES grid %s (%d rows)", csv_path, len(df))
            continue
        feats = grid_descriptors(df, kind)
        rows.append({"material": mat, **feats})
        if grid_dir is not None:
            dest = Path(grid_dir) / kind / f"{mat}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest, index=False)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("material").reset_index(drop=True)
    return out


def run(run_root: Path, data_dir: Path,
        grid_dir: Optional[Path] = None,
        kinds: Optional[List[str]] = None) -> Dict[str, Path]:
    """Extract PES-scan descriptors and write per-material CSVs.

    Args:
        run_root: Simulation root containing ``pes_sheet/`` and/or ``pes_tip/``.
        data_dir: Directory for the ``pes_scan_<kind>.csv`` descriptor tables.
        grid_dir: Optional directory for the tidied per-material grids
            (``<grid_dir>/<kind>/<material>.csv``).
        kinds: Which scans to process (default: both 'sheet' and 'tip').

    Returns:
        Mapping of kind -> written descriptor CSV path (only kinds with data).
    """
    run_root = Path(run_root)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for kind in (kinds or ["sheet", "tip"]):
        df = extract_all(run_root, kind, grid_dir=grid_dir)
        if df.empty:
            logger.info("No %s PES grids found under %s", kind, run_root)
            continue
        out_csv = data_dir / f"pes_scan_{kind}.csv"
        df.to_csv(out_csv, index=False)
        logger.info("Wrote %d %s PES descriptors -> %s", len(df), kind, out_csv)
        written[kind] = out_csv
    return written
