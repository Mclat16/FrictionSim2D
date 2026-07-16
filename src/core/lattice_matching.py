"""Zur-McGill coincidence lattice matching and Hod Registry Index for heterostructures.

This module provides two complementary tools for 2D material heterostructure setup:

1. **Zur-McGill lattice matching** — finds the smallest commensurate supercell for two
   2D lattices by enumerating integer transformation matrices and minimising strain.
   Answers: *"What supercell box should I build?"*

   Reference:
       Zur & McGill, J. Appl. Phys. 55, 378 (1984).

2. **Hod Registry Index (RI)** — scans the geometric overlap of electron-density disks
   between two layers as a function of lateral shift.  RI = 1 at AA stacking (full
   registry, highest friction); RI = 0 at the optimal AB anti-registry site
   (superlubricity regime).

   Reference:
       Hod, O. ChemPhysChem 14, 2376 (2013).
       "The Registry Index: A Quantitative Measure of Materials' Interfacial
        Commensurability"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

import numpy as np


@dataclass
class MatchResult:
    """Coincidence-lattice match between two 2D material cells.

    All matrices are 2×2 integer transformation matrices M such that
    ``supercell = M @ unit_cell`` (row-vector convention).

    Attributes:
        matrix_a: Integer supercell transformation matrix for material A.
        matrix_b: Integer supercell transformation matrix for material B.
        supercell_a: 2×2 supercell lattice vectors for material A (Å).
        supercell_b: 2×2 supercell lattice vectors for material B (Å).
        strain_a: Max absolute principal strain applied to A to reach the common cell.
        strain_b: Max absolute principal strain applied to B to reach the common cell.
        area: Supercell area (Å²).
        mean_strain: Average of ``strain_a`` and ``strain_b``.
    """
    matrix_a: np.ndarray
    matrix_b: np.ndarray
    supercell_a: np.ndarray
    supercell_b: np.ndarray
    strain_a: float
    strain_b: float
    area: float
    mean_strain: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean_strain = (self.strain_a + self.strain_b) / 2.0

    def __repr__(self) -> str:  # pragma: no cover
        ma = self.matrix_a.tolist()
        mb = self.matrix_b.tolist()
        return (
            f"MatchResult(M_A={ma}, M_B={mb}, "
            f"strain_A={self.strain_a:.4f}, strain_B={self.strain_b:.4f}, "
            f"area={self.area:.2f} Å²)"
        )


def cell_from_params(lat_a: float, lat_b: float, ang_gamma_deg: float) -> np.ndarray:
    """Build a 2×2 row-vector lattice matrix from cell parameters.

    Args:
        lat_a: Lattice parameter a (Å).
        lat_b: Lattice parameter b (Å).
        ang_gamma_deg: In-plane angle γ (degrees).

    Returns:
        2×2 array where rows are lattice vectors::

            [[ax, ay],
             [bx, by]]
    """
    gamma = math.radians(ang_gamma_deg)
    return np.array([
        [lat_a, 0.0],
        [lat_b * math.cos(gamma), lat_b * math.sin(gamma)],
    ])


def _max_strain(cell_ref: np.ndarray, cell_def: np.ndarray) -> float:
    """Return the maximum absolute principal strain deforming *cell_def* → *cell_ref*.

    Both cells must have the same number of atoms / area sign so that the
    deformation gradient F = cell_ref @ inv(cell_def) is meaningful.

    Args:
        cell_ref: Target 2×2 cell (rows = vectors).
        cell_def: Starting 2×2 cell (rows = vectors).

    Returns:
        Max |principal strain - 1|.  Zero means perfect match.
    """
    try:
        F = cell_ref @ np.linalg.inv(cell_def)
    except np.linalg.LinAlgError:
        return math.inf
    # Polar decomposition: F = R U.  Eigenvalues of U are stretch ratios.
    U = F.T @ F
    eigvals = np.linalg.eigvalsh(U)
    # principal strains: λ_i = sqrt(eigenvalue) - 1
    stretches = np.sqrt(np.maximum(eigvals, 0.0))
    return float(np.max(np.abs(stretches - 1.0)))


def _enumerate_matrices(n: int):
    """Yield all 2×2 integer matrices with determinant in [1, n] and |entries| <= n.

    Only upper-triangular forms are generated to avoid counting rotationally
    equivalent supercells multiple times (Hermite Normal Form subset).

    Yields:
        2×2 np.ndarray of dtype int with positive determinant.
    """
    for a in range(1, n + 1):
        for c in range(1, n + 1):
            if a * c > n * n:
                break
            for b in range(0, c):
                det = a * c
                if 1 <= det <= n * n:
                    yield np.array([[a, b], [0, c]], dtype=int)


def find_coincidence_lattice(
    cell_a: np.ndarray,
    cell_b: np.ndarray,
    strain_tol: float = 0.02,
    max_supercell: int = 10,
    area_tol: float = 0.10,
) -> Optional[MatchResult]:
    """Find the smallest commensurate supercell for two 2D material cells.

    Implements the Zur-McGill coincidence-lattice algorithm: enumerates integer
    transformation matrices for both cells up to ``max_supercell`` repeat units,
    and accepts pairs whose mutual strain is below ``strain_tol``.  The best
    candidate is the one with the smallest supercell area.

    Args:
        cell_a: 2×2 lattice matrix for material A (row vectors, Å).
        cell_b: 2×2 lattice matrix for material B (row vectors, Å).
        strain_tol: Maximum allowed strain for either material (default 2 %).
        max_supercell: Maximum supercell repeat along each axis (default 10).
        area_tol: Relative tolerance on supercell area agreement (default 10 %).

    Returns:
        :class:`MatchResult` with the best (smallest-area, within-tolerance) match,
        or ``None`` if no match is found within the constraints.
    """
    best: Optional[MatchResult] = None

    for ma in _enumerate_matrices(max_supercell):
        sc_a = ma @ cell_a  # type: ignore[operator]
        area_a = abs(float(np.linalg.det(sc_a)))
        if area_a < 1.0:
            continue

        for mb in _enumerate_matrices(max_supercell):
            sc_b = mb @ cell_b  # type: ignore[operator]
            area_b = abs(float(np.linalg.det(sc_b)))

            # Areas must agree within area_tol before computing full strain.
            if abs(area_a - area_b) / max(area_a, area_b) > area_tol:
                continue

            # Compute strain: how much must each supercell deform to reach a
            # common mean cell?
            mean_cell = (sc_a + sc_b) / 2.0
            sa = _max_strain(mean_cell, sc_a)
            sb = _max_strain(mean_cell, sc_b)

            if sa > strain_tol or sb > strain_tol:
                continue

            area = abs(float(np.linalg.det(mean_cell)))

            if best is None or area < best.area:
                best = MatchResult(
                    matrix_a=ma.copy(),
                    matrix_b=mb.copy(),
                    supercell_a=sc_a.copy(),
                    supercell_b=sc_b.copy(),
                    strain_a=sa,
                    strain_b=sb,
                    area=area,
                )

    return best


def rotation_angle_from_match(match: MatchResult) -> float:
    """Extract the rotation angle (degrees) that aligns material B to material A.

    The deformation gradient ``F = supercell_a @ inv(supercell_b)`` is
    decomposed into ``F = R @ U`` via polar decomposition.  The rotation
    component ``R`` gives the angle material B must be rotated to produce a
    commensurate interface.

    Args:
        match: A :class:`MatchResult` from :func:`find_coincidence_lattice`.

    Returns:
        Rotation angle in degrees (counter-clockwise positive).  Returns 0.0
        when the two supercells are already aligned.
    """
    if match is None:
        return 0.0
    try:
        F = match.supercell_a @ np.linalg.inv(match.supercell_b)
        # Polar decomposition F = R @ U  (right polar)
        U2 = F.T @ F
        # Symmetric square-root of U²
        eigvals, eigvecs = np.linalg.eigh(U2)
        eigvals = np.maximum(eigvals, 0.0)
        U = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        R = F @ np.linalg.inv(U)
        angle_rad = math.atan2(R[1, 0], R[0, 0])
        return math.degrees(angle_rad)
    except np.linalg.LinAlgError:
        return 0.0


def find_coincidence_lattice_from_cif(
    cif_a: "str | None" = None,
    cif_b: "str | None" = None,
    lat_a: Optional[Tuple[float, float, float]] = None,
    lat_b: Optional[Tuple[float, float, float]] = None,
    strain_tol: float = 0.02,
    max_supercell: int = 10,
) -> Optional[MatchResult]:
    """Convenience wrapper that accepts CIF paths or ``(a, b, γ)`` tuples.

    Args:
        cif_a: Path to CIF file for material A.  Mutually exclusive with ``lat_a``.
        cif_b: Path to CIF file for material B.  Mutually exclusive with ``lat_b``.
        lat_a: ``(a, b, gamma_deg)`` tuple for material A.
        lat_b: ``(a, b, gamma_deg)`` tuple for material B.
        strain_tol: Forwarded to :func:`find_coincidence_lattice`.
        max_supercell: Forwarded to :func:`find_coincidence_lattice`.

    Returns:
        :class:`MatchResult` or ``None``.
    """
    from .utils import cifread  # local import to avoid circular dependency

    def _cell(cif_path, params):
        if cif_path is not None:
            data = cifread(cif_path)
            return cell_from_params(data['lat_a'], data['lat_b'], data['ang_g'])
        if params is not None:
            return cell_from_params(*params)
        raise ValueError("Provide either a CIF path or a (a, b, gamma) tuple.")

    cell_a = _cell(cif_a, lat_a)
    cell_b = _cell(cif_b, lat_b)
    return find_coincidence_lattice(cell_a, cell_b, strain_tol=strain_tol, max_supercell=max_supercell)


# =============================================================================
# Hod Registry Index (RI)
# =============================================================================

class StackingResult(NamedTuple):
    """Optimal stacking offsets derived from a Registry Index scan.

    All shifts are Cartesian (Å) displacements of layer B relative to layer A.

    Attributes:
        aa_shift: Shift that maximises the RI (AA-like registry, highest friction).
        ab_shift: Shift that minimises the RI (AB anti-registry, superlubricity).
        aa_ri: Registry Index value at the AA site (≈ 1.0).
        ab_ri: Registry Index value at the AB site (≈ 0.0).
        ri_grid: 2-D array of RI values over the scan grid (shape n_grid × n_grid).
        shifts_x: Cartesian x-coordinates of the grid points (Å).
        shifts_y: Cartesian y-coordinates of the grid points (Å).
    """
    aa_shift: np.ndarray
    ab_shift: np.ndarray
    aa_ri: float
    ab_ri: float
    ri_grid: np.ndarray
    shifts_x: np.ndarray
    shifts_y: np.ndarray


def generate_supercell_positions(
    frac_pos: np.ndarray,
    unit_cell: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """Tile a unit cell into a supercell and return Cartesian atom positions.

    Args:
        frac_pos: (N, 2) fractional coordinates of atoms in the unit cell.
        unit_cell: 2×2 row-vector lattice matrix of the unit cell (Å).
        matrix: 2×2 integer supercell transformation matrix (upper-triangular,
            as returned by :func:`find_coincidence_lattice`).

    Returns:
        (M, 2) Cartesian positions (Å) of all atoms in the supercell, where
        M = N × det(matrix).
    """
    m = int(round(abs(np.linalg.det(matrix))))
    if m < 1:
        raise ValueError("Supercell matrix has non-positive determinant.")

    matrix_inv = np.linalg.inv(matrix)

    # Enumerate all integer lattice-vector offsets (i, j) within ±n of origin.
    # A candidate image of atom f in the unit cell is f + (i,j), which maps to
    # supercell fractional coordinates (f + (i,j)) @ inv(M).
    n = int(math.ceil(math.sqrt(m))) + 2
    i_range = np.arange(-n, n + 1)
    ii, jj = np.meshgrid(i_range, i_range)
    translations = np.column_stack([ii.ravel(), jj.ravel()])  # (K, 2)

    all_cart: List[np.ndarray] = []
    for f in frac_pos:
        images_uc = f + translations                   # (K, 2) unit-cell fractional
        images_sc = images_uc @ matrix_inv             # (K, 2) supercell fractional
        mask = np.all((images_sc >= -1e-6) & (images_sc < 1.0 - 1e-6), axis=1)
        all_cart.append(images_uc[mask] @ unit_cell)  # convert kept images to Å

    return np.vstack(all_cart)


def _circle_overlap_area(d: float, r: float) -> float:
    """Area of intersection of two circles of equal radius *r* with centres distance *d* apart.

    Returns the analytical two-disk overlap (Hod eq. A1).
    """
    if d >= 2.0 * r:
        return 0.0
    if d <= 0.0:
        return math.pi * r * r
    half_d = d / 2.0
    arg = min(half_d / r, 1.0)
    return 2.0 * r * r * math.acos(arg) - half_d * math.sqrt(max(4.0 * r * r - d * d, 0.0))


def _ri_raw(
    cart_a: np.ndarray,
    cart_b: np.ndarray,
    cell: np.ndarray,
    shift_cart: np.ndarray,
    radius: float,
) -> float:
    """Unnormalised overlap integral at a single shift.

    Sums circle-overlap areas for all (i, j) pairs within 2*radius using minimum
    image convention in the common supercell ``cell``.

    Args:
        cart_a: (N_a, 2) Cartesian positions of layer A atoms (Å).
        cart_b: (N_b, 2) Cartesian positions of layer B atoms (Å).
        cell: 2×2 common supercell lattice (row vectors, Å).
        shift_cart: (2,) Cartesian shift applied to layer B (Å).
        radius: Effective atomic radius for disk-overlap model (Å).

    Returns:
        Raw overlap sum (Å²).
    """
    cell_inv = np.linalg.inv(cell)
    b_shifted = cart_b + shift_cart

    total = 0.0
    cutoff = 2.0 * radius

    for pa in cart_a:
        diff = b_shifted - pa                          # (N_b, 2) Cartesian
        frac_diff = diff @ cell_inv                    # fractional in supercell
        frac_diff -= np.round(frac_diff)               # minimum image
        cart_diff = frac_diff @ cell
        dists = np.linalg.norm(cart_diff, axis=1)      # (N_b,)
        nearby = dists[dists < cutoff]
        for d in nearby:
            total += _circle_overlap_area(d, radius)

    return total


def scan_registry_index(
    cart_a: np.ndarray,
    cart_b: np.ndarray,
    cell: np.ndarray,
    n_grid: int = 30,
    atomic_radius: float = 1.5,
) -> StackingResult:
    """Scan the Hod Registry Index over a grid of lateral shifts.

    The RI is normalised so that the maximum value (AA registry) equals 1 and the
    minimum (optimal AB anti-registry) approaches 0.

    Args:
        cart_a: (N_a, 2) Cartesian positions of layer A atoms (Å).
        cart_b: (N_b, 2) Cartesian positions of layer B atoms (Å).
        cell: 2×2 common supercell lattice matrix (row vectors, Å).
        n_grid: Number of grid points along each cell vector (default 30).
        atomic_radius: Effective atomic radius for the disk-overlap model (Å).
            A value of 1.5 Å works well for most 2D materials; adjust for heavier
            atoms (e.g., 1.8 Å for Bi or Te).

    Returns:
        :class:`StackingResult` with AA/AB shifts, RI values, and the full grid.
    """
    # Build fractional grid [0, 1) × [0, 1) and convert to Cartesian.
    fx = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    fy = np.linspace(0.0, 1.0, n_grid, endpoint=False)

    ri_grid = np.zeros((n_grid, n_grid))
    raw_grid = np.zeros((n_grid, n_grid))

    for i, xi in enumerate(fx):
        for j, yj in enumerate(fy):
            shift = np.array([xi, yj]) @ cell       # Cartesian
            raw_grid[i, j] = _ri_raw(cart_a, cart_b, cell, shift, atomic_radius)

    raw_max = raw_grid.max()
    if raw_max > 0.0:
        ri_grid = raw_grid / raw_max
    else:
        ri_grid = raw_grid.copy()

    # Cartesian coordinates of each grid point.
    shifts_cart = np.array([[xi, yj] for xi in fx for yj in fy]) @ cell
    sx = shifts_cart[:, 0].reshape(n_grid, n_grid)
    sy = shifts_cart[:, 1].reshape(n_grid, n_grid)

    aa_idx = np.unravel_index(np.argmax(ri_grid), ri_grid.shape)
    ab_idx = np.unravel_index(np.argmin(ri_grid), ri_grid.shape)

    aa_shift = np.array([sx[aa_idx], sy[aa_idx]])
    ab_shift = np.array([sx[ab_idx], sy[ab_idx]])

    return StackingResult(
        aa_shift=aa_shift,
        ab_shift=ab_shift,
        aa_ri=float(ri_grid[aa_idx]),
        ab_ri=float(ri_grid[ab_idx]),
        ri_grid=ri_grid,
        shifts_x=sx,
        shifts_y=sy,
    )

