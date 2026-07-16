import numpy as np
from ase import Atoms
from ase.io import write, read
from src.core.lattice_apply import strain_monolayer_to_cell


def _write_square_monolayer(path, a=3.0):
    # 2x2 atoms on a square lattice, thin slab in z
    pos = [(0, 0, 5), (a, 0, 5), (0, a, 5), (a, a, 5)]
    atoms = Atoms("H4", positions=pos, cell=[2 * a, 2 * a, 20], pbc=[True, True, False])
    write(str(path), atoms, format="lammps-data")


def test_strain_scales_cell_and_positions(tmp_path):
    src = tmp_path / "in.data"
    out = tmp_path / "out.data"
    _write_square_monolayer(src, a=3.0)                 # in-plane cell = 6.0 x 6.0
    target = np.array([[6.3, 0.0], [0.0, 6.3]])          # +5% isotropic
    strain = strain_monolayer_to_cell(str(src), str(out), target)

    got = read(str(out), format="lammps-data")
    cell = np.array(got.cell)[:2, :2]
    assert np.allclose(cell, target, atol=1e-6)          # box now equals target
    assert 0.049 < strain < 0.051                        # ~5% principal strain
    # z-height unchanged
    assert abs(np.array(got.cell)[2, 2] - 20.0) < 1e-6


def test_strain_is_zero_when_target_equals_source(tmp_path):
    src = tmp_path / "in.data"
    out = tmp_path / "out.data"
    _write_square_monolayer(src, a=3.0)                 # cell 6.0 x 6.0
    target = np.array([[6.0, 0.0], [0.0, 6.0]])          # identical
    strain = strain_monolayer_to_cell(str(src), str(out), target)
    assert strain < 1e-9
