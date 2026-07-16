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


def _write_charged_square_monolayer(path, a=3.0, charges=(0.1, -0.1, 0.2, -0.2)):
    # 2x2 atoms on a square lattice, atom_style=charge with distinct per-atom
    # charges -- a bare ASE lammps-data round-trip silently drops these.
    pos = [(0, 0, 5), (a, 0, 5), (0, a, 5), (a, a, 5)]
    atoms = Atoms("H4", positions=pos, cell=[2 * a, 2 * a, 20], pbc=[True, True, False])
    atoms.set_initial_charges(list(charges))
    write(str(path), atoms, format="lammps-data", atom_style="charge")


def test_strain_preserves_charges_and_atom_style(tmp_path):
    src = tmp_path / "in_charge.data"
    out = tmp_path / "out_charge.data"
    charges = (0.1, -0.1, 0.2, -0.2)
    _write_charged_square_monolayer(src, a=3.0, charges=charges)   # cell 6.0 x 6.0
    target = np.array([[6.3, 0.0], [0.0, 6.3]])                    # +5% isotropic

    strain = strain_monolayer_to_cell(str(src), str(out), target)
    assert 0.049 < strain < 0.051

    # The output must still declare atom_style=charge in its Atoms header.
    out_text = out.read_text(encoding="utf-8")
    header_line = next(l for l in out_text.splitlines() if l.strip().startswith("Atoms"))
    assert "charge" in header_line

    got = read(str(out), format="lammps-data", atom_style="charge")
    got_charges = np.array(got.get_initial_charges())
    # Charges are carried per-atom; matching by value (order may permute) is
    # sufficient to prove they were not dropped or zeroed.
    assert np.allclose(sorted(got_charges), sorted(charges), atol=1e-9)
