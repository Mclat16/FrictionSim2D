"""Tests for the AFM indentation ("indent") builder (src/builders/indent.py)."""

from pathlib import Path

from src.builders.indent import (
    DEFAULT_HOLD_STEPS,
    DEFAULT_N_STEPS,
    DEFAULT_Z_STEP,
    INDENT_MINIMIZE_COMMAND,
    IndentSimulation,
    derive_seed,
)
from src.core.config import AFMSimulationConfig, load_settings


def _afm_cfg(tmp_path: Path, force, indent=None):
    pot = tmp_path / "d.sw"; pot.write_text("# pot", encoding="utf-8")
    cif = tmp_path / "d.cif"; cif.write_text("# cif", encoding="utf-8")
    payload = {
        "general": {"temp": 300.0, "force": force, "scan_angle": 0,
                    "scan_speed": 2.0, "finite_sheet": False},
        "2D": {"mat": "h-MoS2", "pot_type": "sw", "pot_path": str(pot),
               "cif_path": str(cif), "x": 60.0, "y": 60.0, "layers": [1, 2]},
        "tip": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "r": 25.0, "amorph": "c"},
        "sub": {"mat": "Si", "pot_type": "sw", "pot_path": str(pot),
                "cif_path": str(cif), "thickness": 12.0, "amorph": "a"},
        "settings": load_settings().model_dump(),
    }
    if indent is not None:
        payload["indent"] = indent
    return AFMSimulationConfig(**payload)


def _indent_context(**overrides):
    """Minimal render context for afm/indent.lmp (hold + retract)."""
    ctx = {
        # --- hold phase ---
        "scan_angle_config": 0,
        "scan_speed_config": 2,
        "forces": [10, 30],
        "atom_style": "atomic",
        "neighbor_list": 0.3,
        "neigh_modify_command": "neigh_modify every 1 delay 0 check yes",
        "timestep": 0.001,
        "thermo": 1000,
        "output_dir": "indent/h-MoS2/60x_60y/sub_aSi_tip_Si_r25/K300/L1/data",
        "potential_file": "indent/.../lammps/system.in.settings",
        "sheet_types": "5 6",
        "finite_sheet_enabled": False,
        "finite_sheet_edge_mode": "none",
        "finite_sheet_edge_spring_k": 10.0,
        "dump_enabled": False,
        "dump_freq": 1000,
        "dump_file_pattern": "indent/.../visuals/slide.lammpstrj",
        "tip_fix_group": "tip_fix",
        "thermostat_type": "langevin",
        "temp": 300,
        "layer_group": "2D_all",
        "ev_a_to_nn": 1.602176565,
        "results_freq": 1000,
        "results_file_pattern": "indent/.../results/friction_f$(v_find)_a${a}_s${speed}_layer1.txt",
        "damp_ev": None,
        "damping_ratio": 0.0072,
        "spring_ev": 0.5,
        "hold_steps": 10000,
        # --- retract phase ---
        "min_style": "cg",
        "retract_minimize": INDENT_MINIMIZE_COMMAND,
        "results_csv": "indent/.../results/indent_f$(v_find)N_l1.csv",
        "z_step": 0.2,
        "n_steps": 100,
    }
    ctx.update(overrides)
    return ctx


# --- derive_seed -----------------------------------------------------------

def test_derive_seed_is_deterministic_and_in_range():
    a = derive_seed(42, "h-MoS2", 1)
    b = derive_seed(42, "h-MoS2", 1)
    assert a == b
    assert 1 <= a <= 999983


def test_derive_seed_varies_by_material_and_layer():
    base = 42
    assert derive_seed(base, "h-MoS2", 1) != derive_seed(base, "h-WS2", 1)
    assert derive_seed(base, "h-MoS2", 1) != derive_seed(base, "h-MoS2", 2)
    assert derive_seed(1, "h-MoS2", 1) != derive_seed(2, "h-MoS2", 1)


# --- IndentSimulation config wiring ----------------------------------------

def test_defaults_when_no_indent_section(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10, 30]), output_dir=str(tmp_path / "o"))
    assert builder.hold_steps == DEFAULT_HOLD_STEPS
    assert builder.z_step == DEFAULT_Z_STEP
    assert builder.n_steps == DEFAULT_N_STEPS
    assert 1 <= builder.base_seed <= 999999  # freshly drawn


def test_reads_indent_params_and_seed(tmp_path: Path):
    cfg = _afm_cfg(tmp_path, force=[10, 30],
                   indent={"hold_steps": 25000, "z_step": 0.1, "n_steps": 150, "seed": 777})
    builder = IndentSimulation(cfg, output_dir=str(tmp_path / "o"))
    assert builder.hold_steps == 25000
    assert builder.z_step == 0.1
    assert builder.n_steps == 150
    assert builder.base_seed == 777


def test_slide_template_is_the_indent():
    assert IndentSimulation.SLIDE_TEMPLATE == "afm/indent.lmp"


def test_hpc_job_name(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10, 30]), output_dir=str(tmp_path / "o"))
    assert builder._get_hpc_job_name() == "indent_h-MoS2"


def test_velocity_seed_records_and_matches_derivation(tmp_path: Path):
    cfg = _afm_cfg(tmp_path, force=[10, 30], indent={"seed": 100})
    builder = IndentSimulation(cfg, output_dir=str(tmp_path / "o"))
    s1 = builder._velocity_seed(1)
    s2 = builder._velocity_seed(2)
    assert s1 == derive_seed(100, "h-MoS2", 1)
    assert s2 == derive_seed(100, "h-MoS2", 2)
    assert builder.indent_seeds == {1: s1, 2: s2}
    assert s1 != s2  # fresh per layer


def test_build_render_context_hook_adds_hold_and_retract_keys(tmp_path: Path):
    """The indent build_render_context adds both hold + retract keys on top of AFM's."""
    cfg = _afm_cfg(tmp_path, force=[10], indent={"hold_steps": 12345, "z_step": 0.15, "n_steps": 80})
    builder = IndentSimulation(cfg, output_dir=str(tmp_path / "o"))
    from pathlib import PurePosixPath
    builder.relative_run_dir_layer = {1: PurePosixPath("indent/h-MoS2/60x_60y/x/K300/L1")}
    import src.builders.afm as afm_mod
    orig = afm_mod.AFMSimulation.build_render_context
    afm_mod.AFMSimulation.build_render_context = lambda self, n: {"n": n}
    try:
        ctx = builder.build_render_context(1)
    finally:
        afm_mod.AFMSimulation.build_render_context = orig
    assert ctx["n"] == 1
    assert ctx["hold_steps"] == 12345
    assert ctx["z_step"] == 0.15
    assert ctx["n_steps"] == 80
    assert ctx["retract_minimize"] == INDENT_MINIMIZE_COMMAND
    assert "indent_f$(v_find)N_l1.csv" in ctx["results_csv"]


# --- afm/indent.lmp rendering: hold phase ----------------------------------

def test_indent_renders_hold_phase(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10, 30]), output_dir=str(tmp_path / "o"))
    script = builder.render_template("afm/indent.lmp", _indent_context())

    # Reads the loaded contact state and re-applies the load, then holds.
    assert "load_$(v_find)N.data" in script
    assert "aveforce 0.0 0.0 $n" in script
    assert "run             10000" in script                 # hold_steps
    # Rigid tip, free only in z (laterally fixed) — a vertical hold, not a slide.
    assert "rigid/nve single force * off off on" in script
    # Records the out-of-plane channels the penetration descriptor needs.
    assert "v_fz_tip" in script and "v_comz" in script and "v_comz_tip" in script


def test_indent_auto_damping_when_dspring_none(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10]), output_dir=str(tmp_path / "o"))
    script = builder.render_template("afm/indent.lmp", _indent_context(damp_ev=None))
    assert "viscous $(" in script                            # auto zeta-of-critical


# --- afm/indent.lmp rendering: teardown + retract phase --------------------

def test_indent_tears_down_md_fixes_before_retract(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10]), output_dir=str(tmp_path / "o"))
    script = builder.render_template("afm/indent.lmp", _indent_context())
    # The langevin thermostat + aveforce + viscous + rigid fixes must be removed so
    # the minimization sees only conservative forces (else it corrupts the pull-off).
    for unfixed in ("unfix           fc_ave", "unfix           forcetip",
                    "unfix           lang_tip", "unfix           lang_sub",
                    "unfix           nve_all", "unfix           tip_f"):
        assert unfixed in script


def test_indent_renders_retract_phase(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10]), output_dir=str(tmp_path / "o"))
    script = builder.render_template("afm/indent.lmp", _indent_context())
    # Freezes the whole tip and displaces it up in z-steps, minimising each step.
    assert "fix             tip_hold tip_all setforce 0.0 0.0 0.0" in script
    assert "displace_atoms  tip_all move 0.0 0.0 ${zstep} units box" in script
    assert "variable        k loop 100" in script                 # n_steps
    assert "variable        zstep equal 0.2" in script            # z_step
    # Reads the surface's z-force on the tip (the pull-off signal) and writes a CSV.
    assert "f_tip_hold[3]" in script
    assert "step,z_tip,fz_tip,fx_tip,fy_tip,pe" in script
    # Force-converged minimize for a clean curve (not the coarse sliding default).
    assert "minimize 0.0 1.0e-6 10000 100000" in script


def test_indent_never_slides(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10]), output_dir=str(tmp_path / "o"))
    script = builder.render_template("afm/indent.lmp", _indent_context())
    assert "smd cvel" not in script
    assert "spring couple" not in script


def test_indent_multi_load_loops(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10, 30]), output_dir=str(tmp_path / "o"))
    script = builder.render_template("afm/indent.lmp", _indent_context(forces=[10, 30]))
    assert "variable        find index 10 30" in script
    assert "label           force_loop" in script
    assert "jump            SELF force_loop" in script


def test_indent_single_load_no_force_loop(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10]), output_dir=str(tmp_path / "o"))
    script = builder.render_template("afm/indent.lmp", _indent_context(forces=10))
    assert "variable        find equal 10" in script
    assert "force_loop" not in script


# --- system_init.lmp velocity-seed injection -------------------------------

def _system_init_context(**overrides):
    ctx = {
        "atom_style": "atomic", "neighbor_list": 0.3,
        "neigh_modify_command": "neigh_modify every 1 delay 0 check yes",
        "xlo": 0, "xhi": 60, "ylo": 0, "yhi": 60, "zhi_box": 120, "ngroups": 6,
        "sub_file": "build/sub.lmp", "tip_file": "build/tip.lmp", "sheet_file": "build/sheet.lmp",
        "tip_x": 30, "tip_y": 30, "tip_z": 60, "sub_natypes": 2, "offset_2d": 4,
        "sheet_shift_x": 0, "sheet_shift_y": 0, "sheet_z": 15,
        "flake_enabled": False, "potential_file": "lammps/system.in.settings",
        "dump_enabled": False, "min_style": "cg",
        "minimization_command": "minimize 1e-4 1e-8 1000 1000",
        "finite_sheet_enabled": False, "finite_sheet_edge_mode": "none",
        "finite_sheet_edge_spring_k": 10.0, "timestep": 0.001, "thermo": 1000,
        "tip_fix_group": "tip_fix", "temp": 300, "thermostat_type": "langevin",
        "forces": [10, 30], "ev_a_to_nn": 1.602176565, "output_dir": "L1/data",
        "velocity_seed": None,
    }
    ctx.update(overrides)
    return ctx


def test_system_init_injects_explicit_velocity_seed(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10]), output_dir=str(tmp_path / "o"))
    ctx = _system_init_context(velocity_seed=424242)
    script = builder.render_template("afm/system_init.lmp", ctx)
    assert "velocity        system create 300 424242" in script


def test_system_init_random_seed_when_none(tmp_path: Path):
    builder = IndentSimulation(_afm_cfg(tmp_path, force=[10]), output_dir=str(tmp_path / "o"))
    ctx = _system_init_context(velocity_seed=None)
    script = builder.render_template("afm/system_init.lmp", ctx)
    line = next(l for l in script.splitlines() if "system create" in l)
    assert "None" not in line
    assert int(line.split()[-1]) >= 1                        # a real random seed
