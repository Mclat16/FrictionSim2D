"""Tests for indent delivery / postprocessing (src/postprocessing/indent.py)."""

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.postprocessing import indent


# --- unit helpers ----------------------------------------------------------

def test_material_id_underscores():
    assert indent.material_id("h-MoS2") == "h_MoS2"
    assert indent.material_id("a/b") == "a__b"


def test_load_token():
    assert indent._load_token(10.0) == "10"
    assert indent._load_token(30) == "30"
    assert indent._load_token(10.5) == "10.5"


def test_parse_wall_seconds():
    assert indent._parse_wall_seconds("0:00:12") == 12
    assert indent._parse_wall_seconds("1:02:03") == 3723
    assert indent._parse_wall_seconds("45") == 45


def test_split_by_load_groups_per_load_and_seed():
    df = object()  # opaque leaf; split only reshapes the tree
    tree = {
        "h_MoS2": {"60x60y": {"aSi": {"Si": {"r25": {"l1": {"s2": {
            "f10.0": {"a0": df}, "f30.0": {"a0": df},
        }}}}}}},
    }
    per_file = indent.split_by_load(tree, base_seed={"h_MoS2": 777})
    assert set(per_file.keys()) == {("f10.0", 777), ("f30.0", 777)}
    leaf = per_file[("f10.0", 777)]["h_MoS2"]["aSi"]["Si"]["r25"]["l1"]["s2"]
    assert set(leaf.keys()) == {"f10.0"}                       # only that load kept


# --- retract_features reduction --------------------------------------------

def _synthetic_curve():
    """A physical pull-off curve: +repulsive wall -> zero -> adhesive well -> ~0."""
    z = np.linspace(0.0, 20.0, 101)
    fz = 6.0 * np.exp(-z / 0.6) - 2.0 * np.exp(-((z - 2.0) ** 2) / (2 * 0.9 ** 2))
    return pd.DataFrame({
        "step": np.arange(z.size), "z_tip": z + 80.0, "fz_tip": fz,
        "fx_tip": 0.0 * z, "fy_tip": 0.0 * z, "pe": np.cumsum(fz) * 0.2,
    })


def test_retract_features_pulloff_and_work_positive():
    feats = indent.retract_features(_synthetic_curve())
    assert feats["f_adh_min"] < 0
    assert feats["pulloff_force"] == -feats["f_adh_min"]
    assert 1.5 < feats["pulloff_force"] < 2.0
    assert feats["work_of_adhesion"] > 0
    assert feats["detached"] == 1.0


def test_contact_stiffness_is_the_steep_wall_not_the_tail():
    feats = indent.retract_features(_synthetic_curve())
    assert 3.0 < feats["contact_stiffness"] < 7.0


def test_contact_stiffness_nan_without_repulsive_start():
    z = np.linspace(0.0, 5.0, 20)
    df = pd.DataFrame({"z_tip": z, "fz_tip": -np.ones_like(z)})  # purely attractive
    feats = indent.retract_features(df)
    assert math.isnan(feats["contact_stiffness"])
    assert feats["pulloff_force"] == 1.0


# --- fixture: a minimal indent run tree (hold files + retract curves) -------

_HOLD_HEADER = ("# Time-averaged data for fix fc_ave\n"
                "# TimeStep nf lfx lfy comx comy comz tipx tipy tipz\n")


def _write_hold(dir_: Path, load: int, layer: int, tipz: float, comz: float):
    dir_.mkdir(parents=True, exist_ok=True)
    rows = "".join(
        f"{step} 30.0 0.0 0.0 30.0 30.0 {comz} 30.0 30.0 {tipz}\n"
        for step in (1000, 2000, 3000, 4000, 5000)
    )
    (dir_ / f"friction_f{load}_a0_s2_layer{layer}.txt").write_text(
        _HOLD_HEADER + rows, encoding="utf-8")


def _make_run(root: Path, seed=20260710, hold_steps=10000, z_step=0.2, n_steps=100):
    base = root / "indent" / "h-MoS2" / "60x_60y" / "sub_aSi_tip_Si_r25" / "K300"
    for layer in (1, 2):
        results = base / f"L{layer}" / "results"
        # Penetration (hold) files, one per load.
        _write_hold(results, load=10, layer=layer, tipz=12.0, comz=5.0)
        _write_hold(results, load=30, layer=layer, tipz=11.0, comz=5.0)
        # Adhesion (retract) curves, one per load.
        for load in (10, 30):
            _synthetic_curve().to_csv(results / f"indent_f{load}N_l{layer}.csv", index=False)
    prov = base / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "indent_meta.json").write_text(json.dumps({
        "base_seed": seed, "hold_steps": hold_steps, "z_step": z_step, "n_steps": n_steps,
        "material": "h-MoS2", "layers": {"1": 111111, "2": 222222}, "loads": [10.0, 30.0],
    }), encoding="utf-8")
    return root


def test_load_indent_meta(tmp_path: Path):
    _make_run(tmp_path)
    meta = indent.load_indent_meta(tmp_path)
    assert meta["base_seed"]["h_MoS2"] == 20260710
    assert meta["hold_steps"]["h_MoS2"] == 10000
    assert meta["z_step"]["h_MoS2"] == 0.2
    assert meta["n_steps"]["h_MoS2"] == 100
    assert meta["seed"][("h_MoS2", 1)] == 111111
    assert meta["seed"][("h_MoS2", 2)] == 222222


def test_discover_curves_parses_material_layer_load(tmp_path: Path):
    _make_run(tmp_path)
    found = indent.discover_curves(tmp_path)
    conds = {(m, l, ld) for m, l, ld, _p in found}
    assert conds == {("h_MoS2", 1, 10.0), ("h_MoS2", 1, 30.0),
                     ("h_MoS2", 2, 10.0), ("h_MoS2", 2, 30.0)}


def test_run_writes_penetration_json(tmp_path: Path):
    _make_run(tmp_path)
    out = tmp_path / "outputs" / "indent_sims"
    written = indent.run(tmp_path, out)

    # One penetration JSON per load, named by base seed.
    assert "output_indent_f10_s20260710.json" in written
    assert "output_indent_f30_s20260710.json" in written

    payload = json.loads((out / "output_indent_f10_s20260710.json").read_text())
    leaf = (payload["results"]["h_MoS2"]["aSi"]["Si"]["r25"]
            ["l1"]["s2"]["f10.0"]["a0"])
    assert "columns" in leaf and "data" in leaf
    for col in ("nf", "comz", "tipz"):
        assert col in leaf["columns"]                          # penetration channels
    assert payload["metadata"]["indent_load_nN"] == 10.0


def test_run_writes_adhesion_features_and_curves(tmp_path: Path):
    _make_run(tmp_path)
    out = tmp_path / "outputs" / "indent_sims"
    data = tmp_path / "data"
    written = indent.run(tmp_path, out, data_dir=data)

    assert "indent_features.csv" in written
    feats = pd.read_csv(data / "indent_features.csv")
    assert set(feats["layer"]) == {1, 2}
    assert {"pulloff_force", "work_of_adhesion", "contact_stiffness"} <= set(feats.columns)

    # Tidied per-condition curves; two loads per (material, layer) -> load token.
    assert (out / "indent" / "h_MoS2_l1_f10.csv").exists()
    assert (out / "indent" / "h_MoS2_l2_f30.csv").exists()


def test_run_writes_combined_manifest(tmp_path: Path):
    _make_run(tmp_path)
    out = tmp_path / "outputs" / "indent_sims"
    written = indent.run(tmp_path, out)

    assert "indent_manifest.csv" in written
    with (out / "indent_manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 2 layers x 2 loads = 4 conditions, deduped across hold + retract sources.
    assert len(rows) == 4
    keyed = {(r["layer"], r["load"]): r for r in rows}
    assert keyed[("1", "10.0")]["seed"] == "111111"
    assert keyed[("2", "30.0")]["seed"] == "222222"
    assert keyed[("1", "10.0")]["hold_steps"] == "10000"
    assert keyed[("2", "10.0")]["z_step"] == "0.2"
    assert keyed[("2", "10.0")]["n_steps"] == "100"


def test_run_returns_empty_when_no_indent(tmp_path: Path):
    assert indent.run(tmp_path, tmp_path / "out") == {}
