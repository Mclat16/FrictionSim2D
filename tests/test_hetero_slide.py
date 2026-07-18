"""Tests for D1 hetero sheet-on-sheet slide assembly helpers."""
from pathlib import Path
import numpy as np
from ase.io import read
from src.core.config import SheetOnSheetSimulationConfig, load_settings
from src.builders.hetero import build_hetero_structure
from src.builders.hetero_slide import compute_layer_zbands

MAT, POT = "examples/materials", "examples/potentials/sw"


def _hetero_2p2_config():
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": f"{MAT}/h-MoS2.cif", "pot_path": f"{POT}/MoS2_wen.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [2]},
        "2D-2": {"mat": "h-WS2", "cif_path": f"{MAT}/h-WS2.cif", "pot_path": f"{POT}/sw_lammps/t-WS2.sw",
                 "pot_type": "sw", "x": 12.0, "y": 12.0, "layers": [2]},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def test_layer_zbands_partition_all_atoms(tmp_path):
    cfg = _hetero_2p2_config()
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    bands = compute_layer_zbands(stack.data_path, stack.layers)
    assert [b["idx"] for b in bands] == [1, 2, 3, 4]           # 2+2 -> 4 layers, bottom->top
    # ordered, non-overlapping
    for lo, hi in ((bands[i]["zhi"], bands[i+1]["zlo"]) for i in range(3)):
        assert lo <= hi + 1e-9
    # every atom falls in exactly one band
    z = read(str(stack.data_path), format="lammps-data").get_positions()[:, 2]
    for zi in z:
        hits = [b for b in bands if b["zlo"] <= zi <= b["zhi"]]
        assert len(hits) == 1, f"z={zi} in {len(hits)} bands"
