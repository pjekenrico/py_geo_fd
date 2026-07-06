import json
from pathlib import Path

import pytest

from py_geo_fd.stent_config import compute_w_lw, load_config


def _valid_config_dict(tmp_path: Path) -> dict:
    return {
        "centerline_params": {
            "centerline_path": str(tmp_path / "centerline.vtp"),
            "x_smoothing": 0.01,
            "dx_smoothing": 0.001,
            "s_start": 1.0,
        },
        "stent_params": {
            "geo_params": {
                "w": 0.17,
                "lw": 13.0,
                "d_nom": 2.5,
                "Nw": 52,
                "wire_radius": 0.015,
            },
            "numeric_params": {
                "wire_resolution": 20,
                "n_segments": 200,
            },
        },
        "wall_params": {
            "wall_path": str(tmp_path / "surface.stl"),
            "smoothing": 1,
            "margin": 0.0,
            "n_axial_segments": 16,
            "n_radial_segments": 12,
        },
        "n_cpu": 1,
    }


def test_load_config_success(tmp_path: Path) -> None:
    cfg = _valid_config_dict(tmp_path)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(cfg_path)

    assert loaded.n_cpu == 1
    assert loaded.st.geom.Nw == 52
    assert loaded.ctrl.s_start == pytest.approx(1.0)


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("/tmp/this_file_should_not_exist_123456.json")


def test_load_config_invalid_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not-json", encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(bad)


def test_compute_w_lw_updates_file(tmp_path: Path) -> None:
    stent_data = {
        "MODEL_A": {
            "Nw": 52,
            "d": [2.0, 2.5],
            "L": [11.0, 8.0],
        }
    }
    path = tmp_path / "stent_data.json"
    path.write_text(json.dumps(stent_data), encoding="utf-8")

    compute_w_lw(path)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert "w" in written["MODEL_A"]
    assert "lw" in written["MODEL_A"]
    assert written["MODEL_A"]["wire_radius"] == pytest.approx(0.015)
