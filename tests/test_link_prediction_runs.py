from __future__ import annotations

import json

from finsler_mds.link_prediction.runs import create_run_directory, save_json


def test_run_directories_are_unique_and_keep_their_configuration(tmp_path):
    first = create_run_directory(
        tmp_path,
        dataset="chameleon",
        metric="randers",
        dimensions=(5, 10, 20, 50),
        alpha_max=0.8,
        num_trials=20,
        protocol="test_v1",
    )
    second = create_run_directory(
        tmp_path,
        dataset="chameleon",
        metric="randers",
        dimensions=(5, 10, 20, 50),
        alpha_max=0.8,
        num_trials=20,
        protocol="test_v1",
    )

    assert first != second
    assert first.parent == second.parent == tmp_path / "runs"
    assert first.name.endswith("_test_v1")
    save_json(first / "config.json", {"dimension": 10, "alpha_max": 0.8})
    assert json.loads((first / "config.json").read_text()) == {
        "dimension": 10,
        "alpha_max": 0.8,
    }
