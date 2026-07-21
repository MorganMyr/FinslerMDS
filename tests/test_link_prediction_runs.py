from __future__ import annotations

import json

from finsler_mds.link_prediction.runs import create_run_directory, save_json


def test_run_directories_are_unique_and_keep_their_configuration(tmp_path):
    first = create_run_directory(
        tmp_path,
        "chameleon",
        "randers",
        "n20",
    )
    second = create_run_directory(
        tmp_path,
        "chameleon",
        "randers",
        "n20",
    )

    assert first != second
    assert first.parent == second.parent == tmp_path / "runs"
    assert first.name.endswith("_chameleon_randers_n20")
    save_json(first / "config.json", {"dimension": 10, "alpha_max": 0.8})
    assert json.loads((first / "config.json").read_text()) == {
        "dimension": 10,
        "alpha_max": 0.8,
    }
