from __future__ import annotations

import gzip
from hashlib import sha256
from zipfile import ZipFile

from finsler_mds.link_prediction.datasets import (
    DATASET_SPECS,
    EdgeListDatasetSpec,
    load_edge_list_dataset,
)


def test_table2_datasets_are_pinned():
    expected = {
        "arxiv_year": (169_343, 1_166_243, 8_444, 1_157_799),
        "chameleon": (2_277, 36_051, 4_680, 31_371),
        "citeseer": (3_327, 4_608, 56, 4_552),
        "cora": (2_708, 5_429, 151, 5_278),
        "squirrel": (5_201, 216_933, 18_580, 198_353),
    }
    for name, values in expected.items():
        spec = DATASET_SPECS[name]
        assert (
            spec.num_nodes,
            spec.num_non_loop_edges,
            spec.num_reciprocal_pairs,
            spec.num_unordered_pairs,
        ) == values


def test_edge_list_loader_checks_processed_statistics(tmp_path):
    content = b"source\ttarget\n0\t1\n1\t0\n1\t2\n2\t2\n"
    spec = EdgeListDatasetSpec(
        name="tiny",
        url="https://example.test/tiny.tsv",
        sha256=sha256(content).hexdigest(),
        num_nodes=3,
        raw_num_edges=4,
        num_non_loop_edges=3,
        num_reciprocal_pairs=1,
        num_unordered_pairs=2,
        delimiter="\t",
        skip_header=1,
    )
    path = tmp_path / spec.name / "tiny.tsv"
    path.parent.mkdir()
    path.write_bytes(content)

    graph = load_edge_list_dataset(spec, root=tmp_path, download=False)
    assert graph.statistics().as_dict() == {
        "num_nodes": 3,
        "num_directed_edges": 3,
        "num_self_loops": 0,
        "num_non_loop_directed_edges": 3,
        "num_reciprocal_pairs": 1,
        "num_unordered_pairs": 2,
    }


def test_edge_list_loader_reads_a_gzip_member_from_zip(tmp_path):
    path = tmp_path / "tiny_zip" / "tiny.zip"
    path.parent.mkdir()
    with ZipFile(path, "w") as archive:
        archive.writestr("raw/edges.csv.gz", gzip.compress(b"0,1\n1,0\n1,2\n"))
    spec = EdgeListDatasetSpec(
        name="tiny_zip",
        url="https://example.test/tiny.zip",
        sha256=sha256(path.read_bytes()).hexdigest(),
        num_nodes=3,
        raw_num_edges=3,
        num_non_loop_edges=3,
        num_reciprocal_pairs=1,
        num_unordered_pairs=2,
        delimiter=",",
        archive_member="raw/edges.csv.gz",
    )

    graph = load_edge_list_dataset(spec, root=tmp_path, download=False)
    assert graph.num_edges == 3
