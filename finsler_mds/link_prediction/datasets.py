"""Dataset loaders for the Table 2 link-prediction benchmark."""

from __future__ import annotations

import gzip
import tarfile
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .data import DirectedGraphData


_GEOM_GCN_REVISION = "f1fc0d14b3b019c562737240d06ec83b07d16a8f"
_GEOM_GCN_ROOT = (
    "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
    f"{_GEOM_GCN_REVISION}/new_data"
)


@dataclass(frozen=True)
class EdgeListDatasetSpec:
    name: str
    url: str
    sha256: str
    num_nodes: int
    raw_num_edges: int
    num_non_loop_edges: int
    num_reciprocal_pairs: int
    num_unordered_pairs: int
    delimiter: str | None = None
    skip_header: int = 0
    archive_member: str | None = None
    remap_node_ids: bool = False


CHAMELEON = EdgeListDatasetSpec(
    name="chameleon",
    url=f"{_GEOM_GCN_ROOT}/chameleon/out1_graph_edges.txt",
    sha256="45ac593de9207470090230c8fd76beeb2f60b5371a9be088e8bf31e7f0619a4b",
    num_nodes=2_277,
    raw_num_edges=36_101,
    num_non_loop_edges=36_051,
    num_reciprocal_pairs=4_680,
    num_unordered_pairs=31_371,
    delimiter="\t",
    skip_header=1,
)

CORA = EdgeListDatasetSpec(
    name="cora",
    url="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    sha256="0d4ed463d1627bb7f3e8420effe8f5545fd492ae8f88dab44ce86cee7b26d7e8",
    num_nodes=2_708,
    raw_num_edges=5_429,
    num_non_loop_edges=5_429,
    num_reciprocal_pairs=151,
    num_unordered_pairs=5_278,
    archive_member="cora/cora.cites",
    remap_node_ids=True,
)

SQUIRREL = EdgeListDatasetSpec(
    name="squirrel",
    url=f"{_GEOM_GCN_ROOT}/squirrel/out1_graph_edges.txt",
    sha256="453d8e96754d68e2ab0a7ebb56f93506b59531d95abaec7a2b46fe1eae7c39cd",
    num_nodes=5_201,
    raw_num_edges=217_073,
    num_non_loop_edges=216_933,
    num_reciprocal_pairs=18_580,
    num_unordered_pairs=198_353,
    delimiter="\t",
    skip_header=1,
)

CITESEER = EdgeListDatasetSpec(
    name="citeseer",
    url="https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
    sha256="b02ee7b5d83130f8fd45b59017a76fdae3e998629a1904c2c5e07343a9664685",
    num_nodes=3_327,
    raw_num_edges=4_732,
    num_non_loop_edges=4_608,
    num_reciprocal_pairs=56,
    num_unordered_pairs=4_552,
    archive_member="citeseer/citeseer.cites",
    remap_node_ids=True,
)

ARXIV_YEAR = EdgeListDatasetSpec(
    name="arxiv_year",
    url="https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
    sha256="49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276",
    num_nodes=169_343,
    raw_num_edges=1_166_243,
    num_non_loop_edges=1_166_243,
    num_reciprocal_pairs=8_444,
    num_unordered_pairs=1_157_799,
    delimiter=",",
    archive_member="arxiv/raw/edge.csv.gz",
)

DATASET_SPECS = {
    spec.name: spec
    for spec in (ARXIV_YEAR, CHAMELEON, CITESEER, CORA, SQUIRREL)
}
DATASET_NAMES = tuple(sorted(name.replace("_", "-") for name in DATASET_SPECS))


def load_directed_dataset(
    name: str,
    *,
    root="datasets/link_prediction",
    download: bool = True,
    remove_self_loops: bool = True,
) -> DirectedGraphData:
    """Load a registered directed graph in its benchmark representation."""
    key = name.lower().replace("-", "_")
    try:
        spec = DATASET_SPECS[key]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown link-prediction dataset {name!r}. Available: {known}.") from exc
    return load_edge_list_dataset(
        spec,
        root=root,
        download=download,
        remove_self_loops=remove_self_loops,
    )


def load_edge_list_dataset(
    spec: EdgeListDatasetSpec,
    *,
    root="datasets/link_prediction",
    download: bool = True,
    remove_self_loops: bool = True,
) -> DirectedGraphData:
    """Download, verify, and load a two-column directed edge list."""
    dataset_dir = Path(root).expanduser().resolve() / spec.name
    raw_path = dataset_dir / Path(spec.url).name
    if not raw_path.exists():
        if not download:
            raise FileNotFoundError(
                f"Dataset file {raw_path} is missing and download=False."
            )
        _download_verified(spec.url, raw_path, spec.sha256)
    else:
        _verify_checksum(raw_path, spec.sha256)

    edges = _read_edges(raw_path, spec)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"Expected a two-column edge list in {raw_path}, got {edges.shape}.")
    if len(edges) != spec.raw_num_edges:
        raise ValueError(
            f"{spec.name} has {len(edges)} raw edges; expected {spec.raw_num_edges}."
        )

    # Canonicalize edge order and discard duplicate directed arcs.
    linear = edges[:, 0] * spec.num_nodes + edges[:, 1]
    linear = np.unique(linear)
    edge_index = np.vstack((linear // spec.num_nodes, linear % spec.num_nodes))
    graph = DirectedGraphData(
        name=spec.name,
        num_nodes=spec.num_nodes,
        edge_index=edge_index,
        source=spec.url,
        metadata={
            "raw_path": str(raw_path),
            "raw_sha256": spec.sha256,
            "archive_member": spec.archive_member,
            "loader": "sorted_coalesced_edge_list_v1",
        },
    )
    stats = graph.statistics()
    actual = (
        stats.num_non_loop_directed_edges,
        stats.num_reciprocal_pairs,
        stats.num_unordered_pairs,
    )
    expected = (
        spec.num_non_loop_edges,
        spec.num_reciprocal_pairs,
        spec.num_unordered_pairs,
    )
    if actual != expected:
        raise ValueError(f"{spec.name} processed statistics are {actual}; expected {expected}.")
    return graph.without_self_loops() if remove_self_loops else graph


def _read_edges(path: Path, spec: EdgeListDatasetSpec) -> np.ndarray:
    def read(source):
        return np.loadtxt(
            source,
            dtype=str if spec.remap_node_ids else np.int64,
            delimiter=spec.delimiter,
            skiprows=spec.skip_header,
            ndmin=2,
        )

    if spec.archive_member is None:
        edges = read(path)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive, archive.open(spec.archive_member) as member:
            if spec.archive_member.endswith(".gz"):
                with gzip.GzipFile(fileobj=member) as decompressed:
                    edges = read(decompressed)
            else:
                edges = read(member)
    else:
        with tarfile.open(path, "r:*") as archive:
            member = archive.extractfile(spec.archive_member)
            if member is None:
                raise ValueError(f"Missing {spec.archive_member!r} in {path}.")
            edges = read(member)

    if spec.remap_node_ids:
        node_ids, inverse = np.unique(edges, return_inverse=True)
        if len(node_ids) != spec.num_nodes:
            raise ValueError(
                f"{spec.name} has {len(node_ids)} node IDs; expected {spec.num_nodes}."
            )
        edges = inverse.reshape(edges.shape)
    return np.asarray(edges, dtype=np.int64)


def _download_verified(url: str, destination: Path, expected_sha256: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        _verify_checksum(temporary, expected_sha256)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_checksum(path: Path, expected_sha256: str):
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}."
        )


__all__ = [
    "ARXIV_YEAR",
    "CHAMELEON",
    "CITESEER",
    "CORA",
    "DATASET_NAMES",
    "DATASET_SPECS",
    "EdgeListDatasetSpec",
    "SQUIRREL",
    "load_directed_dataset",
    "load_edge_list_dataset",
]
