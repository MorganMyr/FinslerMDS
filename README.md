# Finsler-MDS: Geodesic Extensions and Asymmetric Data

This repository contains work carried out during a research internship on
manifold learning for asymmetric data. It extends
[Finsler Multi-Dimensional Scaling](https://arxiv.org/abs/2503.18010)
([original code](https://github.com/Tommoo/FinslerMDS)), which generalizes
classical MDS to non-symmetric dissimilarity matrices using a Randers metric in
the embedding space.

The repository focuses on three extensions:

- richer Finsler metrics, in particular the **Matsumoto metric**;
- **Finsler-GeoMDS**, which compares target dissimilarities with geodesic
  distances in the embedding rather than only direct distances;
- applications of these methods to synthetic data and two single-cell biology
  datasets, Paul15 and Pancreas.

This is research code. The biological experiments and several alternative
optimizers are exploratory: the repository can be used both to reproduce the
internship experiments and to run new ones.

## Implemented Methods

### Finsler-MDS

Direct Finsler-MDS minimizes a stress between a matrix of target
dissimilarities, which may be asymmetric, and the Finsler distances measured
along the segments connecting points in the embedding.

Two main optimizers are available:

- `smacof_randers` is the Finsler-SMACOF algorithm specialized for Randers
  metrics. The default implementation corrects the majorization and update
  steps of the published version;
- `gradient_descent` minimizes the stress directly and supports all implemented
  metrics.

### Finsler-GeoMDS

Finsler-GeoMDS replaces direct distances in the embedding space with shortest
paths on a kNN graph rebuilt from the embedding. The objective can therefore
preserve path geometry that direct MDS would collapse or distort.

The main optimizer is `path_frozen`. It alternates between:

1. building the current graph and computing its shortest paths;
2. taking a few optimization steps while keeping those paths fixed.

Landmarks, target subsampling, and direct local constraints reduce the cost on
datasets containing several thousand points. Path-Frozen nevertheless remains
slower and more sensitive to initialization than direct-distance methods.
These heuristics and their main parameters are described in
[Path-Frozen details](#path-frozen-details).

The repository also contains differentiable shortest-path approaches:
`datasp` (soft Floyd-Warshall), `soft_bellman_ford`, and
`relaxed_bellman_ford`. They mainly serve as experimental and comparison
methods; in our experiments, Path-Frozen offers the best compromise between
cost and quality.

### Finsler-UMAP

`finsler_umap` is an asymmetric variant of UMAP that also uses a Finsler metric
in the embedding space. It is based on
[Harnessing Data Asymmetry](https://arxiv.org/abs/2603.11396), but has been
modified to accept an arbitrary asymmetric dissimilarity matrix rather than
only asymmetric effects caused by non-uniform point density. As in the paper,
it is not exactly equivalent to UMAP even in the symmetric case.

### Metrics

The Finsler metrics used in the embedding space are defined in
`finsler_mds/metrics.py`. Their preferred direction is always the last
embedding axis.

| Metric | Main use |
| --- | --- |
| `RandersMetric` | simplest Finsler metric, used in the Finsler-MDS paper |
| `MatsumotoMetric` | nonlinear directional dependence, interpretable as travel time along a slope |
| `ConvexifiedMatsumotoMetric` | version that “corrects” the non-convexity of Matsumoto for ‖ω‖ > 1/2 |

The `finsler_mds/evaluation` subpackage provides direct and geodesic stress,
asymmetry-preservation measures, and the evaluation metrics used for RNA
velocity.

## Installation

From the repository root:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins the complete environment, including Scanpy, scVelo,
and CellRank. For synthetic experiments only, the essential dependencies are
NumPy, SciPy, scikit-learn, Matplotlib, Joblib, Numba, and `umap-learn`.

Some experiments require additional dependencies:

- GPU acceleration for some optimizers requires a CuPy version compatible with
  the local CUDA installation;
- `main_paul15_monocle.py` requires R, Monocle 3, and the R packages listed in
  `requirements.txt`.

The Paul15 and Pancreas datasets are loaded through Scanpy and CellRank,
respectively. They are not versioned in this repository. Their first download
and the RNA-velocity computation may take time and require network access.

## API Usage

The common entry point is `fit_finsler_mds`. For example, to optimize direct
Finsler-MDS with Matsumoto:

```python
import numpy as np

from finsler_mds import MatsumotoMetric, fit_finsler_mds

D = np.array(
    [
        [0.0, 1.0, 2.0],
        [1.3, 0.0, 1.0],
        [2.4, 1.2, 0.0],
    ]
)

embedding, stress = fit_finsler_mds(
    D,
    metric=MatsumotoMetric(alpha=0.4),
    optimizer="gradient_descent",
    n_components=2,
    random_state=42,
)
```

The main names accepted by `optimizer` are:

| Optimizer | Objective |
| --- | --- |
| `smacof_randers` | direct Finsler-MDS, Randers only |
| `gradient_descent` | direct Finsler-MDS, generic metric |
| `path_frozen` | Finsler-GeoMDS, recommended method |
| `datasp` | Finsler-GeoMDS with soft Floyd-Warshall |
| `soft_bellman_ford`, `relaxed_bellman_ford` | experimental geodesic variants |
| `finsler_umap` | directed fuzzy graph and UMAP-like objective |

For Path-Frozen, it is generally best to start from an already reasonable
embedding, for example one produced by UMAP, Isomap, or direct Finsler-MDS.
The scripts provide complete configurations for each family of experiments.

## Path-Frozen Details

### Operation and Heuristics

As outlined above, Path-Frozen performs outer iterations that rebuild the kNN
graph of the embedding points, compute shortest paths in that graph with
Dijkstra's algorithm, and finally take a few gradient-descent steps (inner
iterations) on a stress for which the optimal paths are assumed to remain
optimal—the paths are “frozen.”

The exhaustive version becomes too expensive for graphs with more than a few
thousand points: every outer iteration runs Dijkstra from all \(n\) points,
then performs gradient descent over the \(n(n-1)\) directed pairs.

Path-Frozen therefore uses several heuristics to make the computation
tractable:

- run Dijkstra only from a subset of points called **landmarks**;
- subsample the targets associated with each landmark to reduce the number of
  pairs included in gradient descent;
- add *local pairs*—neighbors according to the input dissimilarities—to the
  stress, using their direct rather than geodesic distances to avoid additional
  Dijkstra computations.

We also added two mechanisms to limit problems caused by over-optimizing the
frozen objective during gradient descent, notably collisions between branches:

- an optional direct-distance stress regularizer;
- damping of the displacement between two graph reconstructions.

### Important Parameters

| Parameter | Role and trade-off |
| --- | --- |
| `graph_neighbors` | Number of neighbors in the kNN graph built from the embedding. Too small a value can produce fragile paths or a disconnected graph; too large a value allows more shortcuts. |
| `outer_iter` | Number of graph reconstructions and shortest-path recomputations. |
| `inner_iter` | Maximum number of optimization iterations performed while paths remain fixed. A large value enables rapid changes but may spend too long optimizing paths that have become outdated. |
| `outer_step_size` | Fraction of the proposed displacement applied after an inner optimization. `1` applies the full displacement; a smaller value generally improves stability by damping graph changes, but slows progress. |
| `n_local_pairs` | Number of smallest dissimilarities retained for each source point so that local relationships remain represented. It can be set equal or close to `graph_neighbors`. |
| `local_pair_mode` | With `geodesic`, local pairs use graph paths; this is generally avoided and mainly retained for timing and effect comparisons. With the default `direct`, they use direct distances in the embedding space. |
| `n_landmark` | Number of points used as sources to represent global structure. A higher value gives a more complete but more expensive objective; 10% of the points is a useful starting point. |
| `random_landmark_fraction` | Proportion, between `0` and `1`, of randomly selected landmarks; the others are selected by farthest-point sampling in the target dissimilarities. `0` favors fixed, regular coverage, while `1` uses entirely random sampling. |
| `resample_random_landmarks` | When enabled, as it is by default, reselects the random landmark subset at each outer iteration. The farthest-point subset remains fixed. This is generally disabled only for comparative tests. |
| `targets_per_landmark` | Maximum number of targets sampled for each landmark; targets are randomly reselected at each outer iteration. Weights are corrected to estimate the complete landmark objective. If gradient descent is not the execution-time bottleneck, this can be set fairly high for greater accuracy—typically between 20% and 50% of the points. |
| `local_global_reweighting` | Controls the balancing of local and global groups: none with `none`, by total weight mass with the default `count`, or by target energy with `energy`. The latter multiplies weights by `D^2` before balancing and therefore strongly favors the local term. |
| `local_weight` | Multiplies the local-group weight after any rebalancing. A high value favors local structure. |
| `direct_stress_mode` | Formula used for direct-distance regularization. `hinge` only penalizes direct distances that have become too small; the default `mds` adds a full direct MDS stress. |
| `direct_stress_weight` | Weight of the direct regularizer over all pairs, unaffected by landmarks or target subsampling. `0` disables it. It is useful at the beginning of optimization but should be reduced to avoid biasing the result too strongly. |
| `direct_stress_margin` | In `hinge` mode, the threshold below which a direct distance is penalized, expressed as a fraction of the target dissimilarity. |

### Multi-Stage Optimization

It is generally preferable not to run Path-Frozen only once with fixed
parameters. First run a relatively aggressive exploratory stage to obtain a
good global structure quickly. Then run Path-Frozen again, using the first
embedding as the new `init`, with more conservative parameters to recover the
correct local structure.

A typical exploratory configuration uses, for example, `inner_iter=50`,
`outer_step_size=1`, `direct_stress_weight=0.3`,
`local_global_reweighting="count"`, and `local_weight=0.1`. The refinement
stage can then use `inner_iter=10`, `outer_step_size=0.2`, a
`direct_stress_weight` of `0.05`, `0.01`, or even `0`, and a `local_weight` of
`0.5` or `1`. The final stages may further reduce `inner_iter` and
`outer_step_size` if needed.

These values are starting points and depend on the dataset, initialization,
and other sampling heuristics. This sequence is currently implemented
explicitly in some scripts; automating it would be a useful optimizer
improvement.

## Available Experiments

Run scripts from the repository root with
`python scripts/<script_name>.py`. Most parameters are defined in constants or
dictionaries near the beginning of each file.

### Synthetic Cases

| Scripts | Experiment |
| --- | --- |
| `main_nested.py` | controlled GeoMDS case: two nested rectangles whose geodesic distances cannot be represented well by direct MDS |
| `main_converging_flow.py` | converging-current case whose asymmetry can be preserved better by Finsler-GeoMDS than by direct Finsler-MDS |
| `main_spiral_path_frozen.py` | synthetic case showing that the original kNN graph cannot simply be kept fixed in Path-Frozen |
| `main_mountains.py` | geodesics on a surface with three mountains |
| `main_sea.py`, `main_sea_paths.py` | the Sea1 current map from the Finsler-MDS paper, metric comparisons, and source-target path visualization. Displayed paths are shortest paths on k-NN graphs rather than exact continuous geodesics |
| `main_branching.py` | branching dataset used to test Path-Frozen |
| `benchmark_path_frozen.py` | stress-over-time measurements on Branching and Swiss roll, used to compare Path-Frozen parameters and heuristics |

Figures, and in some cases embeddings, are written to `scripts/res/`, which is
ignored by Git.

### Examples from the Finsler-MDS Paper

`main_swiss_roll_full.py` and `main_swiss_roll_hole.py`
retain the visualization experiments from the original repository, with some
adaptations for the current API. They cover the Swiss roll and robustness to a
hole.

The Link Prediction component is maintained in a
[separate repository](https://github.com/MorganMyr/FinslerLinkPrediction).

### Trajectory Inference on Paul15

Trajectory inference assigns each point a pseudotime representing its progress
through a biological transformation. The experiments investigate whether MDS,
GeoMDS, or their Finsler variants produce better visualizations when pseudotime
or density is used to create asymmetric dissimilarities. The current Paul15
dataset lacks ground truth, and no quantitative evaluation has been performed
so far.

- `main_paul15_finsler.py` is the main entry point. It builds geodesic
  dissimilarities in diffusion space, optionally makes them asymmetric using
  pseudotime or density, and then tests Finsler-MDS and Path-Frozen;
- `main_paul15_baseline.py` builds the Scanpy, PAGA, DPT, and UMAP baselines;
- `main_paul15_diffmap_embedding.py` reruns a targeted configuration from the
  diffusion caches;
- `main_paul15_monocle.py` and the R script in `monocle3_bridge/` compare
  Monocle 3 on UMAP and on a GeoMDS embedding;
- `main_paul15_phate.py` compares MDS and GeoMDS as visualization methods at
  the end of the PHATE pipeline;
- `main_paul15_pseudotime_lift.py` and
  `plot_paul15_paga_pseudotime_embeddings.py` produce complementary
  visualizations.

### RNA Velocity on Pancreas

The Pancreas pipeline starts from cells that each have a velocity vector
representing their direction of evolution, all projected into PCA space. These
vectors are interpreted as currents to compute asymmetric geodesic distances,
using either a Randers or Matsumoto metric, which serve as dissimilarities. One
of the implemented methods—Finsler-MDS, GeoMDS, Finsler-UMAP, and so on—can
then be applied.

- `main_pancreas.py` contains the complete pipeline: loading, preprocessing,
  RNA velocity, dissimilarities, initialization, optimization, and saving;
- `precompute_pancreas_velocity_distance_caches.py` quickly recomputes several
  dissimilarity matrices from an already cached biological state;
- `evaluate_pancreas_embedding.py` evaluates a saved embedding using CBDir,
  local and global velocity consistency, alignment preservation, and,
  optionally, stress;
- `plot_pancreas_velocity_embedding.py` overlays the projected velocity field
  on an embedding;
- `pancreas_gap_distance.py` evaluates gap distance, a metric sensitive to a
  missing region and used, for example, in the VeloViz paper. This script has
  seen little use but is retained for possible follow-up work.

`finsler_mds/utils/pancreas_campaign.py` stores reusable configurations and CSV
helpers for running Pancreas test campaigns rather than launching them
individually through `main_pancreas`.

## Repository Structure

```text
finsler_mds/
  api.py                 common entry point
  metrics.py             Finsler metrics
  optimizers/            Finsler-MDS, GeoMDS, and Finsler-UMAP
  evaluation/            stress, asymmetry, and RNA-velocity metrics
  utils/                 graphs, caches, initializations, and figures
scripts/                  reproducible experiments
docs/                     working notes and supporting material
```

Results, embeddings, and caches are normally created under `scripts/res/`.
They are not tracked by Git. Embedding filenames and metadata encode some of
the parameters used to create them.

## Limitations and Future Work

- The Path-Frozen implementation of Finsler-GeoMDS remains significantly
  slower than gradient descent for standard MDS. It could be improved, or a
  different GeoMDS implementation could be considered.
- Path-Frozen currently often needs to be run several times in succession,
  reducing `inner_iter` and `outer_step_size`, for example. This sequence could
  be automated.
- It would be useful to develop a synthetic dataset on which
  Finsler-MDS/GeoMDS/UMAP clearly produces a better embedding with Matsumoto
  than with Randers. On fixed 3D point clouds, Matsumoto can be shown to
  produce more natural geodesics, such as paths around mountains, but Randers
  can often produce a satisfactory embedding as well.
- We do not yet have quantitative results for trajectory inference, despite it
  being a promising GeoMDS test case. It would be useful to study a dataset
  other than Paul15 with known pseudotime ground truth.
- The RNA-velocity results on Pancreas are already reasonable, but testing
  other metrics such as Gap Distance or another dataset could produce results
  more favorable to Finsler-MDS, with Matsumoto for example, or GeoMDS.

## Main Reference

If this code is reused, please cite at least the Finsler-MDS paper whose
implementation it extends:

```bibtex
@inproceedings{dages2025finsler,
  title     = {Finsler Multi-Dimensional Scaling: Manifold Learning for
               Asymmetric Dimensionality Reduction and Embedding},
  author    = {Dag{\`e}s, Thomas and Weber, Simon and Lin, Ya-Wei Eileen and
               Talmon, Ronen and Cremers, Daniel and Lindenbaum, Michael and
               Bruckstein, Alfred M. and Kimmel, Ron},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition},
  pages     = {25842--25853},
  year      = {2025}
}
```

The code is distributed under the BSD 3-Clause license; see `LICENSE`.
