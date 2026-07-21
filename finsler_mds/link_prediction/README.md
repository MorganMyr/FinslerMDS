# Directed link prediction

This package implements the non-MDS representation-learning experiment from
Table 2 of the Finsler-MDS paper. Every node owns a trainable coordinate; there
is no MLP or GNN. For an ordered pair `(i, j)`, the model uses

```text
d_ij = F(x_j - x_i)
logit_ij = (radius - d_ij**2) / temperature
```

and reconstructs every directed edge in the observed training graph. One
negative per observed arc is resampled every epoch from inverse arcs and random
non-edges. Observed arcs and both orientations of every validation/test query
are excluded. MagNet's noisy binary examples are used only for validation and
test (and to train supervised baselines). Optuna tunes the dimensions
`{5, 10, 20, 50}`, metric asymmetry, decoder radius and temperature, learning
rate, positive-class weight, and fraction of inverse negatives. After tuning on
each split's validation set independently, the selected configuration is
retrained on that split and evaluated once on its untouched test set. The final
score is the mean and standard deviation of the ten test AUCs.

Existence AUC ranks pairs by `logit_ij` (equivalently `p_ij`). Direction AUC
ranks them by `logit_ij - logit_ji`, the log-odds ratio between the two
orientations. This score is used for Optuna, early stopping, and final testing;
the edge-reconstruction cross-entropy remains unchanged.

## Modules

- `data.py`: validated directed-graph containers and graph fingerprints.
- `datasets.py`: pinned, checksummed raw dataset loaders.
- `splits.py`: MagNet noisy task construction; `split_cache.py`: versioned
  persistence.
- `torch_metrics.py`: differentiable Randers and Matsumoto kernels tied to the
  metric objects in `finsler_mds.metrics`.
- `decoder.py` and `model.py`: Fermi-Dirac decoder and direct node embeddings.
- `training.py` and `evaluation.py`: dynamic negative sampling, early stopping,
  and ROC-AUC.
- `optimization.py` and `experiments.py`: independent Optuna selection and
  testing on every split.
- `baselines/`: common external-method runner and thin method adapters. MagNet
  imports the public PyG implementation but always uses this package's splits.

## Experiments

Install the optional dependencies and run:

```bash
pip install -r requirements-link-prediction.txt
python scripts/main_link_prediction_finsler.py --dataset chameleon --metric randers
python scripts/main_link_prediction_finsler.py --dataset citeseer --metric randers
python scripts/main_link_prediction_finsler.py --dataset cora --metric randers
python scripts/main_link_prediction_finsler.py --dataset squirrel --metric randers
python scripts/main_link_prediction_finsler.py --dataset arxiv-year --metric randers
python scripts/main_link_prediction_magnet.py --dataset chameleon
python scripts/main_link_prediction_magnet.py --dataset citeseer
python scripts/main_link_prediction_magnet.py --dataset squirrel
```

Each invocation creates a timestamped run directory. It contains `config.json`,
one summary per requested task, and, when tuning is enabled, one Optuna database
per task containing a separate study for every split. Split caches remain
outside the run directories so every method uses the same examples. Use
`--dimension 50` to restrict Optuna to one dimension (or pass several values)
and `--alpha-max` for asymmetry ablations. `--trials` and `--timeout` apply to
each split. Previous runs are never resumed or overwritten implicitly.

The pinned loaders retain 4,552 underlying pairs for CiteSeer, 5,278 for Cora,
31,371 for Chameleon, 198,353 for Squirrel, and 1,157,799 for Arxiv-Year.
Following MagNet's public StellarGraph
pipeline, 15% of those pairs are held out for test, then 5% of the remainder
for validation, with connectivity preserved. Baseline training examples are
sampled from 99% of the observed pairs. Each underlying edge is alternately presented
in its real or reversed orientation. Reciprocal pairs are retained, but their
direction target is drawn uniformly with the split's seeded generator so their
ambiguous labels contain no node-ordering signal. Existence additionally
samples one graph non-edge per positive candidate, then applies the public
code's class balancing.

The observed graph retains both arcs of reciprocal pairs. Evaluation non-edges
come from the complete-graph complement and are disjoint across partitions.
Embedding training receives no complete graph: it only uses observed arcs and
the identities—not the labels—of held-out queries to exclude both orientations
from negative sampling.

The shared Optuna search space defaults to `alpha_max=0.999`. For plain
Matsumoto, pass `--alpha-max 0.49` to remain in its convex regime; use
`--metric convexified_matsumoto` when exploring larger asymmetry values.
