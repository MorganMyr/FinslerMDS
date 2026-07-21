"""Evaluate MagNet on exactly the same directed-link tasks as Finsler models.

Examples
--------
Tune MagNet on Chameleon and evaluate both tasks::

    python scripts/main_link_prediction_magnet.py --dataset chameleon

Run a quick fixed-parameter smoke test::

    python scripts/main_link_prediction_magnet.py --task direction --fixed \
        --num-splits 1 --max-epochs 20 --patience 5
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.link_prediction.baselines import (  # noqa: E402
    BaselineTrainingConfig,
    MagNetBaseline,
    MagNetHyperparameters,
    evaluate_baseline,
    save_baseline_summary,
    tune_baseline,
)
from finsler_mds.link_prediction.datasets import (  # noqa: E402
    DATASET_NAMES,
    load_directed_dataset,
)
from finsler_mds.link_prediction.runs import (  # noqa: E402
    create_tagged_run_directory,
    save_json,
)
from finsler_mds.link_prediction.split_cache import load_or_create_splits  # noqa: E402
from finsler_mds.link_prediction.splits import (  # noqa: E402
    LinkTask,
    SPLIT_PROTOCOL,
    split_protocol_metadata,
)
from finsler_mds.link_prediction.training import resolve_device  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="chameleon")
    parser.add_argument("--task", choices=("existence", "direction", "both"), default="both")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets/link_prediction")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "scripts/res/link_prediction",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--first-split-seed", type=int, default=0)
    parser.add_argument("--evaluation-reverse-negative-fraction", type=float)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--optuna-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float)
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Skip Optuna and use the parameters below.",
    )

    parser.add_argument("--max-epochs", type=int, default=3_000)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--evaluation-frequency", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-seed", type=int, default=0)

    fixed = parser.add_argument_group("MagNet parameters used with --fixed")
    fixed.add_argument("--q", type=float, default=0.25)
    fixed.add_argument("--hidden-channels", type=int, default=16)
    fixed.add_argument("--dropout", type=float, default=0.5)
    fixed.add_argument("--learning-rate", type=float, default=1e-3)
    fixed.add_argument("--weight-decay", type=float, default=5e-4)
    fixed.add_argument("--chebyshev-order", type=int, default=1)
    fixed.add_argument("--num-layers", type=int, default=2)
    fixed.add_argument("--no-activation", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_splits <= 0:
        raise ValueError("--num-splits must be positive.")
    evaluation_mix = args.evaluation_reverse_negative_fraction
    if evaluation_mix is not None and not 0 <= evaluation_mix <= 1:
        raise ValueError("--evaluation-reverse-negative-fraction must be in [0, 1].")
    if not args.fixed and args.trials <= 0:
        raise ValueError("--trials must be positive.")

    graph = load_directed_dataset(
        args.dataset,
        root=args.data_root,
        download=not args.no_download,
        remove_self_loops=True,
    )
    stats = graph.statistics()
    print(f"{graph.name} graph:", stats.as_dict())
    baseline = MagNetBaseline()
    training_config = BaselineTrainingConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        evaluation_frequency=args.evaluation_frequency,
        device=args.device,
        seed=args.training_seed,
    )
    fixed_hyperparameters = asdict(
        MagNetHyperparameters(
            q=args.q,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            chebyshev_order=args.chebyshev_order,
            num_layers=args.num_layers,
            activation=not args.no_activation,
        )
    )
    dataset_output = args.output_root / graph.name
    run_directory = create_tagged_run_directory(
        dataset_output / "baselines" / baseline.name,
        graph.name,
        baseline.name,
        "fixed" if args.fixed else f"n{args.trials}",
        SPLIT_PROTOCOL,
    )
    tasks = (
        (LinkTask.EXISTENCE, LinkTask.DIRECTION)
        if args.task == "both"
        else (LinkTask(args.task),)
    )
    save_json(
        run_directory / "config.json",
        {
            "format": "link_prediction_baseline_run_v1",
            "run_id": run_directory.name,
            "baseline": baseline.name,
            "dataset": graph.name,
            "graph_fingerprint": graph.fingerprint,
            "graph_statistics": stats.as_dict(),
            "tasks": [task.value for task in tasks],
            "splits": {
                **split_protocol_metadata(),
                "evaluation_reverse_negative_fraction": evaluation_mix,
                "count": args.num_splits,
                "first_seed": args.first_split_seed,
            },
            "training": asdict(training_config),
            "resolved_device": str(resolve_device(args.device)),
            "packages": {
                "torch-geometric": version("torch-geometric"),
                "torch-geometric-signed-directed": version(
                    "torch-geometric-signed-directed"
                ),
            },
            "optimization": (
                {"method": "fixed", "hyperparameters": fixed_hyperparameters}
                if args.fixed
                else {
                    "method": "optuna",
                    "num_trials": args.trials,
                    "seed": args.optuna_seed,
                    "timeout": args.timeout,
                    "search_space": baseline.search_space,
                }
            ),
        },
    )
    print(f"Run directory: {run_directory}")

    for task in tasks:
        task_evaluation_mix = evaluation_mix if task is LinkTask.EXISTENCE else None
        mix_suffix = "" if task_evaluation_mix is None else f"_rev{task_evaluation_mix}"
        split_cache = dataset_output / "split_cache" / (
            f"{SPLIT_PROTOCOL}_{task.value}_n{args.num_splits}_"
            f"seed{args.first_split_seed}{mix_suffix}.npz"
        )
        splits = load_or_create_splits(
            split_cache,
            graph,
            task,
            num_splits=args.num_splits,
            first_seed=args.first_split_seed,
            evaluation_reverse_negative_fraction=task_evaluation_mix,
        )
        print(
            f"{task.value}: {len(splits)} splits; train/val/test examples = "
            f"{len(splits[0].train.labels)}/"
            f"{len(splits[0].validation.labels)}/"
            f"{len(splits[0].test.labels)}; observed arcs = "
            f"{splits[0].observed_edge_index.shape[1]}"
        )
        if args.fixed:
            hyperparameters = fixed_hyperparameters
        else:
            database = (run_directory / f"{task.value}_optuna.sqlite3").resolve()
            hyperparameters, study = tune_baseline(
                baseline,
                graph,
                splits,
                training_config,
                num_trials=args.trials,
                optuna_seed=args.optuna_seed,
                storage=f"sqlite:///{database}",
                timeout=args.timeout,
            )
            print(f"{task.value}: best validation AUC={study.best_value:.6f}")
            print(f"{task.value}: best hyperparameters={hyperparameters}")

        summary = evaluate_baseline(
            baseline,
            graph,
            splits,
            hyperparameters,
            training_config,
        )
        result_path = run_directory / f"{task.value}_summary.json"
        save_baseline_summary(result_path, summary)
        print(
            f"{task.value}: test ROC-AUC = "
            f"{100 * summary.mean_test_roc_auc:.2f} ± "
            f"{100 * summary.std_test_roc_auc:.2f}"
        )
        print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
