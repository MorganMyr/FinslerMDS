"""Evaluate trainable Finsler embeddings on a Table 2 dataset.

Examples
--------
Tune and evaluate the Randers model on both tasks::

    python scripts/main_link_prediction_finsler.py --dataset chameleon

Run a short fixed-parameter smoke test::

    python scripts/main_link_prediction_finsler.py --dataset squirrel \
        --task direction \
        --num-splits 1 --max-epochs 20 --patience 5 \
        --alpha 0.4 --radius 2 --temperature 1 --learning-rate 0.01
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finsler_mds.link_prediction.datasets import (  # noqa: E402
    DATASET_NAMES,
    load_directed_dataset,
)
from finsler_mds.link_prediction.experiments import (  # noqa: E402
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_DIMENSIONS,
    HYPERPARAMETER_SELECTION_PROTOCOL,
    METRIC_NAMES,
    ModelHyperparameters,
    OptunaConfig,
    OptunaSearchSpace,
    evaluate_hyperparameters,
    rerank_hyperparameters,
    save_experiment_summary,
    tune_hyperparameters,
)
from finsler_mds.link_prediction.evaluation import (  # noqa: E402
    SCORING_PROTOCOL,
    score_name,
)
from finsler_mds.link_prediction.initialization import (  # noqa: E402
    DEFAULT_INITIALIZATION,
    INITIALIZATION_NAMES,
)
from finsler_mds.link_prediction.runs import (  # noqa: E402
    create_run_directory,
    save_json,
)
from finsler_mds.link_prediction.split_cache import load_or_create_splits  # noqa: E402
from finsler_mds.link_prediction.splits import (  # noqa: E402
    LinkTask,
    SPLIT_PROTOCOL,
    split_protocol_metadata,
)
from finsler_mds.link_prediction.training import (  # noqa: E402
    EMBEDDING_TRAINING_PROTOCOL,
    TrainingConfig,
    resolve_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="chameleon")
    parser.add_argument(
        "--task", choices=("existence", "direction", "both"), default="both"
    )
    parser.add_argument("--metric", choices=METRIC_NAMES, default="randers")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets/link_prediction",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "scripts/res/link_prediction",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--first-split-seed", type=int, default=0)
    parser.add_argument("--evaluation-reverse-negative-fraction", type=float)
    parser.add_argument(
        "--dimension",
        dest="dimensions",
        type=int,
        nargs="+",
        help="Dimensions explored by Optuna (default: 5 10 20 50).",
    )
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--optuna-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--rerank-top", type=int, default=8)
    parser.add_argument("--rerank-splits", type=int, default=3)

    parser.add_argument("--alpha-max", type=float)
    parser.add_argument("--max-epochs", type=int, default=1_500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--evaluation-frequency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means full-batch.")
    parser.add_argument("--evaluation-batch-size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_NAMES,
        default=DEFAULT_INITIALIZATION,
        help="Node-coordinate initialization (default: current).",
    )

    fixed = parser.add_argument_group(
        "fixed execution (alpha, radius, temperature, and learning rate skip Optuna)"
    )
    fixed.add_argument("--alpha", type=float)
    fixed.add_argument("--radius", type=float)
    fixed.add_argument("--temperature", type=float)
    fixed.add_argument("--learning-rate", type=float)
    fixed.add_argument("--positive-weight", type=float)
    fixed.add_argument("--reverse-negative-fraction", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    requested_dimensions = (
        tuple(args.dimensions) if args.dimensions else DEFAULT_EMBEDDING_DIMENSIONS
    )
    if any(dimension <= 0 for dimension in requested_dimensions):
        raise ValueError("--dimension values must be positive.")
    if len(set(requested_dimensions)) != len(requested_dimensions):
        raise ValueError("--dimension values must not contain duplicates.")
    evaluation_mix = args.evaluation_reverse_negative_fraction
    if evaluation_mix is not None and not 0 <= evaluation_mix <= 1:
        raise ValueError("--evaluation-reverse-negative-fraction must be in [0, 1].")
    graph = load_directed_dataset(
        args.dataset,
        root=args.data_root,
        download=not args.no_download,
        remove_self_loops=True,
    )
    stats = graph.statistics()
    print(f"{graph.name} graph:", stats.as_dict())

    training_config = TrainingConfig(
        learning_rate=1e-2,
        max_epochs=args.max_epochs,
        patience=args.patience,
        evaluation_frequency=args.evaluation_frequency,
        batch_size=None if args.batch_size == 0 else args.batch_size,
        evaluation_batch_size=(
            None if args.evaluation_batch_size == 0 else args.evaluation_batch_size
        ),
        device=args.device,
        seed=args.training_seed,
        initialization=args.initialization,
    )
    fixed_values = (args.alpha, args.radius, args.temperature, args.learning_rate)
    if any(value is not None for value in fixed_values) and not all(
        value is not None for value in fixed_values
    ):
        raise ValueError(
            "Set all of --alpha, --radius, --temperature, and --learning-rate, "
            "or leave all unset to use Optuna."
        )
    fixed_hyperparameters = None
    search_space = None
    if all(value is not None for value in fixed_values):
        if args.alpha_max is not None:
            raise ValueError(
                "--alpha-max cannot be combined with fixed hyperparameters."
            )
        if len(requested_dimensions) != 1:
            if args.dimensions:
                raise ValueError("Fixed hyperparameters require exactly one dimension.")
            requested_dimensions = (DEFAULT_EMBEDDING_DIMENSION,)
        fixed_hyperparameters = ModelHyperparameters(
            dimension=requested_dimensions[0],
            alpha=args.alpha,
            radius=args.radius,
            temperature=args.temperature,
            learning_rate=args.learning_rate,
            positive_weight=(
                1.0 if args.positive_weight is None else args.positive_weight
            ),
            reverse_negative_fraction=(
                0.5
                if args.reverse_negative_fraction is None
                else args.reverse_negative_fraction
            ),
        )
    else:
        if (
            args.positive_weight is not None
            or args.reverse_negative_fraction is not None
        ):
            raise ValueError(
                "--positive-weight and --reverse-negative-fraction are only "
                "used with fixed hyperparameters."
            )
        if args.trials <= 0:
            raise ValueError("--trials must be positive.")
        if args.timeout is not None and args.timeout <= 0:
            raise ValueError("--timeout must be positive when provided.")
        if args.rerank_top <= 0:
            raise ValueError("--rerank-top must be positive.")
        if args.rerank_splits < 0 or args.rerank_splits >= args.num_splits:
            raise ValueError(
                "--rerank-splits must be between 0 and --num-splits - 1."
            )
        search_space = (
            OptunaSearchSpace(dimensions=requested_dimensions)
            if args.alpha_max is None
            else OptunaSearchSpace(
                dimensions=requested_dimensions, alpha_max=args.alpha_max
            )
        )

    tasks = (
        (LinkTask.EXISTENCE, LinkTask.DIRECTION)
        if args.task == "both"
        else (LinkTask(args.task),)
    )
    dataset_output = args.output_root / graph.name
    run_directory = create_run_directory(
        dataset_output,
        dataset=graph.name,
        metric=args.metric,
        dimensions=requested_dimensions,
        alpha_max=None if search_space is None else search_space.alpha_max,
        num_trials=None if search_space is None else args.trials,
        protocol="_".join(
            (
                SPLIT_PROTOCOL,
                EMBEDDING_TRAINING_PROTOCOL,
                SCORING_PROTOCOL,
                f"init{args.initialization}",
                (
                    f"rerank{args.rerank_top}x{args.rerank_splits}"
                    if search_space is not None
                    else "fixed"
                ),
            )
        ),
    )
    optimization = (
        {"method": "fixed", "hyperparameters": asdict(fixed_hyperparameters)}
        if fixed_hyperparameters is not None
        else {
            "method": "optuna",
            "num_trials": args.trials,
            "seed": args.optuna_seed,
            "timeout": args.timeout,
            "search_space": asdict(search_space),
            "selection": {
                "protocol": HYPERPARAMETER_SELECTION_PROTOCOL,
                "tuning_split_index": 0,
                "top_candidates": args.rerank_top,
                "reranking_split_indices": list(
                    range(1, args.rerank_splits + 1)
                ),
                "selection_score": (
                    "mean_validation_roc_auc_excluding_tuning_split"
                    if args.rerank_splits
                    else "tuning_validation_roc_auc"
                ),
            },
        }
    )
    save_json(
        run_directory / "config.json",
        {
            "format": "finsler_link_prediction_run_v5",
            "run_id": run_directory.name,
            "dataset": graph.name,
            "graph_fingerprint": graph.fingerprint,
            "graph_statistics": stats.as_dict(),
            "tasks": [task.value for task in tasks],
            "metric": args.metric,
            "dimensions": list(requested_dimensions),
            "embedding_training_protocol": EMBEDDING_TRAINING_PROTOCOL,
            "evaluation_scores": {task.value: score_name(task) for task in tasks},
            "splits": {
                **split_protocol_metadata(),
                "evaluation_reverse_negative_fraction": evaluation_mix,
                "count": args.num_splits,
                "first_seed": args.first_split_seed,
            },
            "training": asdict(training_config),
            "resolved_device": str(resolve_device(args.device)),
            "optimization": optimization,
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
            f"{task.value}: {len(splits)} cached splits; "
            f"train/val/test examples in split 0 = "
            f"{len(splits[0].train.labels)}/"
            f"{len(splits[0].validation.labels)}/"
            f"{len(splits[0].test.labels)}"
        )

        if fixed_hyperparameters is not None:
            hyperparameters = fixed_hyperparameters
        else:
            database = (run_directory / f"{task.value}_optuna.sqlite3").resolve()
            study = tune_hyperparameters(
                graph,
                splits,
                metric_name=args.metric,
                training_config=training_config,
                search_space=search_space,
                optuna_config=OptunaConfig(
                    num_trials=args.trials,
                    seed=args.optuna_seed,
                    storage=f"sqlite:///{database}",
                    timeout=args.timeout,
                ),
            )
            print(
                f"{task.value}: best split-0 validation "
                f"AUC={study.best_value:.6f}"
            )
            if args.rerank_splits:
                selection = rerank_hyperparameters(
                    graph,
                    splits,
                    study,
                    metric_name=args.metric,
                    top_candidates=args.rerank_top,
                    num_reranking_splits=args.rerank_splits,
                    training_config=training_config,
                    progress=lambda index, total, candidate: print(
                        f"{task.value}: reranking {index}/{total}, "
                        f"trial {candidate.trial_number}, mean validation "
                        f"AUC={candidate.mean_reranking_validation_roc_auc:.6f}"
                    ),
                )
                hyperparameters = selection.hyperparameters
                selection_path = run_directory / f"{task.value}_selection.json"
                save_json(selection_path, selection.as_dict())
                print(
                    f"{task.value}: selected trial "
                    f"{selection.selected_trial_number} after reranking"
                )
                print(f"Saved {selection_path}")
            else:
                hyperparameters = ModelHyperparameters(**study.best_params)
            print(f"{task.value}: best hyperparameters={hyperparameters}")

        summary = evaluate_hyperparameters(
            graph,
            splits,
            hyperparameters,
            metric_name=args.metric,
            training_config=training_config,
        )
        result_path = run_directory / f"{task.value}_summary.json"
        save_experiment_summary(result_path, summary)
        print(
            f"{task.value}: test ROC-AUC = "
            f"{100 * summary.mean_test_roc_auc:.2f} ± "
            f"{100 * summary.std_test_roc_auc:.2f}"
        )
        print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
