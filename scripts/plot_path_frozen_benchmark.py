"""Plot path-frozen benchmark stress-over-time curves."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NAME = "branching"  # "branching" | "swiss_roll"
BENCHMARK_ROOT = SCRIPT_DIR / "res" / "path_frozen_benchmarks"
DATASET_BENCH_DIRS = {
    "branching": "branching_benchmarks",
    "swiss_roll": "swiss_roll_benchmarks",
}
DATASET_TITLES = {
    "branching": "Branching",
    "swiss_roll": "Swiss roll",
}

BENCH_DIR = BENCHMARK_ROOT / DATASET_BENCH_DIRS[DATASET_NAME]
CSV_DIR = BENCH_DIR / "csv"
FIG_DIR = BENCH_DIR / "figures"

N_SAMPLES = 1000
SEEDS = [42, 43, 44]
INNER_ITERS = [5, 20, 50]
LANDMARK_CODES = ["land10p", "land20p"]
FILENAME_RE = re.compile(r"n(?P<n>\d+)_(?P<code>.+)_ii(?P<inner>\d+)_oi(?P<outer>\d+)_s(?P<seed>\d+)\.csv")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for inner_iter in INNER_ITERS:
        plot_schedule(inner_iter)
        for landmark_code in LANDMARK_CODES:
            plot_schedule(inner_iter, landmark_code)
    print(f"Saved benchmark plots in {FIG_DIR}")


def plot_schedule(inner_iter, landmark_code=None):
    configs = benchmark_configs(inner_iter, landmark_code)
    if not configs:
        suffix = f", {landmark_code}" if landmark_code is not None else ""
        print(f"No benchmark CSV found for inner={inner_iter}{suffix}; skipping.")
        return
    fig, ax = plt.subplots(figsize=(11.4, 6.4) if landmark_code is None else (10.2, 6.0))
    for code, paths in configs:
        data = averaged_history(paths)
        ax.plot(
            data["elapsed"],
            data["full_geodesic_stress"],
            marker="o",
            linewidth=1.45,
            markersize=3.2,
            label=label_from_code(code),
        )

    title = f"{DATASET_TITLES[DATASET_NAME]} n={N_SAMPLES}: inner={inner_iter}, mean over seeds {seed_label()}"
    if landmark_code is not None:
        title += f", {landmark_label(landmark_code)} landmarks"
    ax.set_title(title)
    ax.set_xlabel("Optimization time excluding evaluation logs (s)")
    ax.set_ylabel("Full geodesic stress")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    if landmark_code is None:
        save_path = FIG_DIR / f"stress_time_n{N_SAMPLES}_i{inner_iter}.pdf"
    else:
        save_path = FIG_DIR / f"stress_time_i{inner_iter}_{landmark_code}.pdf"
    fig.savefig(save_path)
    plt.close(fig)


def benchmark_configs(inner_iter, landmark_code=None):
    groups = {}
    for path in CSV_DIR.glob(f"n{N_SAMPLES}_*_ii{inner_iter}_oi*_s*.csv"):
        match = FILENAME_RE.fullmatch(path.name)
        if not match:
            continue
        code = match.group("code")
        seed = int(match.group("seed"))
        if seed not in SEEDS:
            continue
        if landmark_code is not None and not code.startswith(landmark_code):
            continue
        groups.setdefault((code, int(match.group("outer"))), []).append(path)

    rows = []
    for (code, _outer), paths in groups.items():
        paths = sorted(paths)
        if len(paths) != len(SEEDS):
            found = sorted(int(FILENAME_RE.fullmatch(path.name).group("seed")) for path in paths)
            raise RuntimeError(f"{code}, inner={inner_iter} has seeds {found}, expected {SEEDS}.")
        rows.append((code, paths))
    return sorted(rows, key=lambda item: code_sort_key(item[0]))


def averaged_history(paths):
    data = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    return (
        data.groupby("outer_iter", as_index=False)[["elapsed", "full_geodesic_stress"]]
        .mean()
        .sort_values("outer_iter")
    )


def code_sort_key(code):
    if code == "AP":
        return (0, "", "", "")
    landmark_match = re.search(r"land(\d+)p", code)
    target_match = re.search(r"targ(\d+)p", code)
    landmark = int(landmark_match.group(1)) if landmark_match else 10_000
    target = -1 if "targ_all" in code else int(target_match.group(1)) if target_match else 10_000
    local = 1 if "locgeo" in code else 2 if "locdir" in code else 0
    return (1, landmark, target, local)


def label_from_code(code):
    if code == "AP":
        return "AP"
    label = code.replace("land", "land ")
    label = label.replace("targ_all", "targ all")
    label = re.sub(r"targ(\d+)p", r"targ \1%", label)
    label = re.sub(r"land (\d+)p", r"land \1%", label)
    label = label.replace("locgeocount", "local geo count")
    label = label.replace("locdircount", "local direct count")
    label = label.replace("locgeo", "local geo")
    label = label.replace("locdir", "local direct")
    return label.replace("_", ", ")


def landmark_label(code):
    match = re.fullmatch(r"land(\d+)p", code)
    return f"{match.group(1)}%" if match else code


def seed_label():
    return ",".join(str(seed) for seed in SEEDS)


if __name__ == "__main__":
    main()
