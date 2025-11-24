import argparse
import os
import re

import numpy as np
import pandas as pd


def calculate_pchr_auc(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df = df.iloc[:-1]

        if "cache_hit_ratio" not in df.columns or "precision" not in df.columns:
            return None

        x = df["cache_hit_ratio"].values
        y = df["precision"].values

        sorted_idx = np.argsort(x)
        x = x[sorted_idx]
        y = y[sorted_idx]

        auc = np.trapezoid(y, x)
        return auc
    except Exception as e:
        return None


def calculate_overlap(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df = df.iloc[:-1]

        if "actual_label" not in df.columns or "similarity_score" not in df.columns:
            return None
        pos = df[df["actual_label"] == 1]["similarity_score"].values
        neg = df[df["actual_label"] == 0]["similarity_score"].values
        if len(pos) == 0 or len(neg) == 0:
            return None
        lo = 0
        hi = 1
        edges = np.linspace(lo, hi, 101)
        pos_hist, _ = np.histogram(pos, bins=edges, density=True)
        neg_hist, _ = np.histogram(neg, bins=edges, density=True)
        widths = np.diff(edges)
        overlap = np.sum(np.minimum(pos_hist, neg_hist) * widths)
        return overlap
    except Exception as e:
        return None


def get_model_short_name(model_name):
    if "gte-modernbert" in model_name:
        return "ModernBERT"
    elif "v1" in model_name:
        return "v1"
    elif "v2" in model_name:
        return "v2"
    elif "v3.1" in model_name:
        return "v3.1"
    elif "v3" in model_name:
        return "v3"
    return model_name.split("/")[-1]


def get_model_name_from_dir(dir_name):
    # Try to reconstruct meaningful model name from directory name
    # e.g. neural_redis_langcache-embed-v1 -> redis/langcache-embed-v1
    # e.g. Alibaba-NLP_gte-modernbert-base -> Alibaba-NLP/gte-modernbert-base

    # If dir_name starts with neural_, strip it
    name = dir_name
    if name.startswith("neural_"):
        name = name[7:]

    # Reconstruct slash
    known_prefixes = ["redis", "Alibaba-NLP"]
    for prefix in known_prefixes:
        if name.startswith(prefix + "_"):
            return name.replace(prefix + "_", prefix + "/", 1)

    # If we have timestamps in the name (old format), strip them
    # Pattern: name_YYYY_MM_DD...
    match = re.search(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", name)
    if match:
        name = name[: match.start()]
        # Try prefix fix again
        for prefix in known_prefixes:
            if name.startswith(prefix + "_"):
                return name.replace(prefix + "_", prefix + "/", 1)

    return name


def crawl_results(base_dir):
    # Structure: base_dir / dataset / model / run_N / timestamp / csv
    # OR old structure: base_dir / dataset / model_timestamp / csv

    results = {}  # dataset -> model -> list of run paths

    if not os.path.exists(base_dir):
        return results

    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        results[dataset] = {}

        # List subdirs (models)
        for model_dir in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_dir)
            if not os.path.isdir(model_path):
                continue

            # Check if this is a model directory (contains run_X) or a timestamped dir (old format)
            if any(sub.startswith("run_") for sub in os.listdir(model_path)):
                # New format: model_dir is the model name (safe)
                model_name = get_model_name_from_dir(model_dir)
                if model_name not in results[dataset]:
                    results[dataset][model_name] = []

                for run_dir in os.listdir(model_path):
                    if not run_dir.startswith("run_"):
                        continue
                    run_path = os.path.join(model_path, run_dir)
                    # Inside run_dir, there should be a timestamped dir
                    timestamp_dirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
                    if timestamp_dirs:
                        # Take the latest one if multiple?
                        timestamp_dirs.sort()
                        final_path = os.path.join(run_path, timestamp_dirs[-1])
                        results[dataset][model_name].append(final_path)

            elif "threshold_sweep_results.csv" in os.listdir(model_path):
                # Old format: model_path IS the run path
                model_name = get_model_name_from_dir(model_dir)
                if model_name not in results[dataset]:
                    results[dataset][model_name] = []
                results[dataset][model_name].append(model_path)

    return results


def print_latex_table(
    title, metric_key, dataset_names, sorted_models, model_short_names, data_map, label, minimize=False
):
    print("\\begin{table*}[h]")
    print("\\centering")

    col_def = "l" + "c" * len(sorted_models)
    print(f"\\begin{{tabular}}{{{col_def}}}")
    print("\\toprule")

    header = "Dataset"
    for model in sorted_models:
        short_name = model_short_names[model]
        header += f" & {short_name}"
    print(f"{header} \\\\")
    print("\\midrule")

    for dataset in dataset_names:
        ds_name = dataset.replace("_test.csv", "").replace("_", "\\_")
        row_str = ds_name

        # Calculate best mean for highlighting
        means = []
        for model in sorted_models:
            stats = data_map[dataset].get(model, {"auc": {"mean": None}, "overlap": {"mean": None}})
            val = stats[metric_key]["mean"]
            if val is not None:
                means.append(val)

        best_mean = None
        if means:
            if minimize:
                best_mean = min(means)
            else:
                best_mean = max(means)

        for model in sorted_models:
            stats = data_map[dataset].get(
                model, {"auc": {"mean": None, "std": None}, "overlap": {"mean": None, "std": None}}
            )
            mean = stats[metric_key]["mean"]
            std = stats[metric_key]["std"]

            if mean is None:
                row_str += " & -"
            else:
                cell_str = f"{mean:.3f}"
                if std is not None:
                    cell_str += f" \\pm {std:.3f}"

                # Highlight best
                if best_mean is not None and abs(mean - best_mean) < 1e-6:
                    row_str += f" & \\textbf{{{cell_str}}}"
                else:
                    row_str += f" & {cell_str}"

        print(f"{row_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print(f"\\caption{{{title}}}")
    print(f"\\label{{{label}}}")
    print("\\end{table*}")
    print("\n")


def main():
    parser = argparse.ArgumentParser("Usage: python plot_similarity_distribution.py --base_dir <base_dir>")
    parser.add_argument("--base_dir", type=str, required=False, default="complete_benchmark_results")
    args = parser.parse_args()

    base_dir = args.base_dir

    # Crawl results instead of reading map
    benchmark_map = crawl_results(base_dir)

    if not benchmark_map:
        print("No results found.")
        return

    dataset_names = sorted(list(benchmark_map.keys()))

    # Collect all models
    all_models = set()
    for ds in benchmark_map:
        for model in benchmark_map[ds]:
            all_models.add(model)
    sorted_models = sorted(list(all_models))

    model_short_names = {m: get_model_short_name(m) for m in sorted_models}

    # Compute stats
    data_map = {}

    for dataset in dataset_names:
        data_map[dataset] = {}
        for model in sorted_models:
            run_paths = benchmark_map[dataset].get(model, [])

            aucs = []
            overlaps = []

            for run_path in run_paths:
                sweep_path = os.path.join(run_path, "threshold_sweep_results.csv")
                details_path = os.path.join(run_path, "llm_as_a_judge_results.csv")

                if os.path.exists(sweep_path):
                    val = calculate_pchr_auc(sweep_path)
                    if val is not None:
                        aucs.append(val)

                if os.path.exists(details_path):
                    val = calculate_overlap(details_path)
                    if val is not None:
                        overlaps.append(val)

            # Compute stats
            auc_stats = {"mean": None, "std": None}
            if aucs:
                auc_stats["mean"] = np.mean(aucs)
                auc_stats["std"] = np.std(aucs) if len(aucs) > 1 else 0.0

            overlap_stats = {"mean": None, "std": None}
            if overlaps:
                overlap_stats["mean"] = np.mean(overlaps)
                overlap_stats["std"] = np.std(overlaps) if len(overlaps) > 1 else 0.0

            data_map[dataset][model] = {"auc": auc_stats, "overlap": overlap_stats}
    # Print Tables
    print_latex_table(
        title="Precision vs Cache Hit Ratio (P-CHR) AUC (Mean $\\pm$ Std, $\\uparrow$)",
        metric_key="auc",
        dataset_names=dataset_names,
        sorted_models=sorted_models,
        model_short_names=model_short_names,
        data_map=data_map,
        label="tab:benchmark_auc",
        minimize=False,
    )
    print_latex_table(
        title="Similarity Distribution Overlap (Mean $\\pm$ Std, $\\downarrow$)",
        metric_key="overlap",
        dataset_names=dataset_names,
        sorted_models=sorted_models,
        model_short_names=model_short_names,
        data_map=data_map,
        label="tab:benchmark_overlap",
        minimize=True,
    )


if __name__ == "__main__":
    main()
