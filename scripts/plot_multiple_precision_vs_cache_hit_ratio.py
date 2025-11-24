import argparse
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_model_name_from_dir(dir_name):
    name = dir_name
    if name.startswith("neural_"):
        name = name[7:]
    known_prefixes = ["redis", "Alibaba-NLP"]
    for prefix in known_prefixes:
        if name.startswith(prefix + "_"):
            return name.replace(prefix + "_", prefix + "/", 1)
    match = re.search(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}", name)
    if match:
        name = name[: match.start()]
        for prefix in known_prefixes:
            if name.startswith(prefix + "_"):
                return name.replace(prefix + "_", prefix + "/", 1)
    return name


def crawl_results(base_dir):
    results = {}
    if not os.path.exists(base_dir):
        return results
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
        results[dataset] = {}
        for model_dir in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_dir)
            if not os.path.isdir(model_path):
                continue
            if any(sub.startswith("run_") for sub in os.listdir(model_path)):
                model_name = get_model_name_from_dir(model_dir)
                if model_name not in results[dataset]:
                    results[dataset][model_name] = []
                for run_dir in os.listdir(model_path):
                    if not run_dir.startswith("run_"):
                        continue
                    run_path = os.path.join(model_path, run_dir)
                    timestamp_dirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
                    if timestamp_dirs:
                        timestamp_dirs.sort()
                        results[dataset][model_name].append(os.path.join(run_path, timestamp_dirs[-1]))
            elif "threshold_sweep_results.csv" in os.listdir(model_path):
                model_name = get_model_name_from_dir(model_dir)
                if model_name not in results[dataset]:
                    results[dataset][model_name] = []
                results[dataset][model_name].append(model_path)
    return results


def main():
    parser = argparse.ArgumentParser("Usage: python plot_precision_vs_cache_hit_ratio.py --base_dir <base_dir>")
    parser.add_argument("--base_dir", type=str, default="complete_benchmark_results")
    args = parser.parse_args()
    base_dir = args.base_dir
    benchmark_map = crawl_results(base_dir)

    if not benchmark_map:
        print("No results found.")
        return

    for dataset_name, model_data in benchmark_map.items():
        print(f"Processing {dataset_name}...")

        dataset_full_path = os.path.join(base_dir, dataset_name)
        if not os.path.exists(dataset_full_path):
            continue
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        sorted_models = sorted(model_data.keys())
        for i, model_name in enumerate(sorted_models):
            run_paths = model_data[model_name]
            precisions_interp = []
            aucs_pchr = []

            common_chr = np.linspace(0, 1, 100)

            valid_runs = 0

            for run_path in run_paths:
                csv_path = os.path.join(run_path, "threshold_sweep_results.csv")
                if not os.path.exists(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    df = df.iloc[:-1]

                    x_chr = df["cache_hit_ratio"].values
                    y_prec = df["precision"].values

                    if len(x_chr) < 2:
                        continue
                    sorted_idx = np.argsort(x_chr)
                    x_chr = x_chr[sorted_idx]
                    y_prec = y_prec[sorted_idx]

                    p_interp = np.interp(common_chr, x_chr, y_prec)
                    precisions_interp.append(p_interp)
                    aucs_pchr.append(np.trapezoid(p_interp, common_chr))

                    valid_runs += 1
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

            if valid_runs == 0:
                continue

            mean_p_chr = np.mean(precisions_interp, axis=0)
            std_p_chr = np.std(precisions_interp, axis=0) if valid_runs > 1 else np.zeros_like(mean_p_chr)
            mean_auc_pchr = np.mean(aucs_pchr)
            std_auc_pchr = np.std(aucs_pchr) if valid_runs > 1 else 0.0

            color = colors[i % len(colors)]

            label_chr = f"{model_name}, AUC: {mean_auc_pchr:.3f} ± {std_auc_pchr:.3f}"
            ax.plot(common_chr, mean_p_chr, label=label_chr, color=color)
            if valid_runs > 1:
                ax.fill_between(common_chr, mean_p_chr - std_p_chr, mean_p_chr + std_p_chr, color=color, alpha=0.2)
        ax.set_xlabel("Cache Hit Ratio")
        ax.set_ylabel("Precision")
        ax.set_title("Precision vs Cache Hit Ratio")
        ax.grid(True)
        ax.legend()
        fig.suptitle(f"Performance on {dataset_name.split('_')[0]}")
        plt.tight_layout()
        output_path = os.path.join(dataset_full_path, "precision_vs_cache_hit_ratio.png")
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    main()
