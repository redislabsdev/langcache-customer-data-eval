import argparse
import os
import re

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
    parser = argparse.ArgumentParser("Usage: python plot_multiple_similarity_distribution.py --base_dir <base_dir>")
    parser.add_argument("--base_dir", type=str, required=False, default="complete_benchmark_results")
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
        n = len(model_data)
        cols = 3
        rows = (n + cols - 1) // cols

        if n == 1:
            rows = 1
            cols = 1
        elif n == 0:
            continue
        fig, axes = plt.subplots(rows, cols, figsize=(22, 6 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        if isinstance(axes, np.ndarray) and axes.ndim == 2:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        sorted_models = sorted(model_data.keys())
        for idx, model_name in enumerate(sorted_models):
            run_paths = model_data[model_name]
            try:
                ax = axes_flat[idx]

                pos_hists = []
                neg_hists = []
                overlaps = []

                edges = np.linspace(0, 1, 101)
                widths = np.diff(edges)

                valid_runs = 0

                for run_path in run_paths:
                    details_path = os.path.join(run_path, "llm_as_a_judge_results.csv")
                    if not os.path.exists(details_path):
                        continue

                    try:
                        df = pd.read_csv(details_path)
                        df = df.iloc[:-1]
                        pos = df[df["actual_label"] == 1]["similarity_score"].values
                        neg = df[df["actual_label"] == 0]["similarity_score"].values

                        if len(pos) == 0 or len(neg) == 0:
                            continue
                        pos_hist, _ = np.histogram(pos, bins=edges, density=True)
                        neg_hist, _ = np.histogram(neg, bins=edges, density=True)

                        pos_hists.append(pos_hist)
                        neg_hists.append(neg_hist)

                        overlap = np.sum(np.minimum(pos_hist, neg_hist) * widths)
                        overlaps.append(overlap)

                        valid_runs += 1
                    except Exception as e:
                        print(f"Error processing run {run_path}: {e}")

                if valid_runs == 0:
                    ax.text(0.5, 0.5, "Data not found", ha="center", va="center")
                    continue
                mean_pos_hist = np.mean(pos_hists, axis=0)
                mean_neg_hist = np.mean(neg_hists, axis=0)

                mean_overlap = np.mean(overlaps)
                std_overlap = np.std(overlaps) if valid_runs > 1 else 0.0

                centers = (edges[:-1] + edges[1:]) / 2
                ax.plot(centers, mean_pos_hist, label="Positive", color="green")
                ax.plot(centers, mean_neg_hist, label="Negative", color="red")

                ax.fill_between(centers, 0, mean_pos_hist, alpha=0.3, color="green")
                ax.fill_between(centers, 0, mean_neg_hist, alpha=0.3, color="red")
                ax.set_title(f"{model_name}\nOverlap = {mean_overlap:.3f} ± {std_overlap:.3f}")
                ax.set_xlabel("Similarity Score")
                ax.set_ylabel("Density")
                ax.legend()
                ax.grid(True, alpha=0.3)

            except Exception as e:
                print(f"Error plotting {model_name}: {e}")
        for idx in range(len(model_data), len(axes_flat)):
            axes_flat[idx].axis("off")
        plt.tight_layout()
        output_path = os.path.join(dataset_full_path, "similarity_distributions.png")
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()
