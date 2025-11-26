import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import crawl_results


def main():
    parser = argparse.ArgumentParser(
        "Usage: python plot_multiple_precision_vs_cache_hit_ratio.py --base_dir <base_dir>"
    )
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

        # Get base rate from first valid run to compute theoretical curves
        base_rate = None
        for model_name in model_data.keys():
            for run_path in model_data[model_name]:
                csv_path = os.path.join(run_path, "threshold_sweep_results.csv")
                if os.path.exists(csv_path):
                    try:
                        df_ref = pd.read_csv(csv_path)
                        row = df_ref.iloc[0]
                        total_pos = row['tp'] + row['fn']
                        total_neg = row['fp'] + row['tn']
                        total = total_pos + total_neg
                        base_rate = total_pos / total
                        break
                    except Exception:
                        pass
            if base_rate is not None:
                break

        # Plot theoretical curves
        if base_rate is not None:
            # Theoretical Perfect (Uniform Negatives)
            x_uniform = np.linspace(base_rate, 1.0, 100)
            y_uniform = base_rate / x_uniform
            x_uniform = np.concatenate(([0], x_uniform))
            y_uniform = np.concatenate(([1], y_uniform))
            auc_uniform = base_rate * (1 - np.log(base_rate))
            ax.plot(x_uniform, y_uniform, '--', color='black', label=f"Perfect (Uniform Negs), AUC: {auc_uniform:.3f}")

            # Theoretical Perfect (Zero Negatives)
            x_zeros = [0, base_rate, 1.0]
            y_zeros = [1.0, 1.0, base_rate]
            auc_zeros = base_rate + 0.5 * (1 - base_rate**2)
            ax.plot(x_zeros, y_zeros, ':', color='black', label=f"Perfect (Zero Negs), AUC: {auc_zeros:.3f}")

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
