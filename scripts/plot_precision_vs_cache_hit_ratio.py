import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        "Usage: python plot_precision_vs_cache_hit_ratio.py --output_dir <output_dir> --title <title>"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--title", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = {
        "v1": "rado_synthetic/v1/threshold_sweep_results.csv",
        "v2": "rado_synthetic/v2/threshold_sweep_results.csv",
        "v3": "rado_synthetic/v3/threshold_sweep_results.csv",
        "v3.1": "rado_synthetic/v3.1/threshold_sweep_results.csv",
    }

    fig, ax = plt.subplots(figsize=(16, 7), ncols=2)

    # Get constants from the first dataframe (assuming all runs use same dataset)
    first_csv = list(csv_paths.values())[0]
    if os.path.exists(first_csv):
        df_ref = pd.read_csv(first_csv)
        row = df_ref.iloc[0]
        total_pos = row['tp'] + row['fn']
        total_neg = row['fp'] + row['tn']
        total = total_pos + total_neg
        base_rate = total_pos / total

        # Theoretical Perfect (Uniform Negatives)
        # Curve: y = 1 for x <= base_rate; y = base_rate/x for x > base_rate
        x_uniform = np.linspace(base_rate, 1.0, 100)
        y_uniform = base_rate / x_uniform
        # Add the initial flat part
        x_uniform = np.concatenate(([0], x_uniform))
        y_uniform = np.concatenate(([1], y_uniform))
        
        auc_uniform = base_rate * (1 - np.log(base_rate))
        
        ax[0].plot(x_uniform, y_uniform, '--', color='black', label=f"Perfect (Uniform Negs), AUC: {auc_uniform:.3f}")

        # Theoretical Perfect (Zero Negatives)
        # Curve: (0,1) -> (base_rate, 1) -> (1, base_rate)
        x_zeros = [0, base_rate, 1.0]
        y_zeros = [1.0, 1.0, base_rate]
        
        auc_zeros = base_rate + 0.5 * (1 - base_rate**2)
        
        ax[0].plot(x_zeros, y_zeros, ':', color='black', label=f"Perfect (Zero Negs), AUC: {auc_zeros:.3f}")

    for run_name, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        x = df["cache_hit_ratio"].values
        x2 = df["recall"].values
        y = df["precision"].values
        sorted_idx = np.argsort(x)
        x = x[sorted_idx]
        x2 = x2[sorted_idx]
        y = y[sorted_idx]
        auc = np.trapezoid(y, x)
        auc2 = np.trapezoid(y, x2)
        ax[0].plot(df["cache_hit_ratio"], df["precision"], markersize=3, label=f"{run_name}, AUC: {auc:.3f}")
        ax[1].plot(df["recall"], df["precision"], markersize=3, label=f"{run_name}, AUC: {auc2:.3f}")

    ax[0].set_xlabel("Cache Hit Ratio")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision vs Cache Hit Ratio")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Recall vs Precision")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].legend()
    ax[1].legend()

    fig.suptitle(args.title)

    plt.tight_layout()

    plt.savefig(os.path.join(args.output_dir, "precision_vs_cache_hit_ratio.png"))
    plt.close()


if __name__ == "__main__":
    main()
