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
        "v1": "vizio/v1/customer_analysis/langcache-v1/threshold_sweep_results.csv",
        "v3": "vizio/v1/customer_analysis/langcache-v3/threshold_sweep_results.csv",
        "MiniLM": "vizio/v1/customer_analysis/langcache-minilm/threshold_sweep_results.csv",
    }

    fig, ax = plt.subplots(figsize=(16, 7), ncols=2)

    for run_name, csv_path in csv_paths.items():
        df = pd.read_csv(csv_path)
        df = df.iloc[:-1]
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
