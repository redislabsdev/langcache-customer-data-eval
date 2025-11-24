import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Usage: python plot_precision_vs_cache_hit_ratio.py <path_to_csv>
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    csv_paths = {
        "v1": "rado_synthetic/v1/",
        "v2": "rado_synthetic/v2/",
        "v3": "rado_synthetic/v3/",
        "v3.1": "rado_synthetic/v3.1/",
    }

    n = len(csv_paths)
    rows = n // 2 + 1
    cols = 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))

    # Handle case where there's only one row
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (run_name, csv_path) in enumerate(csv_paths.items()):
        row = idx // cols
        col = idx % cols

        df = pd.read_csv(csv_path + "llm_as_a_judge_results.csv")
        df = df.iloc[:-1]

        # Extract scores
        pos = df[df["actual_label"] == 1]["similarity_score"].values
        neg = df[df["actual_label"] == 0]["similarity_score"].values

        # Unified bins across both distributions
        lo = min(pos.min(), neg.min())
        hi = max(pos.max(), neg.max())
        edges = np.linspace(lo, hi, 101)  # 100 bins → 101 edges

        # Histogram values (densities)
        pos_hist, _ = np.histogram(pos, bins=edges, density=True)
        neg_hist, _ = np.histogram(neg, bins=edges, density=True)
        widths = np.diff(edges)

        # Sanity: each integrates to ~1
        assert np.isclose(
            np.sum(pos_hist * widths), 1.0, atol=1e-6
        ), f"{run_name}: Positive histogram doesn't integrate to 1"
        assert np.isclose(
            np.sum(neg_hist * widths), 1.0, atol=1e-6
        ), f"{run_name}: Negative histogram doesn't integrate to 1"

        # Compute overlap and TV
        overlap = np.sum(np.minimum(pos_hist, neg_hist) * widths)
        tv = 0.5 * np.sum(np.abs(pos_hist - neg_hist) * widths)

        # Identity should hold tightly now
        assert np.isclose(overlap, 1 - tv, atol=1e-6), f"{run_name}: overlap != 1 - tv"

        # Plot histograms
        axes[row, col].hist(pos, bins=edges, alpha=0.7, label="Positive", color="green", density=True)
        axes[row, col].hist(neg, bins=edges, alpha=0.7, label="Negative", color="red", density=True)

        # Show overlap area in title
        axes[row, col].set_title(f"{run_name}\nOverlap = {overlap:.3f}")

        axes[row, col].set_xlabel("Similarity Score")
        axes[row, col].set_ylabel("Density")
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "similarity_distributions.png"), dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
