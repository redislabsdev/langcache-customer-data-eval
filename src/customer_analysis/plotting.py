import matplotlib.pyplot as plt
import pandas as pd

from src.customer_analysis.file_handler import FileHandler


def generate_plots(
    results_df: pd.DataFrame, output_dir: str = None, precision_filename: str = None, metrics_filename: str = None
):
    # Plot 1: Precision vs Cache Hit Ratio
    results_df.plot(
        x="cache_hit_ratio",
        y="precision",
        kind="line",
        marker="o",
        markersize=1,
        linewidth=1,
        ylabel="Precision",
        xlabel="Cache Hit Ratio",
    )
    FileHandler.save_matplotlib_plot(output_dir, precision_filename)

    # Plot 2: Metrics over Threshold
    plt.figure()
    plt.plot(results_df["threshold"], results_df["precision"], label="Precision")
    plt.plot(results_df["threshold"], results_df["cache_hit_ratio"], label="Cache Hit Ratio")
    plt.plot(
        results_df["threshold"], results_df["precision"] * results_df["cache_hit_ratio"], label="True Positive Rate"
    )
    plt.legend()
    FileHandler.save_matplotlib_plot(output_dir, metrics_filename)


def plot_cache_hit_ratio(results_df: pd.DataFrame, output_dir: str = None, filename: str = None):
    """
    Plot cache hit ratio vs threshold.

    Args:
        results_df: DataFrame with threshold and cache_hit_ratio columns
        plot_handler: FileHandler to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["threshold"], results_df["cache_hit_ratio"], linewidth=2)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Cache Hit Ratio", fontsize=12)
    plt.title("Cache Hit Ratio vs Threshold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    FileHandler.save_matplotlib_plot(output_dir, filename)
