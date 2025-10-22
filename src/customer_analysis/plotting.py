import matplotlib.pyplot as plt

from src.customer_analysis.file_handler import FileHandler


def generate_plots(results_df, precision_plot_path, metrics_plot_path):
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
    precision_handler = FileHandler(precision_plot_path)
    precision_handler.save_matplotlib_plot()

    # Plot 2: Metrics over Threshold
    plt.figure()
    plt.plot(results_df["threshold"], results_df["precision"], label="Precision")
    plt.plot(results_df["threshold"], results_df["cache_hit_ratio"], label="Cache Hit Ratio")
    plt.plot(
        results_df["threshold"], results_df["precision"] * results_df["cache_hit_ratio"], label="True Positive Rate"
    )
    plt.legend()
    metrics_handler = FileHandler(metrics_plot_path)
    metrics_handler.save_matplotlib_plot()
