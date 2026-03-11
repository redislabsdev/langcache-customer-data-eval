import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import crawl_results

def main():
    parser = argparse.ArgumentParser("Usage: python plot_precision_over_threshold.py --base_dir <base_dir>")
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
            
        # Create a single plot for this dataset
        plt.figure(figsize=(12, 8))
        plt.title(f"Precision vs Threshold for {dataset_name}", fontsize=16)
        
        sorted_models = sorted(model_data.keys())
        
        # Common thresholds for interpolation
        common_thresholds = np.linspace(0.5, 1, 200)

        # Use a qualitative colormap with many distinct colors (tab20)
        # and cycle through them if we have more models than colors.
        cmap = plt.get_cmap("tab20")
        
        # Define some linestyles to help distinguish further
        linestyles = ['-', '--', '-.', ':']
        
        for idx, model_name in enumerate(sorted_models):
            # improved color selection:
            color = cmap(idx % 20)
            linestyle = linestyles[(idx // 20) % len(linestyles)]
            
            run_paths = model_data[model_name]
            try:
                all_precisions = []
                valid_runs = 0

                for run_path in run_paths:
                    sweep_path = os.path.join(run_path, "threshold_sweep_results.csv")
                    if not os.path.exists(sweep_path):
                        continue

                    try:
                        df = pd.read_csv(sweep_path)
                        if "threshold" not in df.columns or "precision" not in df.columns:
                            continue
                        
                        # Sort by threshold
                        df = df.sort_values("threshold")
                        
                        x = df["threshold"].values
                        y = df["precision"].values
                        
                        y_interp = np.interp(common_thresholds, x, y)
                        all_precisions.append(y_interp)
                        valid_runs += 1
                        
                    except Exception as e:
                        print(f"Error processing run {run_path}: {e}")

                if valid_runs == 0:
                    continue

                mean_precision = np.mean(all_precisions, axis=0)
                std_precision = np.std(all_precisions, axis=0) if valid_runs > 1 else np.zeros_like(mean_precision)

                plt.plot(common_thresholds, mean_precision, label=model_name, color=color, linestyle=linestyle, linewidth=2)
                plt.fill_between(
                    common_thresholds, 
                    np.maximum(0, mean_precision - std_precision), 
                    np.minimum(1, mean_precision + std_precision), 
                    alpha=0.1, 
                    color=color
                )

            except Exception as e:
                print(f"Error plotting {model_name}: {e}")

        plt.xlabel("Threshold")
        plt.ylabel("Precision")
        plt.ylim(0, 1.05)
        plt.xlim(0.5, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(dataset_full_path, "precision_over_threshold.png")
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()

