import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import crawl_results

def calculate_precision_at_1(csv_path):
    # Reads llm_as_a_judge_results.csv
    try:
        df = pd.read_csv(csv_path)
        if "actual_label" not in df.columns:
            return None
        # actual_label should be 0 or 1
        return df["actual_label"].mean()
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def get_model_short_name(model_name):
    if "gte-modernbert" in model_name:
        return "ModernBERT"
    elif "v1" in model_name:
        return "v1"
    elif "v2" in model_name:
        return "v2"
    elif "v3.1" in model_name:
        return "v3-small"
    return model_name.split("/")[-1]

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
            stats = data_map[dataset].get(model, {metric_key: {"mean": None}})
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
                model, {metric_key: {"mean": None, "std": None}}
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
    parser = argparse.ArgumentParser("Usage: python calculate_precision_at_1.py --base_dir <base_dir>")
    parser.add_argument("--base_dir", type=str, required=False, default="complete_benchmark_results")
    args = parser.parse_args()

    base_dir = args.base_dir
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

    data_map = {}

    # 1. Collect Data
    print(f"{'Dataset':<30} | {'Model':<30} | Precision@1")
    print("-" * 80)

    for dataset in dataset_names:
        data_map[dataset] = {}
        for model in sorted_models:
            run_paths = benchmark_map[dataset].get(model, [])
            precisions = []

            for run_path in run_paths:
                details_path = os.path.join(run_path, "llm_as_a_judge_results.csv")
                if os.path.exists(details_path):
                    p1 = calculate_precision_at_1(details_path)
                    if p1 is not None:
                        precisions.append(p1)
            
            stats = {"mean": None, "std": None}
            if precisions:
                mean_val = np.mean(precisions)
                std_val = np.std(precisions) if len(precisions) > 1 else 0.0
                stats["mean"] = mean_val
                stats["std"] = std_val
                
                print(f"{dataset:<30} | {model:<30} | {mean_val:.4f} ± {std_val:.4f}")
            else:
                 pass # No data for this model/dataset
            
            data_map[dataset][model] = {"precision": stats}

    print("\n" + "="*80 + "\n")

    # 2. Print Latex Table
    print_latex_table(
        title="Precision@1 (Mean $\\pm$ Std, $\\uparrow$)",
        metric_key="precision",
        dataset_names=dataset_names,
        sorted_models=sorted_models,
        model_short_names=model_short_names,
        data_map=data_map,
        label="tab:precision_at_1",
        minimize=False
    )

    # 3. Plot HBar
    for dataset in dataset_names:
        dataset_full_path = os.path.join(base_dir, dataset)
        if not os.path.exists(dataset_full_path):
            continue
            
        models_in_ds = []
        means = []
        stds = []
        
        for model in sorted_models:
            stats = data_map[dataset][model]["precision"]
            if stats["mean"] is not None:
                models_in_ds.append(model_short_names[model])
                means.append(stats["mean"])
                stds.append(stats["std"])
        
        if not models_in_ds:
            continue
            
        # Sort by mean precision
        zipped = sorted(zip(means, stds, models_in_ds))
        means_sorted, stds_sorted, models_sorted = zip(*zipped)
        
        plt.figure(figsize=(10, max(4, len(models_sorted) * 0.8 + 2)))
        y_pos = np.arange(len(models_sorted))
        
        plt.barh(y_pos, means_sorted, xerr=stds_sorted, align='center', alpha=0.8, capsize=5)
        plt.yticks(y_pos, models_sorted)
        plt.xlabel('Precision@1')
        plt.title(f'Precision@1 for {dataset}')
        plt.xlim(0, 1.05)
        plt.grid(axis='x', alpha=0.3)
        
        # Add values to bars
        for i, v in enumerate(means_sorted):
            plt.text(v + 0.01, i, f"{v:.3f}", va='center')

        plt.tight_layout()
        output_path = os.path.join(dataset_full_path, "precision_at_1.png")
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()

