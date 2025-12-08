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
        
        # CHANGED: Create two subplots: one for curves, one for the AUC bar chart
        fig, (ax_main, ax_bar) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})
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

        # Theoretical AUCs storage
        theory_aucs = {}

        # Plot theoretical curves
        if base_rate is not None:
            # Theoretical Perfect (Uniform Negatives)
            x_uniform = np.linspace(base_rate, 1.0, 100)
            y_uniform = base_rate / x_uniform
            x_uniform = np.concatenate(([0], x_uniform))
            y_uniform = np.concatenate(([1], y_uniform))
            auc_uniform = base_rate * (1 - np.log(base_rate))
            ax_main.plot(x_uniform, y_uniform, '--', color='black', label=f"Perfect (Uniform Negs), AUC: {auc_uniform:.3f}")
            theory_aucs['Uniform'] = auc_uniform

            # Theoretical Perfect (Zero Negatives)
            x_zeros = [0, base_rate, 1.0]
            y_zeros = [1.0, 1.0, base_rate]
            auc_zeros = base_rate + 0.5 * (1 - base_rate**2)
            ax_main.plot(x_zeros, y_zeros, ':', color='black', label=f"Perfect (Zero Negs), AUC: {auc_zeros:.3f}")
            theory_aucs['ZeroNegs'] = auc_zeros

        sorted_models = sorted(model_data.keys())
        
        # Store data for the bar plot
        auc_records = []

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
                    # Remove the last row because it's always precision = 1.0
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
                    # Use numpy.trapezoid (NumPy 2.0) or numpy.trapz (older)
                    try:
                        auc_val = np.trapezoid(p_interp, common_chr)
                    except AttributeError:
                        auc_val = np.trapz(p_interp, common_chr)
                    
                    aucs_pchr.append(auc_val)

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

            label_chr = f"{model_name}, AUC: {mean_auc_pchr:.3f} ± {std_auc_pchr:.3f}" # Simplified label for main plot, AUC is in bar chart
            ax_main.plot(common_chr, mean_p_chr, label=label_chr, color=color)
            if valid_runs > 1:
                ax_main.fill_between(common_chr, mean_p_chr - std_p_chr, mean_p_chr + std_p_chr, color=color, alpha=0.2)
            
            # Save data for bar chart
            auc_records.append({
                'name': model_name,
                'mean': mean_auc_pchr,
                'std': std_auc_pchr,
                'color': color
            })

        # --- Configure Main Curve Plot ---
        ax_main.set_xlabel("Cache Hit Ratio")
        ax_main.set_ylabel("Precision")
        ax_main.set_title("Precision vs Cache Hit Ratio")
        ax_main.grid(True)
        ax_main.legend()

        # --- Configure Bar Chart ---
        if auc_records:
            # Sort by mean AUC (ascending so best is at top)
            auc_records.sort(key=lambda x: x['mean'], reverse=False)
            
            names = [r['name'] for r in auc_records]
            means = [r['mean'] for r in auc_records]
            stds = [r['std'] for r in auc_records]
            bar_colors = [r['color'] for r in auc_records]
            y_pos = np.arange(len(names))

            ax_bar.barh(y_pos, means, xerr=stds, color=bar_colors, align='center', capsize=5, alpha=0.8)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(names)
            ax_bar.set_xlabel("AUC")
            ax_bar.set_title("AUC Comparison")
            ax_bar.grid(axis='x', linestyle='--', alpha=0.7)

            # Add theoretical lines to bar chart
            if 'Uniform' in theory_aucs:
                ax_bar.axvline(theory_aucs['Uniform'], color='black', linestyle='--', alpha=0.7)
            if 'ZeroNegs' in theory_aucs:
                ax_bar.axvline(theory_aucs['ZeroNegs'], color='black', linestyle=':', alpha=0.7)
            
            # Set x-limits to focus on relevant area if needed, or 0-1
            # ax_bar.set_xlim(0, 1.05) 

        fig.suptitle(f"Performance on {dataset_name.split('_')[0]}")
        plt.tight_layout()
        output_path = os.path.join(dataset_full_path, "precision_vs_cache_hit_ratio.png")
        print(f"Saving plot to {output_path}")
        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    main()