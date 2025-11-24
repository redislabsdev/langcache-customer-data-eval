import argparse
import glob
import os

# Import logic from evaluation.py
# We need to make sure we can import these.
# Assuming evaluation.py is in the same directory or we add it to path.
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    from llm_sim_eval.models.huggingface import CacheHitClassifier, VLLMConfig
    from llm_sim_eval.prompts import DEFAULT_PROMPTS

    HAS_LLM_SIM_EVAL_LIB = True
except ImportError:
    HAS_LLM_SIM_EVAL_LIB = False

from evaluation import FileHandler, run_chr_analysis, run_full_evaluation, run_matching, run_matching_redis


# Mock args object to pass to evaluation functions
class BenchmarkArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark on a directory of datasets with bootstrapping.")

    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing dataset CSV files.")
    parser.add_argument(
        "--dataset_names", type=str, nargs="+", required=True, help="List of dataset names to evaluate."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory for results.")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="List of model names to evaluate.")
    parser.add_argument("--sentence_column", type=str, required=True, help="Column name for sentences.")

    # Bootstrapping args
    parser.add_argument("--n_runs", type=int, default=5, help="Number of bootstrap runs per dataset/model.")
    parser.add_argument(
        "--sample_ratio", type=float, default=0.8, help="Fraction of data to use for each run (0.8 = 80%)."
    )
    parser.add_argument("--n_samples", type=int, default=100, help="Number of query samples to evaluate per run.")

    # Evaluation args
    parser.add_argument("--full", action="store_true", help="Run full evaluation with LLM-as-a-Judge.")
    parser.add_argument("--llm_name", type=str, default="microsoft/Phi-4-mini-instruct", help="LLM name for judge.")
    parser.add_argument("--use_redis", action="store_true", help="Use Redis for matching.")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379")
    parser.add_argument("--redis_index_name", type=str, default="idx_cache_match")
    parser.add_argument("--redis_doc_prefix", type=str, default="cache:")
    parser.add_argument("--redis_batch_size", type=int, default=256)

    args = parser.parse_args()

    # Initialize Judge Model if needed (to avoid reloading)
    llm_classifier = None
    if args.full:
        if not HAS_LLM_SIM_EVAL_LIB:
            print("Error: llm_sim_eval not found but --full requested.")
            return

        print(f"Initializing Judge Model: {args.llm_name}...")
        prompt_text = DEFAULT_PROMPTS["empty_prompt"].text
        llm_classifier = CacheHitClassifier(
            VLLMConfig(
                model_id=args.llm_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                gpu_memory_utilization=0.6,
            ),
            prompt=prompt_text,
        )

    # Find all CSV datasets
    dataset_files = [args.dataset_dir + "/" + dataset_name for dataset_name in args.dataset_names]
    if not dataset_files:
        print(f"No CSV files found in {args.dataset_dir}")
        return

    print(f"Found {len(dataset_files)} datasets: {[os.path.basename(f) for f in dataset_files]}")
    print(f"Models to evaluate: {args.models}")

    for dataset_path in dataset_files:
        dataset_name = os.path.basename(dataset_path).replace(".csv", "")
        print(f"\n{'='*60}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"{'='*60}")

        try:
            full_df = pd.read_csv(dataset_path)
        except Exception as e:
            print(f"Error reading {dataset_path}: {e}")
            continue

        for model_name in args.models:
            print(f"\n  Model: {model_name}")

            # Sanitize model name for directory structure
            safe_model_name = model_name.replace("/", "_")

            for run_i in range(1, args.n_runs + 1):
                print(f"    Run {run_i}/{args.n_runs}...")

                # 1. Bootstrapping Logic
                # Sample 80% of the universe
                run_universe = full_df.sample(
                    frac=args.sample_ratio, random_state=run_i
                )  # Use run_i as seed for reproducibility per run

                # Split into Queries (n_samples) and Cache (remainder)
                if len(run_universe) <= args.n_samples:
                    print(
                        f"      Warning: Dataset size ({len(run_universe)}) <= n_samples ({args.n_samples}). Skipping."
                    )
                    continue

                queries = run_universe.sample(n=args.n_samples, random_state=run_i + 1000)
                cache = run_universe.drop(queries.index)

                # Shuffle cache
                cache = cache.sample(frac=1, random_state=run_i + 2000).reset_index(drop=True)
                queries = queries.reset_index(drop=True)

                # 2. Construct Output Path
                timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                run_output_dir = os.path.join(args.output_dir, dataset_name, safe_model_name, f"run_{run_i}", timestamp)
                os.makedirs(run_output_dir, exist_ok=True)

                # 3. Prepare Args for Evaluation
                eval_args = BenchmarkArgs(
                    query_log_path=dataset_path,  # Not strictly used by logic below but good for reference
                    sentence_column=args.sentence_column,
                    output_dir=run_output_dir,
                    n_samples=args.n_samples,
                    model_name=model_name,
                    cache_path=None,
                    full=args.full,
                    llm_name=args.llm_name,
                    llm_model=llm_classifier,
                    sweep_steps=200,  # Default
                    use_redis=args.use_redis,
                    redis_url=args.redis_url,
                    redis_index_name=args.redis_index_name,
                    redis_doc_prefix=args.redis_doc_prefix,
                    redis_batch_size=args.redis_batch_size,
                    # device defaults to code logic
                )

                # 4. Run Evaluation
                try:
                    print("      Matching...")
                    if args.use_redis:
                        queries_matched = run_matching_redis(queries.copy(), cache.copy(), eval_args)
                    else:
                        queries_matched = run_matching(queries.copy(), cache.copy(), eval_args)

                    print("      Evaluating...")
                    if args.full:
                        run_full_evaluation(queries_matched, eval_args)
                    else:
                        run_chr_analysis(queries_matched, eval_args)

                except Exception as e:
                    print(f"      Error in run {run_i}: {e}")
                    import traceback

                    traceback.print_exc()

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
