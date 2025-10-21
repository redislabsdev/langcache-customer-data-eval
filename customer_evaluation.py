import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from llm_sim_eval.pipeline import run_llm_local_sim_prediction_pipeline
from llm_sim_eval.prompts import DEFAULT_PROMPTS, Prompt

from src.customer_analysis import (
    FileHandler,
    NeuralEmbedding,
    generate_plots,
    load_data,
    postprocess_results_for_metrics,
    run_matching,
    run_matching_redis,
    sweep_thresholds_on_results,
)

RANDOM_SEED = 42

# set the random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def run_llm_as_a_judge(query_pairs, args):
    prompt = Prompt.load(DEFAULT_PROMPTS["empty_prompt"].path)
    result = run_llm_local_sim_prediction_pipeline(
        llm=args.llm_name,
        sentence_pairs=query_pairs,
        batch_size=2,
        prompt=prompt,
    )

    n_failed = len(result.failed_responses)

    pairs = []
    responses = []
    for response in result.flat_responses:
        pairs.extend(response.pairs)
        responses.extend(response.parsed_responses)

    llm_df = pd.DataFrame(pairs, columns=[args.sentence_column, "matches"])
    llm_df["response"] = responses
    llm_df["response"] = llm_df["response"].apply(lambda x: x["answer"])

    return llm_df, n_failed


def main(args):
    # Construct output paths by joining output_dir with filename
    def make_output_path(filename):
        if args.output_dir.startswith("s3://"):
            # For S3, always use forward slash
            return f"{args.output_dir.rstrip('/')}/{filename}"
        else:
            # For local, use os.path.join
            return os.path.join(args.output_dir, filename)

    # Input paths are used directly - they can already be full S3 or local paths
    queries, cache = load_data(args.query_log_path, args.cache_path, args.n_samples)

    # ------------------------------
    # Stage one: Matching
    # ------------------------------
    print("Stage one: Matching...")
    if args.use_redis:
        queries = run_matching_redis(queries, cache, args)
    else:
        queries = run_matching(queries, cache, args)

    matches_handler = FileHandler(make_output_path("matches.csv"))
    matches_handler.write_csv(queries[[args.sentence_column, "matches", "best_scores"]])

    # ------------------------------
    # Stage two: LLM-as-a-Judge
    # ------------------------------
    print("Stage two: LLM-as-a-Judge...")
    voice_query_pairs = list(zip(queries[args.sentence_column], queries["matches"]))
    llm_df, n_failed = run_llm_as_a_judge(voice_query_pairs, args)
    print("Number of discarded queries during LLM-as-a-Judge:", n_failed)

    # ------------------------------
    # Stage three: Metrics calculation
    # ------------------------------
    print("Stage three: Metrics calculation...")
    final_df = postprocess_results_for_metrics(queries, llm_df, args)

    judge_results_handler = FileHandler(make_output_path("llm_as_a_judge_results.csv"))
    judge_results_handler.write_csv(final_df[[args.sentence_column, "matches", "similarity_score", "actual_label"]])

    results = sweep_thresholds_on_results(final_df, {"model_type": "neural"})
    results_df = pd.DataFrame(results)

    threshold_results_handler = FileHandler(make_output_path("threshold_sweep_results.csv"))
    threshold_results_handler.write_csv(results_df)

    # ------------------------------
    # Stage four: Generating plots
    # ------------------------------
    print("Stage four: Generating plots...")
    precision_plot_path = make_output_path("precision_vs_cache_hit_ratio.png")
    metrics_plot_path = make_output_path("metrics_over_threshold.png")
    generate_plots(results_df, precision_plot_path, metrics_plot_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run customer evaluation")
    parser.add_argument(
        "--query_log_path", type=str, required=True, help="Path to the query log CSV file (local or S3)"
    )
    parser.add_argument("--sentence_column", type=str, required=True, help="Column name for the sentences to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory (local or S3)")
    parser.add_argument(
        "--n_samples", type=int, required=False, default=1000, help="Number of samples to evaluate with LLM"
    )
    parser.add_argument(
        "--model_name", type=str, required=False, default="redis/langcache-embed-v1", help="Name of the model to use"
    )
    parser.add_argument(
        "--cache_path", type=str, required=False, default=None, help="Path to the cache CSV file (local or S3)"
    )
    parser.add_argument(
        "--llm_name", type=str, required=False, default="microsoft/Phi-4-mini-instruct", help="Name of the LLM to use"
    )
    parser.add_argument("--use_redis", action="store_true", help="Use Redis for matching (default: False)")
    parser.add_argument(
        "--redis_url",
        type=str,
        default="redis://localhost:6379",
        help="Redis connection URL (default: redis://localhost:6379)",
    )
    parser.add_argument(
        "--redis_index_name", type=str, default="idx_cache_match", help="Redis index name (default: idx_cache_match)"
    )
    parser.add_argument(
        "--redis_doc_prefix", type=str, default="cache:", help="Redis document key prefix (default: cache:)"
    )
    parser.add_argument(
        "--redis_batch_size", type=int, default=256, help="Batch size for Redis vector operations (default: 256)"
    )
    args = parser.parse_args()

    main(args)
