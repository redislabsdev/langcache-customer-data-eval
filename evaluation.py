import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Conditional import for LLM-as-a-Judge functionality
try:
    from llm_sim_eval.pipeline import run_llm_local_sim_prediction_pipeline
    from llm_sim_eval.prompts import DEFAULT_PROMPTS, Prompt

    HAS_LLM_SIM_EVAL = True
except ImportError:
    HAS_LLM_SIM_EVAL = False

from src.customer_analysis import (
    FileHandler,
    generate_plots,
    load_data,
    plot_cache_hit_ratio,
    postprocess_results_for_metrics,
    run_matching,
    run_matching_redis,
    sweep_thresholds_on_results,
)

RANDOM_SEED = 42

# Set the random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def run_llm_as_a_judge(query_pairs, args):
    """Run LLM-as-a-Judge evaluation on query pairs."""
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


def run_chr_analysis(queries: pd.DataFrame, args):
    """Run cache hit ratio analysis only (fast, no LLM judge)."""
    print("\n" + "=" * 60)
    print("Running Cache Hit Ratio Analysis (CHR-only mode)")
    print("=" * 60)

    # Save matches
    FileHandler.write_csv(queries[[args.sentence_column, "matches", "best_scores"]], args.output_dir, "chr_matches.csv")

    # Sweep cache hit ratios
    similarity_scores = queries["best_scores"].values
    results_df = sweep_thresholds_on_results(
        pd.DataFrame({"similarity_score": similarity_scores})
    )

    # Save sweep results
    FileHandler.write_csv(results_df, args.output_dir, "chr_sweep.csv")

    # Generate plot
    print("Generating plot...")
    plot_cache_hit_ratio(results_df, args.output_dir, "chr_vs_threshold.png")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Cache Hit Ratio Analysis Summary")
    print("=" * 60)
    print(f"Total queries analyzed: {len(queries)}")
    print(f"Similarity score range: [{similarity_scores.min():.4f}, {similarity_scores.max():.4f}]")
    print(f"Mean similarity score: {similarity_scores.mean():.4f}")
    print(f"Median similarity score: {np.median(similarity_scores):.4f}")
    print("=" * 60)


def run_full_evaluation(queries: pd.DataFrame, args):
    """Run full evaluation pipeline with LLM-as-a-Judge."""
    print("\n" + "=" * 60)
    print("Running Full Evaluation (with LLM-as-a-Judge)")
    print("=" * 60)

    # Check if LLM library is available
    if not HAS_LLM_SIM_EVAL:
        raise ImportError(
            "llm-sim-eval is required for full evaluation mode. "
            "Please install it or run without --full flag for CHR-only analysis."
        )

    # Save matches
    FileHandler.write_csv(queries[[args.sentence_column, "matches", "best_scores"]], args.output_dir, "matches.csv")

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

    FileHandler.write_csv(
        final_df[[args.sentence_column, "matches", "similarity_score", "actual_label"]],
        args.output_dir,
        "llm_as_a_judge_results.csv",
    )

    results_df = sweep_thresholds_on_results(final_df)

    FileHandler.write_csv(results_df, args.output_dir, "threshold_sweep_results.csv")

    # ------------------------------
    # Stage four: Generating plots
    # ------------------------------
    print("Stage four: Generating plots...")
    generate_plots(
        results_df,
        output_dir=args.output_dir,
        precision_filename="precision_vs_cache_hit_ratio.png",
        metrics_filename="metrics_over_threshold.png",
    )


def main(args):
    """Main function for evaluation pipeline."""
    # Load data
    print("Loading data...")
    queries, cache = load_data(query_log_path=args.query_log_path, cache_path=args.cache_path, n_samples=args.n_samples)

    # ------------------------------
    # Stage one: Matching
    # ------------------------------
    print("Stage one: Matching...")
    if args.use_redis:
        queries = run_matching_redis(queries, cache, args)
    else:
        queries = run_matching(queries, cache, args)

    # Choose evaluation mode
    if args.full:
        run_full_evaluation(queries, args)
    else:
        run_chr_analysis(queries, args)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic cache evaluation: CHR-only (default) or full evaluation with LLM-as-a-Judge (--full)"
    )
    parser.add_argument(
        "--query_log_path", type=str, required=True, help="Path to the query log CSV file (local or S3)"
    )
    parser.add_argument("--sentence_column", type=str, required=True, help="Column name for the sentences to evaluate")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory (local or S3)")
    parser.add_argument(
        "--n_samples", type=int, required=False, default=100, help="Number of samples to evaluate (default: 100)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="redis/langcache-embed-v3.1",
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--cache_path", type=str, required=False, default=None, help="Path to the cache CSV file (local or S3)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation with LLM-as-a-Judge (default: CHR-only analysis)",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        required=False,
        default="microsoft/Phi-4-mini-instruct",
        help="Name of the LLM to use (only for --full mode)",
    )
    parser.add_argument(
        "--sweep_steps",
        type=int,
        default=200,
        help="Number of threshold steps in sweep (default: 200)",
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
