import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.customer_analysis import FileHandler, NeuralEmbedding, load_data

RANDOM_SEED = 42

# Set the random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def run_matching(queries, cache, args):
    """Run embedding-based matching to find best cache matches for each query."""
    embedding_model = NeuralEmbedding(args.model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    queries["best_scores"] = 0

    best_indices, best_scores, decision_methods = embedding_model.calculate_best_matches_with_cache_large_dataset(
        queries=queries[args.sentence_column].to_list(),
        cache=cache[args.sentence_column].to_list(),
        batch_size=512,
        early_stop=min(args.n_samples, len(queries)),
    )

    queries["best_scores"] = best_scores
    queries["matches"] = cache.iloc[best_indices][args.sentence_column].to_list()

    del embedding_model
    torch.cuda.empty_cache()

    return queries


def calculate_cache_hit_ratio_for_threshold(similarity_scores: np.ndarray, threshold: float) -> float:
    """Calculate cache hit ratio for a given threshold."""
    cache_hits = (similarity_scores >= threshold).sum()
    total_samples = len(similarity_scores)
    return cache_hits / total_samples if total_samples > 0 else 0.0


def sweep_cache_hit_ratios(similarity_scores: np.ndarray, steps: int = 200) -> pd.DataFrame:
    """
    Perform threshold sweep and calculate cache hit ratios.
    
    Args:
        similarity_scores: Array of similarity scores
        steps: Number of threshold steps
        
    Returns:
        DataFrame with columns: threshold, cache_hit_ratio
    """
    print("\nPerforming cache hit ratio threshold sweep...")
    min_score = similarity_scores.min()
    max_score = 1.0
    thresholds = np.linspace(min_score, max_score, steps)
    
    results = []
    for threshold in thresholds:
        chr = calculate_cache_hit_ratio_for_threshold(similarity_scores, threshold)
        results.append({
            "threshold": threshold,
            "cache_hit_ratio": chr
        })
    
    return pd.DataFrame(results)


def plot_cache_hit_ratio(results_df: pd.DataFrame, output_path: str):
    """
    Plot cache hit ratio vs threshold.
    
    Args:
        results_df: DataFrame with threshold and cache_hit_ratio columns
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["threshold"], results_df["cache_hit_ratio"], linewidth=2)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Cache Hit Ratio", fontsize=12)
    plt.title("Cache Hit Ratio vs Threshold", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    handler = FileHandler(output_path)
    handler.save_matplotlib_plot()


def main(args):
    """Main function for cache hit ratio analysis."""
    
    # Construct output paths
    def make_output_path(filename):
        if args.output_dir.startswith("s3://"):
            return f"{args.output_dir.rstrip('/')}/{filename}"
        else:
            return os.path.join(args.output_dir, filename)
    
    # Load data
    print("Loading data...")
    queries, cache = load_data(query_log_path=args.query_log_path, cache_path=args.cache_path, n_samples=args.n_samples)
    
    # Run matching
    print("Running matching...")
    queries = run_matching(queries, cache, args)
    
    # Save matches
    matches_path = make_output_path("chr_matches.csv")
    matches_handler = FileHandler(matches_path)
    matches_handler.write_csv(queries[[args.sentence_column, "matches", "best_scores"]])
    
    # Sweep cache hit ratios
    similarity_scores = queries["best_scores"].values
    results_df = sweep_cache_hit_ratios(similarity_scores, steps=args.sweep_steps)
    
    # Save sweep results
    sweep_path = make_output_path("chr_sweep.csv")
    sweep_handler = FileHandler(sweep_path)
    sweep_handler.write_csv(results_df)
    
    # Generate plot
    print("Generating plot...")
    plot_path = make_output_path("chr_vs_threshold.png")
    plot_cache_hit_ratio(results_df, plot_path)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Cache Hit Ratio Analysis Summary")
    print("="*50)
    print(f"Total queries analyzed: {len(queries)}")
    print(f"Similarity score range: [{similarity_scores.min():.4f}, {similarity_scores.max():.4f}]")
    print(f"Mean similarity score: {similarity_scores.mean():.4f}")
    print(f"Median similarity score: {np.median(similarity_scores):.4f}")
    print("\nCache Hit Ratios at common thresholds:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        chr = calculate_cache_hit_ratio_for_threshold(similarity_scores, threshold)
        print(f"  Threshold {threshold:.1f}: {chr:.2%}")
    print("="*50)
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache Hit Ratio Analysis")
    parser.add_argument(
        "--query_log_path", 
        type=str, 
        required=True, 
        help="Path to the query log CSV file (local or S3)"
    )
    parser.add_argument(
        "--cache_path", 
        type=str, 
        default=None, 
        help="Path to the cache CSV file (local or S3)"
    )
    parser.add_argument(
        "--sentence_column", 
        type=str, 
        required=True, 
        help="Column name for the sentences to analyze"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the output directory (local or S3)"
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=100, 
        help="Number of samples to analyze (default: 100)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="redis/langcache-embed-v3.1", 
        help="Name of the embedding model to use"
    )
    parser.add_argument(
        "--sweep_steps", 
        type=int, 
        default=200, 
        help="Number of threshold steps in sweep (default: 200)"
    )
    
    args = parser.parse_args()
    main(args)

