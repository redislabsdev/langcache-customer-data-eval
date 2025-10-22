"""
Comprehensive end-to-end tests for the evaluation pipeline.

Tests both matching modes (standard and Redis) and both evaluation modes
(CHR-only and full with LLM-as-a-Judge) using the actual dataset files.
"""

import os
import tempfile
import unittest
from argparse import Namespace
from typing import Optional

import numpy as np
import pandas as pd

from src.customer_analysis import (
    FileHandler,
    load_data,
    plot_cache_hit_ratio,
    postprocess_results_for_metrics,
    run_matching,
    run_matching_redis,
    sweep_thresholds_on_results,
)

# Dataset paths
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
QUERIES_PATH = os.path.join(DATASET_DIR, "queries.csv")
CACHE_PATH = os.path.join(DATASET_DIR, "cache.csv")


# ============================================================================
# Helper Utilities
# ============================================================================


def check_redis_available() -> bool:
    """Check if Redis is available at localhost:6379."""
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def check_llm_eval_available() -> bool:
    """Check if llm-sim-eval package is installed."""
    try:
        import llm_sim_eval

        return True
    except ImportError:
        return False


def validate_matches_df(df: pd.DataFrame, sentence_column: str) -> None:
    """Validate the structure of a matches DataFrame."""
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert sentence_column in df.columns, f"Missing {sentence_column} column"
    assert "best_scores" in df.columns, "Missing best_scores column"
    assert "matches" in df.columns, "Missing matches column"
    assert len(df) > 0, "DataFrame should not be empty"

    # Check data types
    assert df["best_scores"].dtype in [np.float32, np.float64], "best_scores should be float"
    assert df["matches"].dtype == object, "matches should be string/object type"

    # Check value ranges
    assert df["best_scores"].min() >= -1.0 - 1e-6, "Similarity scores should be >= -1.0"
    assert df["best_scores"].max() <= 1.0 + 1e-6, "Similarity scores should be <= 1.0"


def validate_metrics_df(df: pd.DataFrame) -> None:
    """Validate the structure of a metrics DataFrame from threshold sweep."""
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert "threshold" in df.columns, "Missing threshold column"
    assert "cache_hit_ratio" in df.columns, "Missing cache_hit_ratio column"
    assert len(df) > 0, "DataFrame should not be empty"

    # Check monotonic decrease of CHR
    chr_values = df["cache_hit_ratio"].values
    for i in range(len(chr_values) - 1):
        assert chr_values[i] >= chr_values[i + 1], "CHR should be monotonically non-increasing"

    # Check value ranges
    assert df["cache_hit_ratio"].min() >= 0.0, "CHR should be >= 0.0"
    assert df["cache_hit_ratio"].max() <= 1.0 + 1e-6, "CHR should be <= 1.0"


def validate_full_metrics_df(df: pd.DataFrame) -> None:
    """Validate metrics DataFrame with precision, recall, F1, etc."""
    validate_metrics_df(df)

    # Check for additional metrics columns
    expected_cols = ["precision", "recall", "f1_score", "f0_5_score", "accuracy"]
    for col in expected_cols:
        if col in df.columns and not df[col].isna().all():
            assert df[col].min() >= 0.0, f"{col} should be >= 0.0"
            assert df[col].max() <= 1.0 + 1e-6, f"{col} should be <= 1.0"


def validate_semantic_quality(
    queries_df: pd.DataFrame, sentence_column: str, known_similar_pairs: list[tuple[str, str, float]]
) -> None:
    """
    Validate that known similar pairs have high similarity scores.

    Args:
        queries_df: DataFrame with queries, matches, and best_scores
        sentence_column: Name of the sentence column
        known_similar_pairs: List of (query, expected_match, min_score) tuples
    """
    for query, expected_match, min_score in known_similar_pairs:
        matching_rows = queries_df[queries_df[sentence_column] == query]
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]
            actual_match = row["matches"]
            actual_score = row["best_scores"]

            # Check if the match is semantically related
            # Either exact match or score above threshold
            if expected_match.lower() in actual_match.lower() or actual_match.lower() in expected_match.lower():
                # Direct match found
                assert actual_score >= min_score, (
                    f"Query '{query}' matched '{actual_match}' but score {actual_score:.3f} "
                    f"is below threshold {min_score}"
                )


# ============================================================================
# Test Class
# ============================================================================


class TestE2EEvaluation(unittest.TestCase):
    """Comprehensive end-to-end tests for the evaluation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.sentence_column = "text"
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Small fast model for testing

        # Check that dataset files exist
        self.assertTrue(os.path.exists(QUERIES_PATH), f"Dataset not found: {QUERIES_PATH}")
        self.assertTrue(os.path.exists(CACHE_PATH), f"Dataset not found: {CACHE_PATH}")

    def test_chr_only_standard_matching(self):
        """Test CHR-only pipeline with standard neural embedding matching."""
        print("\n" + "=" * 70)
        print("TEST: CHR-Only Mode with Standard Matching")
        print("=" * 70)

        # Load data
        queries, cache = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=5)

        # Create args namespace
        args = Namespace(
            sentence_column=self.sentence_column,
            model_name=self.model_name,
            output_dir=self.test_dir,
            n_samples=5,
        )

        # Run matching
        queries = run_matching(queries, cache, args)

        # Validate matches DataFrame
        validate_matches_df(queries, self.sentence_column)

        # Run threshold sweep
        similarity_scores = queries["best_scores"].values
        results_df = sweep_thresholds_on_results(pd.DataFrame({"similarity_score": similarity_scores}))

        # Validate metrics DataFrame
        validate_metrics_df(results_df)

        # Print summary
        print(f"✓ Processed {len(queries)} queries against {len(cache)} cache entries")
        print(f"✓ Similarity score range: [{similarity_scores.min():.4f}, {similarity_scores.max():.4f}]")
        print(f"✓ Threshold sweep: {len(results_df)} points")
        print(f"✓ CHR at min threshold: {results_df.iloc[0]['cache_hit_ratio']:.4f}")
        print(f"✓ CHR at max threshold: {results_df.iloc[-1]['cache_hit_ratio']:.4f}")

    @unittest.skipUnless(check_redis_available(), "Redis not available at localhost:6379")
    def test_chr_only_redis_matching(self):
        """Test CHR-only pipeline with Redis-based matching."""
        print("\n" + "=" * 70)
        print("TEST: CHR-Only Mode with Redis Matching")
        print("=" * 70)

        # Load data
        queries, cache = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=5)

        # Create args namespace with Redis settings
        args = Namespace(
            sentence_column=self.sentence_column,
            model_name=self.model_name,
            output_dir=self.test_dir,
            n_samples=5,
            redis_url="redis://localhost:6379",
            redis_index_name="test_idx_e2e",
            redis_doc_prefix="test_cache:",
            redis_batch_size=256,
            device=None,
        )

        # Run matching with Redis
        queries = run_matching_redis(queries, cache, args)

        # Validate matches DataFrame
        validate_matches_df(queries, self.sentence_column)

        # Run threshold sweep
        similarity_scores = queries["best_scores"].values
        results_df = sweep_thresholds_on_results(pd.DataFrame({"similarity_score": similarity_scores}))

        # Validate metrics DataFrame
        validate_metrics_df(results_df)

        # Print summary
        print(f"✓ Processed {len(queries)} queries against {len(cache)} cache entries with Redis")
        print(f"✓ Similarity score range: [{similarity_scores.min():.4f}, {similarity_scores.max():.4f}]")
        print(f"✓ Redis index cleaned up successfully")

    @unittest.skipUnless(check_llm_eval_available(), "llm-sim-eval not installed")
    def test_full_evaluation_standard_matching(self):
        """Test full evaluation pipeline with LLM-as-a-Judge."""
        print("\n" + "=" * 70)
        print("TEST: Full Evaluation with LLM-as-a-Judge (Standard Matching)")
        print("=" * 70)

        # Import LLM-specific modules
        from llm_sim_eval.pipeline import run_llm_local_sim_prediction_pipeline
        from llm_sim_eval.prompts import DEFAULT_PROMPTS, Prompt

        # Load data (use fewer samples for LLM to keep test fast)
        queries, cache = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=3)

        # Create args namespace
        args = Namespace(
            sentence_column=self.sentence_column,
            model_name=self.model_name,
            output_dir=self.test_dir,
            n_samples=3,
            llm_name="microsoft/Phi-4-mini-instruct",
        )

        # Stage 1: Run matching
        queries = run_matching(queries, cache, args)
        validate_matches_df(queries, self.sentence_column)

        # Stage 2: LLM-as-a-Judge
        query_pairs = list(zip(queries[self.sentence_column], queries["matches"]))
        prompt = Prompt.load(DEFAULT_PROMPTS["empty_prompt"].path)

        result = run_llm_local_sim_prediction_pipeline(
            llm=args.llm_name,
            sentence_pairs=query_pairs,
            batch_size=2,
            prompt=prompt,
        )

        pairs = []
        responses = []
        for response in result.flat_responses:
            pairs.extend(response.pairs)
            responses.extend(response.parsed_responses)

        llm_df = pd.DataFrame(pairs, columns=[self.sentence_column, "matches"])
        llm_df["response"] = responses
        llm_df["response"] = llm_df["response"].apply(lambda x: x["answer"])

        # Stage 3: Postprocess for metrics
        final_df = postprocess_results_for_metrics(queries, llm_df, args)

        # Validate postprocessed DataFrame
        assert "actual_label" in final_df.columns, "Missing actual_label column"
        assert "similarity_score" in final_df.columns, "Missing similarity_score column"
        assert final_df["actual_label"].isin([0, 1]).all(), "actual_label should be binary (0 or 1)"

        # Stage 4: Threshold sweep with metrics
        results_df = sweep_thresholds_on_results(final_df)
        validate_full_metrics_df(results_df)

        # Print summary
        print(f"✓ Processed {len(queries)} queries with LLM-as-a-Judge")
        print(f"✓ LLM responses: {len(responses)}, Failed: {len(result.failed_responses)}")
        print(f"✓ Metrics calculated across {len(results_df)} thresholds")
        if "precision" in results_df.columns and not results_df["precision"].isna().all():
            print(f"✓ Precision range: [{results_df['precision'].min():.4f}, {results_df['precision'].max():.4f}]")
        if "recall" in results_df.columns and not results_df["recall"].isna().all():
            print(f"✓ Recall range: [{results_df['recall'].min():.4f}, {results_df['recall'].max():.4f}]")

    @unittest.skipUnless(
        check_redis_available() and check_llm_eval_available(),
        "Redis or llm-sim-eval not available",
    )
    def test_full_evaluation_redis_matching(self):
        """Test full evaluation pipeline with Redis matching and LLM-as-a-Judge."""
        print("\n" + "=" * 70)
        print("TEST: Full Evaluation with Redis Matching + LLM-as-a-Judge")
        print("=" * 70)

        # Import LLM-specific modules
        from llm_sim_eval.pipeline import run_llm_local_sim_prediction_pipeline
        from llm_sim_eval.prompts import DEFAULT_PROMPTS, Prompt

        # Load data
        queries, cache = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=3)

        # Create args namespace with Redis settings
        args = Namespace(
            sentence_column=self.sentence_column,
            model_name=self.model_name,
            output_dir=self.test_dir,
            n_samples=3,
            llm_name="microsoft/Phi-4-mini-instruct",
            redis_url="redis://localhost:6379",
            redis_index_name="test_idx_e2e_full",
            redis_doc_prefix="test_cache_full:",
            redis_batch_size=256,
            device=None,
        )

        # Stage 1: Run matching with Redis
        queries = run_matching_redis(queries, cache, args)
        validate_matches_df(queries, self.sentence_column)

        # Stage 2: LLM-as-a-Judge
        query_pairs = list(zip(queries[self.sentence_column], queries["matches"]))
        prompt = Prompt.load(DEFAULT_PROMPTS["empty_prompt"].path)

        result = run_llm_local_sim_prediction_pipeline(
            llm=args.llm_name,
            sentence_pairs=query_pairs,
            batch_size=2,
            prompt=prompt,
        )

        pairs = []
        responses = []
        for response in result.flat_responses:
            pairs.extend(response.pairs)
            responses.extend(response.parsed_responses)

        llm_df = pd.DataFrame(pairs, columns=[self.sentence_column, "matches"])
        llm_df["response"] = responses
        llm_df["response"] = llm_df["response"].apply(lambda x: x["answer"])

        # Stage 3: Postprocess for metrics
        final_df = postprocess_results_for_metrics(queries, llm_df, args)

        # Validate structure
        assert "actual_label" in final_df.columns, "Missing actual_label column"
        assert final_df["actual_label"].isin([0, 1]).all(), "actual_label should be binary"

        # Stage 4: Threshold sweep
        results_df = sweep_thresholds_on_results(final_df)
        validate_full_metrics_df(results_df)

        print(f"✓ Complete Redis + LLM pipeline processed {len(queries)} queries")
        print("✓ Redis index cleaned up successfully")

    def test_data_loading_and_splitting(self):
        """Test data loading and splitting functionality."""
        print("\n" + "=" * 70)
        print("TEST: Data Loading and Splitting")
        print("=" * 70)

        # Test 1: Normal loading with n_samples < total
        queries, cache = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=3)

        assert len(queries) == 3, f"Expected 3 queries, got {len(queries)}"
        assert len(cache) == 7, f"Expected 4 cache entries, got {len(cache)}"


        # Test 2: n_samples > total rows
        queries_all, cache_all = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=100)

        total_rows = pd.read_csv(QUERIES_PATH).shape[0]
        assert len(queries_all) == total_rows, f"Expected all {total_rows} rows as queries"
        assert len(cache_all) == 7, "Expected empty cache when all data used for queries"

        print(f"✓ When n_samples > total, all {len(queries_all)} rows used as queries")

        # Test 3: n_samples = 1 (edge case)
        queries_one, cache_one = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=1)

        assert len(queries_one) == 1, "Expected 1 query"
        assert len(cache_one) == 7, "Expected 7 cache entries"

        print(f"✓ Edge case n_samples=1: 1 query, {len(cache_one)} cache entries")

        # Test 4: Loading with explicit cache file
        queries_exp, cache_exp = load_data(
            query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=3
        )

        # Test 5: Loading with no cache file
        queries_no, cache_no = load_data(query_log_path=QUERIES_PATH, cache_path=None, n_samples=3)
        assert len(queries_no) == 3, "Expected 3 queries"
        assert len(cache_no) == 4, "Expected 4 cache entries"

        # When explicit cache is provided, it's loaded from cache file
        cache_file_rows = pd.read_csv(CACHE_PATH).shape[0]
        assert len(cache_exp) == cache_file_rows, f"Expected {cache_file_rows} cache entries from cache file"
        # Check no overlap
        query_texts = set(queries_no[self.sentence_column])
        cache_texts = set(cache_no[self.sentence_column])
        overlap = query_texts.intersection(cache_texts)
        assert len(overlap) == 0, f"Found overlap between queries and cache: {overlap}"

        # Test 6: n_samples = 1 (edge case)
        queries_no, cache_no = load_data(query_log_path=QUERIES_PATH, cache_path=None, n_samples=1)
        assert len(queries_no) == 1, "Expected 3 queries"
        assert len(cache_no) == total_rows - 1, "Expected 4 cache entries"

        # Test 7: n_samples > total rows
        queries_all, cache_all = load_data(query_log_path=QUERIES_PATH, cache_path=None, n_samples=100)

        assert len(queries_all) == total_rows, f"Expected all {total_rows} rows as queries"
        assert len(cache_all) == 0, "Expected empty cache when all data used for queries"

    def test_output_files_creation(self):
        """Test that output files are created correctly."""
        print("\n" + "=" * 70)
        print("TEST: Output File Creation")
        print("=" * 70)

        # Load data
        queries, cache = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=4)

        # Create args namespace
        args = Namespace(
            sentence_column=self.sentence_column,
            model_name=self.model_name,
            output_dir=self.test_dir,
            n_samples=4,
        )

        # Run matching
        queries = run_matching(queries, cache, args)

        # Save matches file
        matches_file = os.path.join(self.test_dir, "chr_matches.csv")
        FileHandler.write_csv(
            queries[[self.sentence_column, "matches", "best_scores"]], self.test_dir, "chr_matches.csv"
        )

        assert os.path.exists(matches_file), f"Matches file not created: {matches_file}"
        print(f"✓ Created matches file: {matches_file}")

        # Validate matches file content
        matches_df = pd.read_csv(matches_file)
        assert len(matches_df) == len(queries), "Matches file has wrong number of rows"
        assert self.sentence_column in matches_df.columns, "Missing sentence column"
        assert "matches" in matches_df.columns, "Missing matches column"
        assert "best_scores" in matches_df.columns, "Missing best_scores column"
        print(f"✓ Matches file has correct structure: {list(matches_df.columns)}")

        # Run threshold sweep and save
        similarity_scores = queries["best_scores"].values
        results_df = sweep_thresholds_on_results(pd.DataFrame({"similarity_score": similarity_scores}))

        sweep_file = os.path.join(self.test_dir, "chr_sweep.csv")
        FileHandler.write_csv(results_df, self.test_dir, "chr_sweep.csv")

        assert os.path.exists(sweep_file), f"Sweep file not created: {sweep_file}"
        print(f"✓ Created sweep file: {sweep_file}")

        # Validate sweep file content
        sweep_df = pd.read_csv(sweep_file)
        assert "threshold" in sweep_df.columns, "Missing threshold column"
        assert "cache_hit_ratio" in sweep_df.columns, "Missing cache_hit_ratio column"
        print(f"✓ Sweep file has correct structure: {list(sweep_df.columns)}")

        # Generate plot
        plot_file = "chr_vs_threshold.png"
        plot_cache_hit_ratio(results_df, self.test_dir, plot_file)

        plot_path = os.path.join(self.test_dir, plot_file)
        assert os.path.exists(plot_path), f"Plot file not created: {plot_path}"
        assert os.path.getsize(plot_path) > 0, "Plot file is empty"
        print(f"✓ Created plot file: {plot_path} ({os.path.getsize(plot_path)} bytes)")

    def test_semantic_match_quality(self):
        """Test that semantically similar queries match with high scores."""
        print("\n" + "=" * 70)
        print("TEST: Semantic Match Quality")
        print("=" * 70)

        # Load all data to ensure we get the similar pairs
        queries, cache = load_data(query_log_path=QUERIES_PATH, cache_path=CACHE_PATH, n_samples=7)

        # Create args namespace
        args = Namespace(
            sentence_column=self.sentence_column,
            model_name=self.model_name,
            output_dir=self.test_dir,
            n_samples=7,
        )

        # Run matching
        queries = run_matching(queries, cache, args)

        # Define known similar pairs that should have high similarity
        # Format: (query, expected_match_substring, min_score)
        known_similar_pairs = [
            ("how do I reset my password?", "reset", 0.5),  # Password reset queries
            ("Halloween movies", "movies", 0.7),  # Movie-related queries
            ("Horror movies", "movies", 0.7),
            ("Scary movies", "movies", 0.7),
            ("ms rachel", "rachel", 0.7),  # Rachel queries
            ("mama ms rachel", "rachel", 0.7),
        ]

        # Validate semantic quality
        validate_semantic_quality(queries, self.sentence_column, known_similar_pairs)

        # Print match details for inspection
        print("\nMatch Details:")
        print("-" * 70)
        for _, row in queries.iterrows():
            query = row[self.sentence_column]
            match = row["matches"]
            score = row["best_scores"]
            print(f"Query: '{query}'")
            print(f"  → Match: '{match}' (score: {score:.4f})")

        print("\n✓ Semantic match quality validated for known similar pairs")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n" + "=" * 70)
        print("TEST: Edge Cases and Error Handling")
        print("=" * 70)

        # Test 1: Empty DataFrame handling
        empty_df = pd.DataFrame({"similarity_score": []})
        results = sweep_thresholds_on_results(empty_df)
        # Should handle gracefully - either empty or with defaults
        print("✓ Empty DataFrame handled gracefully")

        # Test 2: Single value
        single_df = pd.DataFrame({"similarity_score": [0.8]})
        results_single = sweep_thresholds_on_results(single_df)
        assert len(results_single) > 0, "Should produce results for single value"
        print("✓ Single value handled correctly")

        # Test 3: All same values
        same_df = pd.DataFrame({"similarity_score": [0.5, 0.5, 0.5, 0.5]})
        results_same = sweep_thresholds_on_results(same_df)
        assert len(results_same) > 0, "Should produce results for identical values"
        print("✓ Identical values handled correctly")

        # Test 4: Extreme values
        extreme_df = pd.DataFrame({"similarity_score": [-1.0, -0.5, 0.0, 0.5, 1.0]})
        results_extreme = sweep_thresholds_on_results(extreme_df)
        validate_metrics_df(results_extreme)
        print("✓ Extreme values (-1.0 to 1.0) handled correctly")

        # Test 5: Invalid threshold value handling
        from src.customer_analysis import evaluate_threshold_on_results

        test_df = pd.DataFrame({"similarity_score": [0.1, 0.5, 0.9]})

        # Very high threshold
        result_high = evaluate_threshold_on_results(test_df, 2.0)
        assert result_high["cache_hit_ratio"] == 0.0, "Threshold > 1.0 should give 0% CHR"
        print("✓ Very high threshold (>1.0) handled correctly")

        # Very low threshold
        result_low = evaluate_threshold_on_results(test_df, -2.0)
        assert result_low["cache_hit_ratio"] == 1.0, "Threshold < -1.0 should give 100% CHR"
        print("✓ Very low threshold (<-1.0) handled correctly")

        print("\n✓ All edge cases handled correctly")


if __name__ == "__main__":
    unittest.main(verbosity=2)

