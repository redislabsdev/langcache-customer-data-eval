import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chr_analysis import (
    calculate_cache_hit_ratio_for_threshold,
    load_data,
    sweep_cache_hit_ratios,
)


class TestCacheHitRatioAnalysis(unittest.TestCase):
    """Test suite for cache hit ratio analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

    def test_calculate_cache_hit_ratio_for_threshold(self):
        """Test cache hit ratio calculation for a single threshold."""
        # Test data: 10 scores
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        # At threshold 0.5, we expect 6 scores >= 0.5 (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        chr = calculate_cache_hit_ratio_for_threshold(scores, 0.5)
        self.assertAlmostEqual(chr, 0.6, places=2)
        
        # At threshold 0.0, all scores should be cache hits
        chr = calculate_cache_hit_ratio_for_threshold(scores, 0.0)
        self.assertAlmostEqual(chr, 1.0, places=2)
        
        # At threshold 1.0, only score 1.0 should be a cache hit
        chr = calculate_cache_hit_ratio_for_threshold(scores, 1.0)
        self.assertAlmostEqual(chr, 0.1, places=2)
        
        # At threshold > 1.0, no cache hits
        chr = calculate_cache_hit_ratio_for_threshold(scores, 1.1)
        self.assertAlmostEqual(chr, 0.0, places=2)

    def test_calculate_cache_hit_ratio_empty_array(self):
        """Test cache hit ratio with empty array."""
        scores = np.array([])
        chr = calculate_cache_hit_ratio_for_threshold(scores, 0.5)
        self.assertEqual(chr, 0.0)

    def test_sweep_cache_hit_ratios(self):
        """Test threshold sweep for cache hit ratios."""
        # Test data with known distribution
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        # Sweep with 5 steps
        results_df = sweep_cache_hit_ratios(scores, steps=5)
        
        # Check DataFrame structure
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn("threshold", results_df.columns)
        self.assertIn("cache_hit_ratio", results_df.columns)
        self.assertEqual(len(results_df), 5)
        
        # Check that cache hit ratio decreases as threshold increases
        chr_values = results_df["cache_hit_ratio"].values
        for i in range(len(chr_values) - 1):
            self.assertGreaterEqual(chr_values[i], chr_values[i + 1])
        
        # Check bounds
        self.assertAlmostEqual(results_df.iloc[0]["cache_hit_ratio"], 1.0, places=2)
        self.assertGreaterEqual(results_df.iloc[-1]["cache_hit_ratio"], 0.0)

    def test_sweep_cache_hit_ratios_monotonic(self):
        """Test that cache hit ratio is monotonically decreasing."""
        # Larger dataset
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100)
        
        results_df = sweep_cache_hit_ratios(scores, steps=50)
        
        # Cache hit ratio should be monotonically non-increasing
        chr_values = results_df["cache_hit_ratio"].values
        for i in range(len(chr_values) - 1):
            self.assertGreaterEqual(chr_values[i], chr_values[i + 1])

    def test_load_data_basic(self):
        """Test loading data for cache hit ratio analysis."""
        # Create a temporary CSV file
        test_csv_path = os.path.join(self.test_dir, "test_data.csv")
        test_data = pd.DataFrame({
            "sentence1": ["query1", "query2", "query3", "query4", "query5"],
            "sentence2": ["match1", "match2", "match3", "match4", "match5"],
        })
        test_data.to_csv(test_csv_path, index=False)
        
        # Load data
        queries, cache = load_data(test_csv_path, n_samples=3)
        
        # Check that queries has correct size
        self.assertEqual(len(queries), 3)
        
        # Check that cache has remaining data (5 total - 3 queries = 2 cache entries)
        self.assertEqual(len(cache), 2)
        
        # Check that queries and cache don't overlap (no common indices)
        query_sentences = set(queries["sentence1"])
        cache_sentences = set(cache["sentence1"])
        self.assertEqual(len(query_sentences.intersection(cache_sentences)), 0)
        
        # Check that all original data is accounted for
        all_original_sentences = set(test_data["sentence1"])
        self.assertEqual(all_original_sentences, query_sentences.union(cache_sentences))

    def test_load_data_all_samples(self):
        """Test loading all data when n_samples >= total rows."""
        test_csv_path = os.path.join(self.test_dir, "test_data_small.csv")
        test_data = pd.DataFrame({
            "text": ["a", "b", "c"],
        })
        test_data.to_csv(test_csv_path, index=False)
        
        # Request more samples than available
        queries, cache = load_data(test_csv_path, n_samples=10)
        
        # Should only get 3 (all available) for queries
        self.assertEqual(len(queries), 3)
        # Cache should be empty since all data was used for queries
        self.assertEqual(len(cache), 0)
        
        # All original data should be in queries
        all_original_text = set(test_data["text"])
        all_queries_text = set(queries["text"])
        self.assertEqual(all_original_text, all_queries_text)

    def test_cache_hit_ratio_edge_cases(self):
        """Test edge cases for cache hit ratio calculation."""
        # All same values
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        chr = calculate_cache_hit_ratio_for_threshold(scores, 0.5)
        self.assertEqual(chr, 1.0)
        
        chr = calculate_cache_hit_ratio_for_threshold(scores, 0.51)
        self.assertEqual(chr, 0.0)
        
        # Single value
        scores = np.array([0.7])
        chr = calculate_cache_hit_ratio_for_threshold(scores, 0.7)
        self.assertEqual(chr, 1.0)
        
        chr = calculate_cache_hit_ratio_for_threshold(scores, 0.8)
        self.assertEqual(chr, 0.0)

    def test_sweep_with_uniform_distribution(self):
        """Test sweep with uniformly distributed scores."""
        # Create uniformly distributed scores
        scores = np.linspace(0, 1, 101)  # 101 points from 0 to 1
        
        results_df = sweep_cache_hit_ratios(scores, steps=11)
        
        # At threshold 0.0, all should be hits
        self.assertAlmostEqual(results_df.iloc[0]["cache_hit_ratio"], 1.0, places=2)
        
        # At threshold 0.5, approximately half should be hits
        mid_idx = len(results_df) // 2
        mid_chr = results_df.iloc[mid_idx]["cache_hit_ratio"]
        self.assertGreater(mid_chr, 0.4)
        self.assertLess(mid_chr, 0.6)
        
        # At threshold 1.0, only one should be a hit
        self.assertAlmostEqual(results_df.iloc[-1]["cache_hit_ratio"], 1/101, places=2)


if __name__ == "__main__":
    unittest.main()

