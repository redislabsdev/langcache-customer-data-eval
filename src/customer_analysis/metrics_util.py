from typing import Dict, List

import numpy as np
import pandas as pd


def evaluate_threshold_on_results(results_df: pd.DataFrame, threshold: float) -> Dict:
    """Evaluate metrics for a given threshold on pre-computed results."""

    # Calculate cache hit ratio
    cache_hits = (results_df["similarity_score"] >= threshold).sum()
    total_samples = len(results_df)
    cache_hit_ratio = cache_hits / total_samples if total_samples > 0 else 0.0

    if "actual_label" not in results_df.columns or results_df["actual_label"].isnull().all():
        return {
            "threshold": threshold,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "f0_5_score": np.nan,
            "f05_chr_score": np.nan,
            "cache_hit_ratio": cache_hit_ratio,
            "tp": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tn": np.nan,
            "accuracy": np.nan,
        }
    else:
        pred_labels = (results_df["similarity_score"] >= threshold).astype(int)
        actual_labels = results_df["actual_label"].astype(int)

        # Calculate confusion matrix
        tp = ((pred_labels == 1) & (actual_labels == 1)).sum()
        fp = ((pred_labels == 1) & (actual_labels == 0)).sum()
        fn = ((pred_labels == 0) & (actual_labels == 1)).sum()
        tn = ((pred_labels == 0) & (actual_labels == 0)).sum()

        # Calculate metrics
        metrics = calculate_metrics(tp, fp, fn)

        # This is similar to the FBeta score (a balanced metric between precision and recall).
        # In this case we are using the cache hit ratio instead of recall.
        # This creates a score that puts about 4x more weight on precision than cache hit ratio but still balances the two.
        f05_chr_score = calculate_f_beta_score(metrics["precision"], cache_hit_ratio, 0.5)
        return {
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "f0_5_score": metrics["f0_5_score"],
            "f05_chr_score": f05_chr_score,
            "cache_hit_ratio": cache_hit_ratio,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "accuracy": (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0,
        }


def sweep_thresholds_on_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Perform threshold sweep and return results."""
    print("\nPerforming threshold sweep")
    min_score = results_df["similarity_score"].min()
    max_score = results_df["similarity_score"].max() 
    steps = 200
    thresholds = np.linspace(min_score, 1.0 if max_score < 1.0 else max_score, steps)
    results = []

    for i, threshold in enumerate(thresholds):
        result = evaluate_threshold_on_results(results_df, float(threshold))
        results.append(result)

    return pd.DataFrame(results)


def calculate_f_beta_score(a, b, beta: float) -> float:
    """
    Calculate F-beta score. A harmonic mean for beta=0.5.
    You can use this to balance different metrics, such as precision and recall, or precision and cache hit ratio.

    Args:
        a: First metric
        b: Second metric
        beta: Beta value for the harmonic mean

    Returns:
        F-beta score
    """
    f_beta = (1 + beta**2) * a * b / ((beta**2 * a) + b) if (a + b) else 0.0
    return f_beta


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Calculate classification metrics: precision, recall, f1_score."""
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = calculate_f_beta_score(precision, recall, 1.0)
    f0_5 = calculate_f_beta_score(precision, recall, 0.5)
    return {"precision": precision, "recall": recall, "f1_score": f1, "f0_5_score": f0_5}
