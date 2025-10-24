"""Customer analysis package for semantic cache evaluation."""

from .data_processing import (
    load_data,
    postprocess_results_for_metrics,
    run_matching,
    run_matching_redis,
)
from .embedding_interface import NeuralEmbedding
from .file_handler import FileHandler
from .metrics_util import (
    calculate_f_beta_score,
    calculate_metrics,
    evaluate_threshold_on_results,
    sweep_thresholds_on_results,
)
from .plotting import generate_plots, plot_cache_hit_ratio
from .query_engine import RedisVectorIndex
from .s3_util import s3_upload_dataframe_csv, s3_upload_matplotlib_png

__all__ = [
    "FileHandler",
    "NeuralEmbedding",
    "load_data",
    "postprocess_results_for_metrics",
    "run_matching",
    "run_matching_redis",
    "evaluate_threshold_on_results",
    "sweep_thresholds_on_results",
    "calculate_f_beta_score",
    "calculate_metrics",
    "generate_plots",
    "plot_cache_hit_ratio",
    "s3_upload_dataframe_csv",
    "s3_upload_matplotlib_png",
    "RedisVectorIndex",
]
