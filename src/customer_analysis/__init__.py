"""Customer analysis package for semantic cache evaluation."""

from .data_processing import (
    load_data,
    postprocess_results_for_metrics,
    run_matching,
    run_matching_redis,
)
from .embedding_interface import EmbeddingModel, get_embedding_model
from .embedding_providers import (
    EmbeddingProvider,
    GeminiProvider,
    HuggingFaceProvider,
    OpenAIProvider,
    get_embedding_provider,
)
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
from .similarity_matcher import SimilarityMatcher

__all__ = [
    # File handling
    "FileHandler",
    # Embedding
    "EmbeddingModel",
    "get_embedding_model",
    # Providers
    "EmbeddingProvider",
    "HuggingFaceProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "get_embedding_provider",
    "SimilarityMatcher",
    # Data processing
    "load_data",
    "postprocess_results_for_metrics",
    "run_matching",
    "run_matching_redis",
    # Metrics
    "evaluate_threshold_on_results",
    "sweep_thresholds_on_results",
    "calculate_f_beta_score",
    "calculate_metrics",
    # Plotting
    "generate_plots",
    "plot_cache_hit_ratio",
    # S3 utilities
    "s3_upload_dataframe_csv",
    "s3_upload_matplotlib_png",
    # Redis
    "RedisVectorIndex",
]
