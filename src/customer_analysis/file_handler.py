import os

import matplotlib.pyplot as plt
import pandas as pd

from src.customer_analysis.s3_util import s3_upload_dataframe_csv, s3_upload_matplotlib_png


def make_output_path(output_dir: str, filename: str) -> str:
    """Construct output path for both local and S3 paths."""
    if output_dir.startswith("s3://"):
        # For S3, always use forward slash
        return f"{output_dir.rstrip('/')}/{filename}"
    else:
        # For local, use os.path.join
        return os.path.join(output_dir or "", filename or "")


class FileHandler:
    """Abstraction for handling both local and S3 file operations."""

    @staticmethod
    def write_csv(df: pd.DataFrame, output_dir: str = None, filename: str = None, index: bool = False):
        """Write dataframe to CSV file."""
        assert output_dir is not None and filename is not None, "output_dir and filename must be provided"
        path = make_output_path(output_dir, filename)
        if output_dir.startswith("s3://"):
            s3_upload_dataframe_csv(df.reset_index() if index else df, path)
        else:
            df.to_csv(path, index=index)
        print(f"{'Uploaded' if output_dir.startswith('s3://') else 'Saved'}:", path)

    @staticmethod
    def read_csv(file_path: str = None) -> pd.DataFrame:
        """Read CSV file to dataframe."""
        assert file_path is not None, "file_path must be provided"
        return pd.read_csv(file_path)

    @staticmethod
    def save_matplotlib_plot(output_dir: str = None, filename: str = None, dpi: int = 150):
        """Save current matplotlib plot."""
        path = make_output_path(output_dir, filename)
        if path.startswith("s3://"):
            s3_upload_matplotlib_png(path, dpi=dpi)
        else:
            plt.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close()
        print(f"{'Uploaded' if path.startswith('s3://') else 'Saved'}:", path)
