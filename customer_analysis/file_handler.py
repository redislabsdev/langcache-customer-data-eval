import os

import matplotlib.pyplot as plt
import pandas as pd
from s3_util import s3_upload_dataframe_csv, s3_upload_matplotlib_png


class FileHandler:
    """Abstraction for handling both local and S3 file operations."""

    def __init__(self, path: str):
        self.path = path
        self.is_s3 = path.startswith("s3://")

        # For local paths, ensure directory exists
        if not self.is_s3:
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

    def write_csv(self, df: pd.DataFrame, index: bool = False):
        """Write dataframe to CSV file."""
        if self.is_s3:
            s3_upload_dataframe_csv(df.reset_index() if index else df, self.path)
        else:
            df.to_csv(self.path, index=index)
        print(f"{'Uploaded' if self.is_s3 else 'Saved'}:", self.path)

    def read_csv(self) -> pd.DataFrame:
        """Read CSV file to dataframe."""
        return pd.read_csv(self.path)

    def save_matplotlib_plot(self, dpi: int = 150):
        """Save current matplotlib plot."""
        if self.is_s3:
            s3_upload_matplotlib_png(self.path, dpi=dpi)
        else:
            plt.savefig(self.path, dpi=dpi, bbox_inches="tight")
            plt.close()
        print(f"{'Uploaded' if self.is_s3 else 'Saved'}:", self.path)
