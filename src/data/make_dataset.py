from __future__ import annotations

from pathlib import Path
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


DATASET_HANDLE = "saurabhshahane/fake-news-classification"


def _pick_csv_file(dataset_dir: Path) -> Path:
    """
    Find the most likely CSV file inside the downloaded Kaggle dataset folder.
    Priority:
      1) file name contains 'welfake'
      2) otherwise first .csv found
    """
    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No .csv files found inside: {dataset_dir}. "
            "Open the folder and check what files exist."
        )

    for p in csv_files:
        if "welfake" in p.name.lower():
            return p

    return csv_files[0]


def load_welfake_from_kaggle() -> pd.DataFrame:
    """
    Downloads dataset via kagglehub, finds the CSV, then loads it into a DataFrame.
    """
    # 1) Download dataset locally (returns local folder path as string)
    dataset_dir_str = kagglehub.dataset_download(DATASET_HANDLE)
    dataset_dir = Path(dataset_dir_str)

    # 2) Pick CSV file automatically
    csv_path = _pick_csv_file(dataset_dir)

    # 3) Load using dataset_load (new API)
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        DATASET_HANDLE,
        str(csv_path.relative_to(dataset_dir)).replace("\\", "/"),
    )
    return df
