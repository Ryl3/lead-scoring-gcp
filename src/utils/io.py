"""Shared I/O helpers — load raw tables, save artefacts."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
import config


def load_raw() -> dict:
    """Load all five raw CSVs into a dict keyed by table name."""
    return {name: pd.read_csv(path) for name, path in config.RAW_FILES.items()}


def load_master() -> pd.DataFrame:
    """Join all raw tables on lead_id and return the master dataframe."""
    dfs = load_raw()
    return (
        dfs["crm"]
        .merge(dfs["web"],      on="lead_id")
        .merge(dfs["email"],    on="lead_id")
        .merge(dfs["trial"],    on="lead_id")
        .merge(dfs["outcomes"], on="lead_id")
    )


def load_feature_matrix() -> pd.DataFrame:
    """Load the processed feature matrix (output of Stage 2)."""
    return pd.read_csv(config.FEATURE_MATRIX)


def save_feature_matrix(df: pd.DataFrame) -> None:
    config.DATA_PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.FEATURE_MATRIX, index=False)
    print(f"Saved feature matrix → {config.FEATURE_MATRIX}")