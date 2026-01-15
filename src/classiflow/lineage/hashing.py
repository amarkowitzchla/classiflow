"""Data hashing utilities for lineage tracking."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_file_hash(
    file_path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """
    Compute cryptographic hash of a file.

    Parameters
    ----------
    file_path : Path
        Path to file
    algorithm : str
        Hash algorithm (sha256, sha1, md5)
    chunk_size : int
        Size of chunks for reading file

    Returns
    -------
    hash_hex : str
        Hexadecimal hash digest
    """
    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_dataframe_hash(
    df: pd.DataFrame,
    algorithm: str = "sha256",
    canonical: bool = True,
) -> str:
    """
    Compute hash of a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    algorithm : str
        Hash algorithm
    canonical : bool
        If True, sort columns and rows for canonical representation

    Returns
    -------
    hash_hex : str
        Hexadecimal hash digest
    """
    hasher = hashlib.new(algorithm)

    if canonical:
        # Sort columns and rows for reproducibility
        df = df.sort_index(axis=0).sort_index(axis=1)

    # Convert to bytes using parquet-like serialization
    # This handles mixed types better than pickle
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from io import BytesIO

        table = pa.Table.from_pandas(df)
        buf = BytesIO()
        pq.write_table(table, buf, compression="none")
        hasher.update(buf.getvalue())
    except (ImportError, Exception):
        # Fallback: use pickle (less stable across pandas versions)
        logger.warning("pyarrow not available, using pickle for hashing (less stable)")
        import pickle
        hasher.update(pickle.dumps(df, protocol=4))

    return hasher.hexdigest()


def compute_canonical_hash(
    data: Union[pd.DataFrame, np.ndarray, bytes],
    algorithm: str = "sha256",
) -> str:
    """
    Compute canonical hash of various data types.

    Parameters
    ----------
    data : DataFrame, ndarray, or bytes
        Input data
    algorithm : str
        Hash algorithm

    Returns
    -------
    hash_hex : str
        Hexadecimal hash digest
    """
    hasher = hashlib.new(algorithm)

    if isinstance(data, pd.DataFrame):
        return compute_dataframe_hash(data, algorithm=algorithm, canonical=True)
    elif isinstance(data, np.ndarray):
        # Ensure C-contiguous for reproducibility
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        hasher.update(data.tobytes())
    elif isinstance(data, bytes):
        hasher.update(data)
    else:
        raise TypeError(f"Unsupported data type for hashing: {type(data)}")

    return hasher.hexdigest()


def get_file_metadata(file_path: Path) -> dict:
    """
    Get metadata about a file.

    Parameters
    ----------
    file_path : Path
        Path to file

    Returns
    -------
    metadata : dict
        Dictionary with size_bytes, row_count (if CSV), and hash
    """
    metadata = {}

    # File size
    metadata["size_bytes"] = file_path.stat().st_size

    # Row count for CSV/TSV
    if file_path.suffix.lower() in [".csv", ".tsv", ".txt"]:
        try:
            df = pd.read_csv(file_path, sep="," if file_path.suffix == ".csv" else "\t")
            metadata["row_count"] = len(df)
            metadata["column_count"] = len(df.columns)
        except Exception as e:
            logger.warning(f"Could not read file to count rows: {e}")
            metadata["row_count"] = None
            metadata["column_count"] = None

    # Compute hash
    metadata["sha256_hash"] = compute_file_hash(file_path, algorithm="sha256")

    return metadata
