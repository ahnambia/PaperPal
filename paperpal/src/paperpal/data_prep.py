'''
What it does: Converts raw → pandas DataFrame, cleans abstracts, filters by word count/date, de-duplicates, splits into train/val/test, writes outputs.

Why it matters: This is the canonical dataset builder you'll reuse across experiments.
'''
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Iterable, Dict, Any, Optional
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .utils.text import clean_abstract, word_count
from .utils.io import read_jsonl

DATE_FMT = "%Y-%m-%d"

def _parse_date(s: str) -> Optional[datetime]:
    """Parse arXiv date strings."""
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            # some arXiv dates come as 'YYYY-MM-DD HH:MM:SS'
            return datetime.strptime(s.split(" ")[0], DATE_FMT)
        except Exception:
            return None

def _apply_date_range(df: pd.DataFrame, date_range: str) -> pd.DataFrame:
    """
    date_range like 'YYYY-MM-DD..YYYY-MM-DD' (both inclusive).
    """
    if not date_range:
        return df
    try:
        left, right = [p.strip() for p in date_range.split("..", 1)]
    except ValueError:
        return df  # silently skip malformed range

    start = datetime.strptime(left, DATE_FMT) if left else None
    end = datetime.strptime(right, DATE_FMT) if right else None

    def in_range(d: Optional[datetime]) -> bool:
        """Return True if d is in [start, end]."""
        if d is None:
            return False
        if start and d < start:
            return False
        if end and d > end:
            return False
        return True

    df["_published_dt"] = df["published"].apply(_parse_date)
    df = df[df["_published_dt"].apply(in_range)].copy()
    df.drop(columns=["_published_dt"], inplace=True)
    return df

def _latest_raw_file(raw_dir: str) -> str:
    """Find the latest raw file in the raw directory."""
    candidates = sorted(glob(str(Path(raw_dir) / "arxiv_*.jsonl")))
    if not candidates:
        # allow smoke files from step 3
        candidates = sorted(glob(str(Path(raw_dir) / "_smoke_*.jsonl")))
    if not candidates:
        raise SystemExit("No raw JSONL files found in data/raw/. Run the downloader first.")
    return candidates[-1]

def build_dataframe(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Build a pandas DataFrame from raw arXiv metadata."""
    df = pd.DataFrame(list(rows))
    # Expect keys: arxiv_id, title, abstract, authors, categories, published, pdf_url
    df["abstract"] = df["abstract"].fillna("").astype(str).apply(clean_abstract)
    df["abs_words"] = df["abstract"].apply(word_count)
    # Keep core columns in a consistent order
    keep = ["arxiv_id", "title", "abstract", "abs_words", "authors", "categories", "published", "pdf_url"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.rename(columns={"arxiv_id": "id"})
    # Drop empties
    df = df[(df["title"].astype(str).str.strip() != "") & (df["abstract"].astype(str).str.strip() != "")]
    return df

def split_dataframe(df: pd.DataFrame, test_size: float, val_size: float, seed: int
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train, val, and test sets.
    
    Args:
        df: DataFrame to split
        test_size: Fraction of data to use for testing
        val_size: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    # Handle edge case: if dataset is too small, put everything in train
    n = len(df)
    if n < 3 or test_size + val_size == 0:
        return df, pd.DataFrame(), pd.DataFrame()
    
    # Calculate minimum samples needed for split
    min_test_val = int(n * (test_size + val_size))
    if min_test_val < 1:
        return df, pd.DataFrame(), pd.DataFrame()
    
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=seed, shuffle=True)
    
    # If temp_df has only 1 sample, skip further split
    if len(temp_df) < 2:
        return train_df, pd.DataFrame(), temp_df
    
    rel_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=1 - rel_val, random_state=seed, shuffle=True)
    return train_df, val_df, test_df

def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> None:
    """Save train, val, and test splits to JSONL and Parquet files.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        out_dir: Output directory
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        if d is None or d.empty:
            continue
        d.to_json(f"{out_dir}/papers_{name}.jsonl", orient="records", lines=True, force_ascii=False)
        try:
            d.to_parquet(f"{out_dir}/papers_{name}.parquet")
        except Exception:
            pass

def prepare_dataset(cfg: Config) -> Dict[str, int]:
    """Prepare the dataset by downloading, filtering, and splitting.
    
    Args:
        cfg: Configuration object
    
    Returns:
        Dictionary of dataset statistics
    """

    raw_path = _latest_raw_file(cfg.raw_dir)
    rows = read_jsonl(raw_path)
    df = build_dataframe(rows)

    before = len(df)
    # Optional date filter
    df = _apply_date_range(df, cfg.date_range)
    # Minimum abstract length
    df = df[df["abs_words"] >= cfg.min_abs_words].copy()
    # Deduplicate by title (keep latest by published)
    df = df.sort_values("published").drop_duplicates(subset=["title"], keep="last")

    train_df, val_df, test_df = split_dataframe(df, test_size=cfg.test_size, val_size=cfg.val_size, seed=cfg.random_seed)
    save_splits(train_df, val_df, test_df, cfg.processed_dir)

    stats = {"raw": before, "after_filters": len(df), "train": len(train_df), "val": len(val_df), "test": len(test_df)}
    return stats