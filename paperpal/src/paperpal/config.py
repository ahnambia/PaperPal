'''What it does: Central place to read settings from environment variables (.env) and define canonical data paths.
Why it matters: Keeps configuration out of code so you can tweak categories, dataset sizes, etc., without editing scripts.'''

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    """
    Project-wide configuration.
    Reads environment variables and exposes strongly-typed fields.
    """
    # Paths
    data_dir: str = os.getenv("DATA_DIR", "data")
    raw_dir: str = os.path.join(data_dir, "raw")
    interim_dir: str = os.path.join(data_dir, "interim")
    processed_dir: str = os.path.join(data_dir, "processed")

    # arXiv search parameters
    categories: str = os.getenv("ARXIV_CATEGORIES", "cs.CL,cs.LG,stat.ML")
    free_text: str = os.getenv("ARXIV_FREE_TEXT", "")
    max_results: int = int(os.getenv("ARXIV_MAX_RESULTS", 200))
    date_range: str = os.getenv("ARXIV_DATE_RANGE", "")  # NEW: "YYYY-MM-DD..YYYY-MM-DD"

    # Dataset settings
    val_size: float = float(os.getenv("VAL_SIZE", 0.1))
    test_size: float = float(os.getenv("TEST_SIZE", 0.1))
    random_seed: int = int(os.getenv("RANDOM_SEED", 42))
    min_abs_words: int = int(os.getenv("MIN_ABS_WORDS", 60))