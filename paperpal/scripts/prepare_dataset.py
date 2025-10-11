'''What it does: One-shot CLI to run the dataset builder with optional overrides.
Why it matters: Repeatable, parameterized prep—easy to re-run with different thresholds/splits.
'''

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paperpal.config import Config
from src.paperpal.data_prep import prepare_dataset

def parse_args():
    p = argparse.ArgumentParser(description="Clean/filter raw arXiv JSONL and create train/val/test splits.")
    p.add_argument("--min-abs-words", type=int, default=None, help="Minimum abstract length in words")
    p.add_argument("--val-size", type=float, default=None, help="Validation fraction (0..1)")
    p.add_argument("--test-size", type=float, default=None, help="Test fraction (0..1)")
    p.add_argument("--date-range", type=str, default=None, help="YYYY-MM-DD..YYYY-MM-DD (inclusive)")
    return p.parse_args()

def main():
    cfg = Config()
    args = parse_args()

    if args.min_abs_words is not None:
        cfg.min_abs_words = args.min_abs_words
    if args.val_size is not None:
        cfg.val_size = args.val_size
    if args.test_size is not None:
        cfg.test_size = args.test_size
    if args.date_range is not None:
        cfg.date_range = args.date_range

    stats = prepare_dataset(cfg)
    print("[paperpal] dataset stats:", stats)

if __name__ == "__main__":
    main()

