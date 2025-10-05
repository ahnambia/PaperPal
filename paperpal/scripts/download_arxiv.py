'''What it does:

Loads your Config

Builds the query from .env values

Fetches results via arxiv_client

Writes a timestamped JSONL to data/raw/

Why it matters:
Creates a reproducible “raw snapshot” you can re-run any time. It is the single source for downstream dataset prep.
'''

from pathlib import Path
from datetime import datetime
import argparse

from src.paperpal.config import Config
from src.paperpal.utils.io import ensure_dirs, write_jsonl
from src.paperpal.arxiv_client import build_query, search_arxiv

def parse_args():
    p = argparse.ArgumentParser(description="Download arXiv metadata → data/raw/*.jsonl")
    p.add_argument("--query", type=str, default=None, help="Override arXiv API query")
    p.add_argument("--max-results", type=int, default=None, help="Override max results")
    return p.parse_args()

def main():
    cfg = Config()
    ensure_dirs(cfg.raw_dir, cfg.interim_dir, cfg.processed_dir)

    args = parse_args()
    query = args.query or build_query(cfg.categories, cfg.free_text)
    max_results = args.max_results or cfg.max_results

    if not query:
        raise SystemExit("Empty query: set ARXIV_CATEGORIES and/or ARXIV_FREE_TEXT in .env or use --query")

    print(f"[paperpal] Query: {query}")
    print(f"[paperpal] Max results: {max_results}")

    papers = search_arxiv(query=query, max_results=max_results)
    rows = [p.as_dict() for p in papers]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(cfg.raw_dir) / f"arxiv_{ts}.jsonl"

    n = write_jsonl(str(out_path), rows)
    print(f"[paperpal] Wrote {n} records → {out_path}")

if __name__ == "__main__":
    main()