'''What it does: Tiny I/O helpers—ensures directories exist and writes fast JSONL.
Why it matters: We create/read datasets a lot; consistent, fast I/O keeps the pipeline simple and reliable.'''

from pathlib import Path
import orjson
from typing import Iterable, Dict, Any

def ensure_dirs(*paths: str) -> None:
    """
    Create directories if they don't exist.
    Safe to call multiple times.
    """
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    """
    Write an iterable of dict rows to JSONL using orjson (fast).
    Returns number of rows written.
    """
    n = 0
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")
            n += 1
    return n

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """
    Stream JSONL rows as dicts. Caller can wrap in list(...) if needed.
    """
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)

