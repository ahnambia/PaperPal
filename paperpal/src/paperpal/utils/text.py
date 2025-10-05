'''What it does: Minimal text utilities to clean abstracts (strip simple LaTeX/citations) and count words.
Why it matters: Clean input improves fine-tuning quality; word count helps filter out trivial abstracts.'''

import re
from typing import List


def clean_abstract(text: str) -> str:
    """
    Light cleanup: collapse whitespace, remove simple LaTeX citations and inline math.
    We'll extend this later if needed.
    """
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text)
    t = re.sub(r"\\cite\{[^}]+\}", "", t)  # remove \cite{...}
    t = re.sub(r"\$[^$]+\$", "", t)        # remove inline math $...$
    return t.strip()

def word_count(text: str) -> int:
    """
    Quick tokenless word counter for filtering.
    """
    return len(re.findall(r"\w+", text or ""))

