''' What it does:

Builds an arXiv query from categories + optional keywords

Calls the arXiv API with polite paging/rate limiting

Returns clean Python dicts (id/title/abstract/authors/categories/published/pdf_url)

Why it matters:
This module isolates all API interaction. Keeping it separate lets you test/download without touching data-prep or model code.
'''

# src/paperpal/arxiv_client.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterable
import arxiv

@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published: str  # ISO8601
    pdf_url: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published,
            "pdf_url": self.pdf_url,
        }

def build_query(categories_csv: str, free_text: str = "") -> str:
    """
    Compose an arXiv API query. Example:
      categories_csv="cs.CL,cs.LG" and free_text="transformer OR BART"
      -> "(cat:cs.CL OR cat:cs.LG) AND (transformer OR BART)"
    """
    cats = [c.strip() for c in categories_csv.split(",") if c.strip()]
    cat_clause = " OR ".join([f"cat:{c}" for c in cats]) if cats else ""
    terms = []
    if cat_clause:
        terms.append(f"({cat_clause})")
    if free_text:
        terms.append(f"({free_text})")
    return " AND ".join(terms) if terms else ""

def _iter_results(query: str, max_results: int) -> Iterable[arxiv.Result]:
    """
    Generator over arXiv results, obeying client paging. arxiv.Client
    handles politeness (delay_seconds between requests).
    """
    client = arxiv.Client(page_size=100, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    for res in client.results(search):
        yield res

def search_arxiv(query: str, max_results: int = 200) -> List[ArxivPaper]:
    results: List[ArxivPaper] = []
    for r in _iter_results(query, max_results):
        # Normalize
        published_dt = r.published
        if isinstance(published_dt, datetime):
            published_iso = published_dt.replace(tzinfo=None).isoformat()
        else:
            published_iso = str(published_dt)

        results.append(
            ArxivPaper(
                arxiv_id=r.get_short_id(),
                title=(r.title or "").strip(),
                abstract=(r.summary or "").strip(),
                authors=[a.name for a in r.authors or []],
                categories=list(r.categories or []),
                published=published_iso,
                pdf_url=r.pdf_url,
            )
        )
    return results
