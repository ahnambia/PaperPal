"""Direct HTTP-based arXiv downloader (fallback method)"""

import requests
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from datetime import datetime
import time

def fetch_arxiv_papers(category='cs.CL', max_results=100):
    """Fetch papers directly via arXiv HTTP API"""
    
    base_url = 'http://export.arxiv.org/api/query'
    query = f'cat:{category}'
    
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    print(f"Fetching from arXiv API...")
    print(f"Query: {query}")
    print(f"Max results: {max_results}")
    
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return []
    
    print(f"✓ Got response ({len(response.content)} bytes)")
    
    # Parse XML
    root = ET.fromstring(response.content)
    
    # Namespace for arXiv API
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    papers = []
    entries = root.findall('atom:entry', ns)
    
    print(f"Found {len(entries)} entries")
    
    for entry in entries:
        # Extract data
        arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        published = entry.find('atom:published', ns).text
        
        # Authors
        authors = [
            author.find('atom:name', ns).text 
            for author in entry.findall('atom:author', ns)
        ]
        
        # Categories
        categories = [
            cat.attrib['term'] 
            for cat in entry.findall('atom:category', ns)
        ]
        
        # PDF URL
        pdf_link = None
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf':
                pdf_link = link.attrib['href']
                break
        
        paper = {
            'arxiv_id': arxiv_id,
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'categories': categories,
            'published': published,
            'pdf_url': pdf_link
        }
        
        papers.append(paper)
        print(f"  ✓ {arxiv_id}: {title[:60]}...")
    
    return papers


def main():
    # Create output directory
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # Fetch papers
    papers = fetch_arxiv_papers(category='cs.CL', max_results=100)
    
    if not papers:
        print("\n❌ No papers fetched!")
        return
    
    # Save to JSONL
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data/raw/arxiv_{ts}.jsonl'
    
    with open(output_file, 'w') as f:
        for paper in papers:
            f.write(json.dumps(paper) + '\n')
    
    print(f"\n✓ Saved {len(papers)} papers to {output_file}")


if __name__ == '__main__':
    main()