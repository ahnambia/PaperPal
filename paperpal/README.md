# 📄 PaperPal — Research Paper Summarizer & Insight Extractor (HF + PyTorch)

**PaperPal** is an end-to-end NLP system that downloads research papers from **arXiv**, summarizes them using a fine-tuned transformer model, and extracts key insights such as methods and results.  
The project culminates in an interactive **Streamlit UI**, served via **FastAPI**, and exported as a **Hugging Face Space** demo.

---

## 🧠 Tech Stack
- **Core:** Python • Hugging Face Transformers • PyTorch  
- **Retrieval:** FAISS • RAG (Retrieval-Augmented Generation)  
- **Frontend & Serving:** Streamlit • FastAPI  
- **Utilities:** pandas • scikit-learn • dotenv • arxiv API

---

## ✅ Progress (Completed – Day 1)

### 📦 Repository Scaffold
- Organized `src/`, `scripts/`, and `data/` directories for clean modular development.  
- Added configuration system (`Config` class + `.env`) to manage categories, keywords, and dataset splits.

### 🔍 arXiv Downloader
- Built `arxiv_client.py` for reliable API queries with paging and polite rate-limiting.  
- Script `scripts/download_arxiv.py` saves timestamped raw metadata → `data/raw/arxiv_<timestamp>.jsonl`.

### 🧹 Dataset Preparation
- Implemented text cleaning (remove LaTeX/citations).  
- Added filtering (min abstract length + date range support).  
- Deduplicated papers by title and generated **train / val / test** splits.  
- Outputs both `.jsonl` and `.parquet` files under `data/processed/`.

---

## 🚀 Quickstart (So Far)

```bash
# 1. Create & activate environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env  # edit categories or limits if desired

# 4. Download and prepare data
python scripts/download_arxiv.py --max-results 200
python scripts/prepare_dataset.py --min-abs-words 60 --val-size 0.1 --test-size 0.1

Outputs

Raw → data/raw/arxiv_<timestamp>.jsonl

Processed → data/processed/papers_{train,val,test}.{jsonl,parquet}

paperpal/
├─ README.md
├─ .gitignore
├─ .env.example
├─ requirements.txt
├─ data/
│  ├─ raw/         # arXiv JSONL snapshots (ignored by git)
│  ├─ interim/     # intermediate files (ignored)
│  └─ processed/   # cleaned splits (ignored)
├─ src/
│  └─ paperpal/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ arxiv_client.py
│     ├─ data_prep.py
│     └─ utils/
│        ├─ io.py
│        └─ text.py
└─ scripts/
   ├─ download_arxiv.py
   └─ prepare_dataset.py

Next Steps (Planned)
Day	Focus	Key Goals
Day 2	Fine-tuning Prep	Generate silver summaries, load tokenizer, and configure BART/T5 training
Day 3	RAG Integration	Build FAISS index + context retrieval pipeline
Day 4	Deployment & Demo	Streamlit UI + FastAPI API + Hugging Face Space export

Status: ✅ Day 1 Complete — Raw Downloader + Dataset Prep in place.
Next → Day 2 Fine-Tuning Setup 🚀