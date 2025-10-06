# 📄 PaperPal — Research Paper Summarizer & Insight Extractor

**PaperPal** is an end-to-end NLP system that downloads research papers from **arXiv**, summarizes them using a fine-tuned transformer model, and extracts key insights such as methods and results.  
The project culminates in an interactive **Streamlit UI**, served via **FastAPI**, and exported as a **Hugging Face Space** demo.

---

## 🧠 Tech Stack

- **Core:** Python • Hugging Face Transformers • PyTorch  
- **Retrieval:** FAISS • RAG (Retrieval-Augmented Generation)  
- **Frontend & Serving:** Streamlit • FastAPI  
- **Utilities:** pandas • scikit-learn • dotenv • arxiv API
- **ML Tools:** Weights & Biases • TensorBoard • ROUGE metrics

---

## ✅ Progress

### Day 1: Data Collection & Preparation ✅
- ✅ Repository scaffold with modular structure
- ✅ Configuration system (`.env` + `Config` class)
- ✅ arXiv downloader with rate-limiting
- ✅ Dataset preparation pipeline
- ✅ Train/val/test splits
- ✅ JSONL and Parquet outputs

### Day 2: Fine-Tuning Setup ✅
- ✅ Silver summary generation with BART
- ✅ Multi-task learning (summaries, methods, results)
- ✅ Training pipeline with Hugging Face Trainer
- ✅ Weights & Biases integration
- ✅ CPU-optimized training
- ✅ Evaluation metrics (ROUGE)
- ✅ Model checkpointing and saving
- ✅ Evaluation notebook

### Day 3: RAG Integration (Upcoming)
- ⏳ FAISS index building
- ⏳ Semantic search with sentence-transformers
- ⏳ Context retrieval pipeline
- ⏳ RAG query interface

### Day 4: Deployment & Demo (Upcoming)
- ⏳ Streamlit UI
- ⏳ FastAPI backend
- ⏳ Hugging Face Space deployment
- ⏳ Docker containerization

---

## 🚀 Quickstart

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ahnambia/PaperPal.git
cd PaperPal

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup configuration
cp .env.example .env
# Edit .env to customize categories, keywords, etc.
```

### Day 1: Data Collection

```bash
# Download papers from arXiv
python scripts/download_arxiv.py --max-results 100

# Prepare and clean dataset
python scripts/prepare_dataset.py --min-abs-words 60 --val-size 0.1 --test-size 0.1
```

**Output:**
- Raw data: `data/raw/arxiv_<timestamp>.jsonl`
- Processed data: `data/processed/papers_{train,val,test}.{jsonl,parquet}`

### Day 2: Fine-Tuning

```bash
# Step 1: Generate silver summaries
python scripts/generate_summaries.py --all

# Step 2: Train the model
python scripts/train_model.py

# Step 3: Test the model
python scripts/test_model.py

# Step 4: Evaluate (optional)
# Open and run: notebooks/evaluate_model.ipynb
```

**Output:**
- Model checkpoints: `models/checkpoints/`
- Final model: `models/checkpoints/final_model/`
- Training logs: `logs/`
- W&B dashboard: Check your Weights & Biases project

### Training Configuration

Edit `config/training_config.yaml` to customize:
- Model selection (BART-base, BART-large, T5)
- Training hyperparameters
- Batch size and gradient accumulation
- Evaluation settings
- W&B project name

---

## 📁 Project Structure

```
paperpal/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── config/
│   └── training_config.yaml    # Training configuration
│
├── data/
│   ├── raw/                     # Raw arXiv downloads
│   ├── interim/                 # Intermediate processing
│   └── processed/               # Final train/val/test splits
│
├── models/
│   ├── cache/                   # Hugging Face model cache
│   └── checkpoints/             # Training checkpoints
│
├── logs/                        # TensorBoard logs
│
├── notebooks/
│   └── evaluate_model.ipynb     # Evaluation dashboard
│
├── src/
│   └── paperpal/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       ├── arxiv_client.py      # arXiv API client
│       ├── data_prep.py         # Dataset preparation
│       ├── silver_summaries.py  # Silver label generation
│       ├── model.py             # Model wrapper
│       ├── trainer.py           # Training pipeline
│       └── utils/
│           ├── io.py            # I/O utilities
│           └── text.py          # Text processing
│
└── scripts/
    ├── download_arxiv.py        # Download papers
    ├── prepare_dataset.py       # Prepare datasets
    ├── generate_summaries.py    # Generate silver summaries
    ├── train_model.py           # Train model
    └── test_model.py            # Test model
```

---

## 🎯 Features

### Current (Days 1-2)

✅ **Automated Data Pipeline**
- Query arXiv API with custom filters
- Clean and preprocess abstracts
- Automatic train/val/test splitting

✅ **Silver Summary Generation**
- Zero-shot summarization with BART
- Multi-task: summaries, methods, results
- Quality validation and filtering

✅ **Production-Ready Training**
- Hugging Face Trainer integration
- Mixed precision training (when GPU available)
- Gradient accumulation for larger effective batch sizes
- Early stopping and checkpointing
- Comprehensive logging (W&B + TensorBoard)

✅ **Evaluation & Testing**
- ROUGE score computation
- Interactive testing script
- Jupyter notebook for analysis
- Example predictions and visualizations

### Upcoming (Days 3-4)

🔜 **RAG Integration**
- FAISS vector database
- Semantic paper search
- Context-aware summarization

🔜 **Interactive Demo**
- Streamlit web interface
- FastAPI REST API
- Real-time paper summarization
- Paper recommendations

---

## 📊 Monitoring & Logging

### Weights & Biases

Setup W&B for experiment tracking:

```bash
# Login to W&B
wandb login

# Your runs will appear at: https://wandb.ai/<username>/paperpal
```

Edit `config/training_config.yaml`:
```yaml
wandb:
  project: "paperpal"
  entity: "your-username"
  tags: ["bart-base", "summarization"]
```

### TensorBoard

```bash
# View training logs
tensorboard --logdir logs/
# Open: http://localhost:6006
```

---

## 🎓 Model Details

### Base Model
- **BART-base** (facebook/bart-base)
- 139M parameters
- Pre-trained on CNN/DailyMail summarization

### Fine-Tuning
- **Task:** Abstractive summarization of research papers
- **Dataset:** arXiv papers (cs.CL, cs.LG, cs.AI)
- **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L
- **Training:** 3 epochs, ~2-4 hours on CPU

### Multi-Task Learning
1. **Summarization:** Generate concise abstracts
2. **Method Extraction:** Identify research methods
3. **Result Extraction:** Extract key findings

---

## 🛠️ Advanced Usage

### Custom Model Training

```bash
# Train with different model
python scripts/train_model.py --model facebook/bart-large

# Adjust hyperparameters
python scripts/train_model.py --epochs 5 --batch-size 4 --learning-rate 5e-5

# Resume from checkpoint
python scripts/train_model.py --resume-from-checkpoint ./models/checkpoints/checkpoint-1000
```

### Interactive Testing

```bash
# Interactive mode - paste your own abstracts
python scripts/test_model.py --interactive

# Test single abstract
python scripts/test_model.py --abstract "Your abstract here..."

# Test from file
python scripts/test_model.py --file path/to/abstract.txt
```

### Custom Data Collection

Edit `.env` to customize data collection:
```bash
# Target specific categories
ARXIV_CATEGORIES="cs.CL,cs.LG,cs.AI,stat.ML"

# Add keyword filters
ARXIV_FREE_TEXT="transformer OR attention OR BERT OR GPT"

# Collect more papers
ARXIV_MAX_RESULTS=500
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures (T5, Pegasus)
- Better evaluation metrics
- Web scraping for non-arXiv papers
- Citation graph integration
- Multi-language support

---

## 📝 Citation

If you use PaperPal in your research, please cite:

```bibtex
@software{paperpal2025,
  author = {Your Name},
  title = {PaperPal: AI-Powered Research Paper Summarization},
  year = {2025},
  url = {https://github.com/ahnambia/PaperPal}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- arXiv for providing open access to research papers
- BART authors (Lewis et al., 2019)
- The open-source ML community

---

## 📞 Contact

- GitHub: [@ahnambia](https://github.com/ahnambia)
- Project Link: [https://github.com/ahnambia/PaperPal](https://github.com/ahnambia/PaperPal)

---

## 🗺️ Roadmap

- [x] Day 1: Data Collection & Preparation
- [x] Day 2: Fine-Tuning Setup
- [ ] Day 3: RAG Integration
- [ ] Day 4: Deployment & Demo
- [ ] Future: Multi-modal support (PDFs, images)
- [ ] Future: Citation network analysis
- [ ] Future: Collaborative features

---

**Status:** ✅ Day 2 Complete — Fine-tuning pipeline ready!  
**Next:** Day 3 — RAG Integration 🚀
