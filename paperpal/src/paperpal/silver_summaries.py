"""
Silver Summary Generation Module

This module generates high-quality pseudo-summaries using a pre-trained BART model
in zero-shot mode. These "silver" summaries serve as training targets for fine-tuning.

Key Features:
- Zero-shot summarization with BART-base
- Multi-task: summaries, methods, results extraction
- Quality filtering and validation
- Batch processing for efficiency
- Progress tracking with tqdm

Why Silver Summaries?
- Better quality than extractive (first N sentences)
- Creates self-distillation effect (BART learns from BART)
- Scalable to any dataset size
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from tqdm import tqdm

from .utils.text import word_count


@dataclass
class SilverSummaryConfig:
    """Configuration for silver summary generation."""
    model_name: str = "facebook/bart-base"
    max_length: int = 128
    min_length: int = 30
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3
    batch_size: int = 4
    device: str = "cpu"  # or "cuda" if available
    
    # Quality thresholds
    min_summary_words: int = 10
    max_summary_words: int = 100


class SilverSummaryGenerator:
    """
    Generates silver summaries using pre-trained models.
    
    This class handles:
    1. Model loading and initialization
    2. Batch processing of abstracts
    3. Multi-task generation (summary, methods, results)
    4. Quality validation and filtering
    """
    
    def __init__(self, config: SilverSummaryConfig):
        """
        Initialize the generator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"[SilverSummaryGenerator] Loading model: {config.model_name}")
        print(f"[SilverSummaryGenerator] Device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        print("[SilverSummaryGenerator] Model loaded successfully")
    
    def _create_summarization_pipeline(self) -> pipeline:
        """Create a summarization pipeline."""
        return pipeline(
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device.type == "cuda" else -1,
            batch_size=self.config.batch_size
        )
    
    def _validate_summary(self, summary: str) -> bool:
        """
        Validate that a summary meets quality criteria.
        
        Args:
            summary: Generated summary text
            
        Returns:
            True if summary is valid, False otherwise
        """
        if not summary or not summary.strip():
            return False
        
        words = word_count(summary)
        
        # Check word count boundaries
        if words < self.config.min_summary_words:
            return False
        if words > self.config.max_summary_words:
            return False
        
        # Check for placeholder text
        placeholder_phrases = [
            "unable to generate",
            "error occurred",
            "could not summarize"
        ]
        summary_lower = summary.lower()
        if any(phrase in summary_lower for phrase in placeholder_phrases):
            return False
        
        return True
    
    def generate_summary(self, abstract: str) -> Optional[str]:
        """
        Generate a single summary from an abstract.
        
        Args:
            abstract: Input abstract text
            
        Returns:
            Generated summary or None if generation fails
        """
        if not abstract or len(abstract.strip()) < 50:
            return None
        
        try:
            # Prepare input
            inputs = self.tokenizer(
                abstract,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=self.config.max_length,
                    min_length=self.config.min_length,
                    num_beams=self.config.num_beams,
                    length_penalty=self.config.length_penalty,
                    early_stopping=self.config.early_stopping,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Validate and return
            if self._validate_summary(summary):
                return summary.strip()
            return None
            
        except Exception as e:
            print(f"[Error] Failed to generate summary: {e}")
            return None
    
    def extract_methods(self, abstract: str) -> Optional[str]:
        """
        Extract methodology information from abstract.
        
        Uses prompting to guide the model toward method extraction.
        
        Args:
            abstract: Input abstract text
            
        Returns:
            Extracted methods or None
        """
        # Add prefix to guide generation
        prompt = f"Extract the research methods used: {abstract}"
        return self._generate_with_prompt(prompt, max_length=100)
    
    def extract_results(self, abstract: str) -> Optional[str]:
        """
        Extract key results from abstract.
        
        Args:
            abstract: Input abstract text
            
        Returns:
            Extracted results or None
        """
        prompt = f"Summarize the main findings and results: {abstract}"
        return self._generate_with_prompt(prompt, max_length=100)
    
    def _generate_with_prompt(
        self,
        prompt: str,
        max_length: int = 100
    ) -> Optional[str]:
        """
        Generate text with a custom prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum output length
            
        Returns:
            Generated text or None
        """
        try:
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=20,
                    num_beams=3,
                    length_penalty=1.5,
                    early_stopping=True
                )
            
            output = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return output.strip() if output.strip() else None
            
        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            return None
    
    def generate_batch(
        self,
        abstracts: List[str],
        include_methods: bool = True,
        include_results: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate summaries for a batch of abstracts.
        
        Args:
            abstracts: List of abstract texts
            include_methods: Whether to extract methods
            include_results: Whether to extract results
            
        Returns:
            List of dictionaries with generated content
        """
        results = []
        
        for abstract in tqdm(abstracts, desc="Generating silver summaries"):
            item = {"abstract": abstract}
            
            # Generate summary
            summary = self.generate_summary(abstract)
            item["summary"] = summary
            
            # Generate methods
            if include_methods and summary:  # Only if summary succeeded
                methods = self.extract_methods(abstract)
                item["methods"] = methods
            
            # Generate results
            if include_results and summary:
                results_text = self.extract_results(abstract)
                item["results"] = results_text
            
            results.append(item)
        
        return results
    
    def generate_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics on generated summaries.
        
        Args:
            results: List of generation results
            
        Returns:
            Dictionary of statistics
        """
        total = len(results)
        
        # Count successes
        valid_summaries = sum(1 for r in results if r.get("summary"))
        valid_methods = sum(1 for r in results if r.get("methods"))
        valid_results = sum(1 for r in results if r.get("results"))
        
        # Calculate average lengths
        summary_lengths = [
            word_count(r["summary"]) 
            for r in results 
            if r.get("summary")
        ]
        avg_summary_length = (
            sum(summary_lengths) / len(summary_lengths) 
            if summary_lengths else 0
        )
        
        stats = {
            "total_papers": total,
            "valid_summaries": valid_summaries,
            "valid_methods": valid_methods,
            "valid_results": valid_results,
            "summary_success_rate": valid_summaries / total if total > 0 else 0,
            "avg_summary_length": avg_summary_length
        }
        
        return stats


def extractive_summary(abstract: str, num_sentences: int = 3) -> str:
    """
    Fallback: Generate extractive summary (first N sentences).
    
    This is a simple baseline used when model-based generation fails.
    
    Args:
        abstract: Input abstract
        num_sentences: Number of sentences to extract
        
    Returns:
        Extractive summary
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', abstract)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Take first N sentences
    selected = sentences[:num_sentences]
    return '. '.join(selected) + '.' if selected else ""