"""
Model Loading and Management Module

This module handles:
- Loading pre-trained models and tokenizers
- Model configuration and optimization
- Inference utilities
- Checkpoint management

Supports:
- BART (facebook/bart-base, facebook/bart-large)
- T5 (t5-base, google/flan-t5-base)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    PreTrainedModel
)


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""
    model_name: str = "facebook/bart-base"
    cache_dir: Optional[str] = "./models/cache"
    device: str = "cpu"
    max_input_length: int = 512
    max_target_length: int = 128
    
    # Generation parameters
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    no_repeat_ngram_size: int = 3


class PaperPalModel:
    """
    Wrapper class for PaperPal's summarization models.
    
    Provides a unified interface for loading, inference, and management
    of sequence-to-sequence models.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        load_model: bool = True
    ):
        """
        Initialize the model wrapper.
        
        Args:
            config: Model configuration
            load_model: Whether to load the model immediately
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize as None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model: Optional[PreTrainedModel] = None
        
        if load_model:
            self.load()
    
    def load(self):
        """Load tokenizer and model from Hugging Face."""
        print(f"[PaperPalModel] Loading model: {self.config.model_name}")
        print(f"[PaperPalModel] Device: {self.device}")
        print(f"[PaperPalModel] Cache dir: {self.config.cache_dir}")
        
        # Create cache directory if specified
        if self.config.cache_dir:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[PaperPalModel] ✓ Model loaded successfully")
        print(f"[PaperPalModel] Model size: "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters")
    
    def prepare_input(
        self,
        text: Union[str, List[str]],
        padding: str = "max_length",
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text for model.
        
        Args:
            text: Input text or list of texts
            padding: Padding strategy
            truncation: Whether to truncate
            
        Returns:
            Dictionary of tokenized inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def generate(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for input text(s).
        
        Args:
            text: Input text or list of texts
            max_length: Maximum output length
            min_length: Minimum output length
            num_beams: Number of beams for beam search
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated summaries
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Prepare inputs
        inputs = self.prepare_input(text)
        
        # Set generation parameters
        gen_kwargs = {
            "max_length": max_length or self.config.max_target_length,
            "min_length": min_length or 30,
            "num_beams": num_beams or self.config.num_beams,
            "length_penalty": self.config.length_penalty,
            "early_stopping": self.config.early_stopping,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                **gen_kwargs
            )
        
        # Decode outputs
        summaries = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return [s.strip() for s in summaries]
    
    def summarize(
        self,
        abstract: str,
        **kwargs
    ) -> str:
        """
        Generate a single summary.
        
        Args:
            abstract: Input abstract
            **kwargs: Generation parameters
            
        Returns:
            Generated summary
        """
        summaries = self.generate([abstract], **kwargs)
        return summaries[0] if summaries else ""
    
    def batch_summarize(
        self,
        abstracts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for multiple abstracts in batches.
        
        Args:
            abstracts: List of abstracts
            batch_size: Batch size for processing
            **kwargs: Generation parameters
            
        Returns:
            List of summaries
        """
        all_summaries = []
        
        for i in range(0, len(abstracts), batch_size):
            batch = abstracts[i:i + batch_size]
            summaries = self.generate(batch, **kwargs)
            all_summaries.extend(summaries)
        
        return all_summaries
    
    def save_checkpoint(self, output_dir: str):
        """
        Save model and tokenizer checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Nothing to save.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[PaperPalModel] Saving checkpoint to: {output_dir}")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("[PaperPalModel] ✓ Checkpoint saved")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "cpu",
        **config_kwargs
    ) -> 'PaperPalModel':
        """
        Load model from a saved checkpoint.
        
        Args:
            checkpoint_dir: Directory containing checkpoint
            device: Device to load model on
            **config_kwargs: Additional config parameters
            
        Returns:
            Loaded PaperPalModel instance
        """
        config = ModelConfig(
            model_name=checkpoint_dir,
            device=device,
            **config_kwargs
        )
        
        return cls(config, load_model=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        return {
            "status": "loaded",
            "model_name": self.config.model_name,
            "device": str(self.device),
            "total_parameters": num_params,
            "trainable_parameters": num_trainable,
            "model_size_mb": num_params * 4 / (1024 ** 2),  # Assuming fp32
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else None,
            "max_input_length": self.config.max_input_length,
            "max_target_length": self.config.max_target_length
        }
    
    def __repr__(self) -> str:
        """String representation."""
        info = self.get_model_info()
        if info["status"] == "not_loaded":
            return f"PaperPalModel(status=not_loaded)"
        
        return (
            f"PaperPalModel(\n"
            f"  model={self.config.model_name},\n"
            f"  device={self.device},\n"
            f"  parameters={info['total_parameters']:,}\n"
            f")"
        )


def load_model_from_config(config_dict: Dict[str, Any]) -> PaperPalModel:
    """
    Load model from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary (typically from YAML)
        
    Returns:
        Initialized PaperPalModel
    """
    model_cfg = config_dict.get("model", {})
    tokenizer_cfg = config_dict.get("tokenizer", {})
    
    config = ModelConfig(
        model_name=model_cfg.get("name", "facebook/bart-base"),
        cache_dir=model_cfg.get("cache_dir", "./models/cache"),
        device="cpu",  # Will be set by trainer
        max_input_length=tokenizer_cfg.get("max_input_length", 512),
        max_target_length=tokenizer_cfg.get("max_target_length", 128)
    )
    
    return PaperPalModel(config)