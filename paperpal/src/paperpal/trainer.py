"""
Training Module with W&B Integration

This module provides a comprehensive training pipeline with:
- Hugging Face Trainer integration
- Weights & Biases logging
- Custom callbacks for monitoring
- Multi-task learning support
- CPU-optimized settings

Key Features:
- Automatic metric computation (doles out ROUGE scores)
- Early stopping
- Create Model checkpoints
- Scheduling of learning rate 
- Tracking of Progess during run time
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import evaluate

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[Warning] wandb not installed. Install with: pip install wandb")


class PaperPalTrainer:
    """
    Custom trainer for PaperPal fine-tuning.
    
    Wraps Hugging Face Trainer with PaperPal-specific functionality.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: Seq2SeqTrainingArguments,
        compute_metrics_fn: Optional[callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            compute_metrics_fn: Optional custom metrics function
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.training_args = training_args
        
        # Setup data collator
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
        
        # Setup metrics
        self.compute_metrics_fn = (
            compute_metrics_fn or self._default_compute_metrics
        )
        
        # Initialize trainer
        self.trainer = None
        self._setup_trainer()
    
    def _setup_trainer(self):
        """Setup Hugging Face Trainer."""
        # Callbacks
        callbacks = []
        
        # Early stopping
        if hasattr(self.training_args, 'load_best_model_at_end'):
            if self.training_args.load_best_model_at_end:
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=3,
                        early_stopping_threshold=0.001
                    )
                )
        
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics_fn,
            callbacks=callbacks
        )
    
    def _default_compute_metrics(self, eval_pred):
        """
        Default metrics computation (ROUGE scores).
        
        Args:
            eval_pred: Tuple of (predictions, labels)
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )
        
        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )
        
        # Clean up text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        # Load ROUGE metric
        rouge = evaluate.load("rouge")
        
        # Compute ROUGE scores
        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Extract and round scores
        result = {
            key: round(value * 100, 2)
            for key, value in result.items()
        }
        
        return result
    
    def train(self):
        """Run training."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Train
        train_result = self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        
        # Save metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        return train_result
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on eval dataset.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")
        metrics = self.trainer.evaluate()
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics
    
    def predict(self, test_dataset: Dataset) -> Any:
        """
        Generate predictions on test dataset.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Prediction output
        """
        print("\nGenerating predictions...")
        predictions = self.trainer.predict(test_dataset)
        return predictions


def create_training_args(
    config: Dict[str, Any],
    output_dir: str = "./models/checkpoints"
) -> Seq2SeqTrainingArguments:
    """
    Create training arguments from config dictionary.
    
    Args:
        config: Training configuration
        output_dir: Output directory for checkpoints
        
    Returns:
        Seq2SeqTrainingArguments
    """
    training_cfg = config.get("training", {})
    
    # Determine report_to
    report_to = training_cfg.get("report_to", ["tensorboard"])
    if WANDB_AVAILABLE and "wandb" in report_to:
        # Initialize W&B if configured
        wandb_cfg = config.get("wandb", {})
        if wandb_cfg.get("project"):
            try:
                wandb.init(
                    project=wandb_cfg.get("project", "paperpal"),
                    entity=wandb_cfg.get("entity"),
                    name=wandb_cfg.get("name"),
                    tags=wandb_cfg.get("tags", []),
                    notes=wandb_cfg.get("notes", ""),
                    config=config
                )
                print("[W&B] Initialized successfully")
            except Exception as e:
                print(f"[W&B] Failed to initialize: {e}")
                report_to = ["tensorboard"]
    else:
        report_to = ["tensorboard"]
    
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get(
            "per_device_train_batch_size", 2
        ),
        per_device_eval_batch_size=training_cfg.get(
            "per_device_eval_batch_size", 4
        ),
        gradient_accumulation_steps=training_cfg.get(
            "gradient_accumulation_steps", 8
        ),
        learning_rate=training_cfg.get("learning_rate", 3e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_steps=training_cfg.get("warmup_steps", 500),
        
        # Optimization
        fp16=training_cfg.get("fp16", False),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        optim=training_cfg.get("optim", "adamw_torch"),
        
        # Evaluation & Saving
        evaluation_strategy=training_cfg.get("evaluation_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 500),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 500),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=training_cfg.get("metric_for_best_model", "rouge1"),
        greater_is_better=training_cfg.get("greater_is_better", True),
        
        # Logging
        logging_dir=training_cfg.get("logging_dir", "./logs"),
        logging_steps=training_cfg.get("logging_steps", 100),
        report_to=report_to,
        
        # Generation config for evaluation
        predict_with_generate=True,
        generation_max_length=config.get("tokenizer", {}).get(
            "max_target_length", 128
        ),
        generation_num_beams=4,
        
        # Other
        seed=config.get("seed", 42),
        dataloader_num_workers=0,  # macOS compatibility
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    return args


def prepare_dataset_for_training(
    dataset: Dataset,
    tokenizer,
    max_input_length: int = 512,
    max_target_length: int = 128,
    input_column: str = "abstract",
    target_column: str = "summary"
) -> Dataset:
    """
    Prepare dataset for training by tokenizing.
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer
        max_input_length: Max input sequence length
        max_target_length: Max target sequence length
        input_column: Name of input column
        target_column: Name of target column
        
    Returns:
        Tokenized dataset
    """
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples[input_column],
            max_length=max_input_length,
            truncation=True,
            padding=False  # Will be done by data collator
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples[target_column],
            max_length=max_target_length,
            truncation=True,
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def load_datasets_from_jsonl(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None
) -> DatasetDict:
    """
    Load datasets from JSONL files.
    
    Args:
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        test_path: Optional path to test JSONL
        
    Returns:
        DatasetDict with train/val/test splits
    """
    import pandas as pd
    
    datasets = {}
    
    # Load train
    if Path(train_path).exists():
        df_train = pd.read_json(train_path, lines=True)
        datasets["train"] = Dataset.from_pandas(df_train)
        print(f"[Dataset] Loaded train: {len(datasets['train'])} examples")
    
    # Load validation
    if Path(val_path).exists():
        df_val = pd.read_json(val_path, lines=True)
        datasets["validation"] = Dataset.from_pandas(df_val)
        print(f"[Dataset] Loaded validation: {len(datasets['validation'])} examples")
    
    # Load test (optional)
    if test_path and Path(test_path).exists():
        df_test = pd.read_json(test_path, lines=True)
        datasets["test"] = Dataset.from_pandas(df_test)
        print(f"[Dataset] Loaded test: {len(datasets['test'])} examples")
    
    return DatasetDict(datasets)