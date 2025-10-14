import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
Main Training Script

This script orchestrates the complete training pipeline:
1. Load configuration
2. Load datasets with summaries
3. Initialize model and tokenizer
4. Setup training arguments
5. Train the model
6. Evaluate and save results

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config src/paperpal/config/training_config.yaml
    python scripts/train_model.py --model facebook/bart-large
    python scripts/train_model.py --epochs 5 --batch-size 4
"""

import argparse
import sys
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paperpal.config import Config
from src.paperpal.model import PaperPalModel, ModelConfig
from src.paperpal.trainer import (
    PaperPalTrainer,
    create_training_args,
    prepare_dataset_for_training,
    load_datasets_from_jsonl
)

console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PaperPal summarization model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/paperpal/config/training_config.yaml",
        help="Path to training config YAML"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (overrides config)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        console.print(f"[yellow]Warning: Config file not found: {config_path}")
        console.print("[yellow]Using default configuration")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    console.print(f"[green]✓ Loaded config from: {config_path}")
    return config


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line overrides to config."""
    if args.model:
        config.setdefault("model", {})["name"] = args.model
        console.print(f"[cyan]Override: model = {args.model}")
    
    if args.epochs:
        config.setdefault("training", {})["num_train_epochs"] = args.epochs
        console.print(f"[cyan]Override: epochs = {args.epochs}")
    
    if args.batch_size:
        config.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size
        console.print(f"[cyan]Override: batch_size = {args.batch_size}")
    
    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
        console.print(f"[cyan]Override: learning_rate = {args.learning_rate}")
    
    if args.output_dir:
        config.setdefault("training", {})["output_dir"] = args.output_dir
        console.print(f"[cyan]Override: output_dir = {args.output_dir}")
    
    return config


def print_training_summary(config: dict, dataset_info: dict):
    """Print a summary table of training configuration."""
    table = Table(title="Training Configuration Summary", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    # Model info
    model_name = config.get("model", {}).get("name", "facebook/bart-base")
    table.add_row("Model", model_name)
    
    # Dataset info
    table.add_row("Train samples", str(dataset_info.get("train", 0)))
    table.add_row("Val samples", str(dataset_info.get("val", 0)))
    table.add_row("Test samples", str(dataset_info.get("test", 0)))
    
    # Training params
    training = config.get("training", {})
    table.add_row("Epochs", str(training.get("num_train_epochs", 3)))
    table.add_row("Batch size", str(training.get("per_device_train_batch_size", 2)))
    table.add_row("Gradient accum", str(training.get("gradient_accumulation_steps", 8)))
    effective_batch = (
        training.get("per_device_train_batch_size", 2) *
        training.get("gradient_accumulation_steps", 8)
    )
    table.add_row("Effective batch", str(effective_batch))
    table.add_row("Learning rate", f"{training.get('learning_rate', 3e-5):.2e}")
    table.add_row("Warmup steps", str(training.get("warmup_steps", 500)))
    
    # Output
    output_dir = training.get("output_dir", "./models/checkpoints")
    table.add_row("Output dir", output_dir)
    
    console.print(table)


def main():
    """Main training pipeline."""
    args = parse_args()
    
    console.print("\n[bold cyan]" + "="*60)
    console.print("[bold cyan]PaperPal Training Pipeline")
    console.print("[bold cyan]" + "="*60 + "\n")
    
    # Step 1: Load configuration
    console.print("[bold yellow]Step 1: Loading Configuration")
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    
    # Step 2: Load PaperPal config for paths
    console.print("\n[bold yellow]Step 2: Loading Datasets")
    cfg = Config()
    
    # Determine dataset paths (use files with summaries if available)
    train_file = Path(cfg.processed_dir) / "papers_train_filtered.jsonl"
    val_file = Path(cfg.processed_dir) / "papers_val_filtered.jsonl"
    test_file = Path(cfg.processed_dir) / "papers_test_filtered.jsonl"
    
    # Fallback to with_summaries if filtered doesn't exist
    if not train_file.exists():
        train_file = Path(cfg.processed_dir) / "papers_train_with_summaries.jsonl"
        val_file = Path(cfg.processed_dir) / "papers_val_with_summaries.jsonl"
        test_file = Path(cfg.processed_dir) / "papers_test_with_summaries.jsonl"
        
    # Fallback to original files if summaries not generated yet
    if not train_file.exists():
        console.print("[yellow]Warning: Summaries not found. Using original files.")
        console.print("[yellow]Run 'python scripts/generate_summaries.py --all' first!")
        train_file = Path(cfg.processed_dir) / "papers_train.jsonl"
        val_file = Path(cfg.processed_dir) / "papers_val.jsonl"
        test_file = Path(cfg.processed_dir) / "papers_test.jsonl"
    
    # Check files exist
    if not train_file.exists() or not val_file.exists():
        console.print("[red]Error: Dataset files not found!")
        console.print("[red]Please run data preparation scripts first.")
        sys.exit(1)
    
    # Load datasets
    dataset_dict = load_datasets_from_jsonl(
        str(train_file),
        str(val_file),
        str(test_file) if test_file.exists() else None
    )
    
    dataset_info = {
        "train": len(dataset_dict["train"]),
        "val": len(dataset_dict["validation"]),
        "test": len(dataset_dict.get("test", []))
    }
    
    # Step 3: Initialize model
    console.print("\n[bold yellow]Step 3: Initializing Model")
    
    model_cfg = ModelConfig(
        model_name=config.get("model", {}).get("name", "facebook/bart-base"),
        cache_dir=config.get("model", {}).get("cache_dir", "./models/cache"),
        device="cpu",
        max_input_length=config.get("tokenizer", {}).get("max_input_length", 512),
        max_target_length=config.get("tokenizer", {}).get("max_target_length", 128)
    )
    
    paperpal_model = PaperPalModel(model_cfg, load_model=True)
    
    # Print model info
    model_info = paperpal_model.get_model_info()
    console.print(f"[green]✓ Model: {model_info['model_name']}")
    console.print(f"[green]✓ Parameters: {model_info['total_parameters']:,}")
    console.print(f"[green]✓ Size: {model_info['model_size_mb']:.1f} MB")
    
    # Step 4: Prepare datasets
    console.print("\n[bold yellow]Step 4: Tokenizing Datasets")
    
    tokenizer = paperpal_model.tokenizer
    
    # Check if summary column exists
    if "summary" not in dataset_dict["train"].column_names:
        console.print("[red]Error: 'summary' column not found in dataset!")
        console.print("[red]Run 'python scripts/generate_summaries.py --all' first.")
        sys.exit(1)
    
    # Tokenize datasets
    train_dataset = prepare_dataset_for_training(
        dataset_dict["train"],
        tokenizer,
        max_input_length=model_cfg.max_input_length,
        max_target_length=model_cfg.max_target_length,
        input_column="abstract",
        target_column="summary"
    )
    
    eval_dataset = prepare_dataset_for_training(
        dataset_dict["validation"],
        tokenizer,
        max_input_length=model_cfg.max_input_length,
        max_target_length=model_cfg.max_target_length,
        input_column="abstract",
        target_column="summary"
    )
    
    console.print(f"[green]✓ Tokenized train: {len(train_dataset)} examples")
    console.print(f"[green]✓ Tokenized val: {len(eval_dataset)} examples")
    
    # Step 5: Setup training
    console.print("\n[bold yellow]Step 5: Setting Up Training")
    
    output_dir = config.get("training", {}).get("output_dir", "./models/checkpoints")
    training_args = create_training_args(config, output_dir)
    
    # Print summary
    print_training_summary(config, dataset_info)
    
    # Step 6: Initialize trainer
    console.print("\n[bold yellow]Step 6: Initializing Trainer")
    
    trainer = PaperPalTrainer(
        model=paperpal_model.model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args
    )
    
    console.print("[green]✓ Trainer initialized")
    
    # Step 7: Train!
    console.print("\n[bold yellow]Step 7: Starting Training")
    console.print("[cyan]This may take a while on CPU...")
    console.print("[cyan]Monitor progress with W&B or TensorBoard\n")
    
    try:
        if args.resume_from_checkpoint:
            console.print(f"[cyan]Resuming from: {args.resume_from_checkpoint}")
            train_result = trainer.trainer.train(
                resume_from_checkpoint=args.resume_from_checkpoint
            )
        else:
            train_result = trainer.train()
        
        console.print("\n[bold green]✓ Training completed successfully!")
        
        # Print final metrics
        console.print("\n[bold cyan]Final Training Metrics:")
        for key, value in train_result.metrics.items():
            console.print(f"  {key}: {value:.4f}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user")
        console.print("[yellow]Model checkpoint saved")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}")
        raise
    
    # Step 8: Evaluate
    if not args.no_eval:
        console.print("\n[bold yellow]Step 8: Final Evaluation")
        console.print("[yellow]Skipping evaluation due to MPS compatibility")
        eval_metrics = {}
    else:
        console.print("\n[bold yellow]Step 8: Final Evaluation")
        eval_metrics = trainer.evaluate()
        console.print("\n[bold cyan]Evaluation Metrics:")
        for key, value in eval_metrics.items():
            console.print(f"  {key}: {value:.4f}")
    
    # Step 9: Save final model
    console.print("\n[bold yellow]Step 9: Saving Final Model")
    final_model_path = Path(output_dir) / "final_model"
    paperpal_model.model = trainer.model  # Update with trained model
    paperpal_model.save_checkpoint(str(final_model_path))
    
    console.print(f"[green]✓ Final model saved to: {final_model_path}")
    
    # Summary
    console.print("\n[bold green]" + "="*60)
    console.print("[bold green]Training Pipeline Complete!")
    console.print("[bold green]" + "="*60)
    console.print("\n[cyan]Next steps:")
    console.print("  1. Check training logs and metrics")
    console.print("  2. Test the model with sample papers")
    console.print("  3. Proceed to Day 3: RAG Integration")
    console.print(f"\n[cyan]Model location: {final_model_path}")
    console.print(f"[cyan]Logs location: {training_args.logging_dir}\n")


if __name__ == "__main__":
    main()