"""
Generate Silver Summaries Script

This script processes the prepared dataset and generates silver summaries
for training. It supports multi-task generation and quality validation.

Usage:
    python scripts/generate_summaries.py --split train
    python scripts/generate_summaries.py --split val --no-methods --no-results
    python scripts/generate_summaries.py --all
    
Output:
    Enriched JSONL files with summary, methods, and results fields
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import track

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paperpal.config import Config
from src.paperpal.silver_summaries import (
    SilverSummaryGenerator,
    SilverSummaryConfig,
    extractive_summary
)
from src.paperpal.utils.io import write_jsonl


console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate silver summaries for training"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default=None,
        help="Which split to process (train/val/test)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all splits"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-base",
        help="Model to use for generation"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation"
    )
    
    parser.add_argument(
        "--no-methods",
        action="store_true",
        help="Skip method extraction"
    )
    
    parser.add_argument(
        "--no-results",
        action="store_true",
        help="Skip result extraction"
    )
    
    parser.add_argument(
        "--fallback-extractive",
        action="store_true",
        help="Use extractive summaries as fallback"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (cpu/cuda)"
    )
    
    return parser.parse_args()


def load_training_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_dataset(split: str, cfg: Config) -> pd.DataFrame:
    """
    Load a dataset split.
    
    Args:
        split: Split name (train/val/test)
        cfg: Config object
        
    Returns:
        DataFrame with papers
    """
    file_path = Path(cfg.processed_dir) / f"papers_{split}.jsonl"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Run scripts/prepare_dataset.py first!"
        )
    
    console.print(f"[cyan]Loading {split} split from: {file_path}")
    df = pd.read_json(file_path, lines=True)
    console.print(f"[green]✓ Loaded {len(df)} papers")
    
    return df


def process_split(
    split: str,
    generator: SilverSummaryGenerator,
    cfg: Config,
    args: argparse.Namespace
) -> dict:
    """
    Process a single dataset split.
    
    Args:
        split: Split name
        generator: Summary generator
        cfg: Config object
        args: Command line arguments
        
    Returns:
        Statistics dictionary
    """
    console.print(f"\n[bold yellow]Processing {split} split...")
    
    # Load data
    df = load_dataset(split, cfg)
    abstracts = df["abstract"].tolist()
    
    # Generate summaries
    console.print(f"[cyan]Generating summaries for {len(abstracts)} papers...")
    
    results = generator.generate_batch(
        abstracts=abstracts,
        include_methods=not args.no_methods,
        include_results=not args.no_results
    )
    
    # Add generated fields to DataFrame
    df["summary"] = [r.get("summary") for r in results]
    
    if not args.no_methods:
        df["methods"] = [r.get("methods") for r in results]
    
    if not args.no_results:
        df["results"] = [r.get("results") for r in results]
    
    # Apply fallback if requested
    if args.fallback_extractive:
        console.print("[yellow]Applying extractive fallback for missing summaries...")
        mask = df["summary"].isna() | (df["summary"] == "")
        df.loc[mask, "summary"] = df.loc[mask, "abstract"].apply(extractive_summary)
    
    # Save enriched dataset
    output_dir = Path(cfg.processed_dir)
    output_path = output_dir / f"papers_{split}_with_summaries.jsonl"
    
    # Convert to records and write
    records = df.to_dict("records")
    n_written = write_jsonl(str(output_path), records)
    
    console.print(f"[green]✓ Wrote {n_written} records to: {output_path}")
    
    # Calculate stats
    stats = generator.generate_stats(results)
    stats["split"] = split
    stats["output_file"] = str(output_path)
    
    return stats


def print_stats(all_stats: list):
    """Print generation statistics."""
    console.print("\n[bold cyan]═══ Generation Statistics ═══")
    
    for stats in all_stats:
        console.print(f"\n[bold]{stats['split'].upper()} Split:")
        console.print(f"  Total papers: {stats['total_papers']}")
        console.print(f"  Valid summaries: {stats['valid_summaries']} "
                     f"({stats['summary_success_rate']:.1%})")
        
        if "valid_methods" in stats:
            console.print(f"  Valid methods: {stats['valid_methods']}")
        
        if "valid_results" in stats:
            console.print(f"  Valid results: {stats['valid_results']}")
        
        console.print(f"  Avg summary length: {stats['avg_summary_length']:.1f} words")
        console.print(f"  Output: {stats['output_file']}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate arguments
    if not args.split and not args.all:
        console.print("[red]Error: Specify --split or --all")
        sys.exit(1)
    
    # Load configs
    cfg = Config()
    training_cfg = load_training_config(args.config)
    
    # Setup generator config
    silver_cfg_dict = training_cfg.get("silver_summaries", {})
    silver_cfg = SilverSummaryConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        max_length=silver_cfg_dict.get("max_length", 128),
        min_length=silver_cfg_dict.get("min_length", 30),
        num_beams=silver_cfg_dict.get("num_beams", 4),
        length_penalty=silver_cfg_dict.get("length_penalty", 2.0),
        no_repeat_ngram_size=silver_cfg_dict.get("no_repeat_ngram_size", 3),
        min_summary_words=silver_cfg_dict.get("min_summary_words", 10),
        max_summary_words=silver_cfg_dict.get("max_summary_words", 100)
    )
    
    # Initialize generator
    console.print("[bold cyan]Initializing Silver Summary Generator...")
    generator = SilverSummaryGenerator(silver_cfg)
    
    # Determine splits to process
    splits = ["train", "val", "test"] if args.all else [args.split]
    
    # Process each split
    all_stats = []
    for split in splits:
        try:
            stats = process_split(split, generator, cfg, args)
            all_stats.append(stats)
        except Exception as e:
            console.print(f"[red]Error processing {split}: {e}")
            continue
    
    # Print summary
    print_stats(all_stats)
    
    console.print("\n[bold green]✓ Summary generation complete!")
    console.print("\n[cyan]Next steps:")
    console.print("  1. Review generated summaries")
    console.print("  2. Run training: python scripts/train_model.py")


if __name__ == "__main__":
    main()