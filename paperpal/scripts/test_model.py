"""
Quick Model Testing Script

Test your trained model with sample inputs.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --model ./models/checkpoints/final_model
    python scripts/test_model.py --interactive
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paperpal.model import PaperPalModel, ModelConfig

console = Console()


# Sample abstracts for testing
SAMPLE_ABSTRACTS = [
    """
    We introduce BERT, a new language representation model which stands for 
    Bidirectional Encoder Representations from Transformers. Unlike recent language 
    representation models, BERT is designed to pre-train deep bidirectional 
    representations from unlabeled text by jointly conditioning on both left and 
    right context in all layers. As a result, the pre-trained BERT model can be 
    fine-tuned with just one additional output layer to create state-of-the-art 
    models for a wide range of tasks, such as question answering and language 
    inference, without substantial task-specific architecture modifications.
    """,
    """
    Attention mechanisms have become an integral part of compelling sequence modeling 
    and transduction models in various tasks, allowing modeling of dependencies without 
    regard to their distance in the input or output sequences. In this work we propose 
    the Transformer, a model architecture eschewing recurrence and instead relying 
    entirely on an attention mechanism to draw global dependencies between input and 
    output. The Transformer allows for significantly more parallelization and can reach 
    a new state of the art in translation quality after being trained for as little as 
    twelve hours on eight P100 GPUs.
    """,
    """
    Generative Pre-trained Transformer (GPT) models have demonstrated impressive 
    performance across a wide range of natural language processing tasks. In this paper, 
    we scale up language models to 175 billion parameters and show that these models can 
    perform various tasks with few-shot learning capabilities. We evaluate GPT-3 on over 
    two dozen NLP datasets as well as several novel tasks designed to test rapid adaptation 
    to new tasks, and find that GPT-3 achieves strong performance on many tasks without 
    any gradient updates or fine-tuning, through in-context learning.
    """
]


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained PaperPal model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="./models/checkpoints/final_model",
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - paste your own abstracts"
    )
    
    parser.add_argument(
        "--abstract",
        type=str,
        default=None,
        help="Single abstract to summarize"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="File containing abstract (one abstract per file)"
    )
    
    return parser.parse_args()


def load_model(model_path: str) -> PaperPalModel:
    """Load the trained model."""
    console.print(f"[cyan]Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        console.print(f"[red]Error: Model not found at {model_path}")
        console.print("[yellow]Have you trained the model yet?")
        console.print("[yellow]Run: python scripts/train_model.py")
        sys.exit(1)
    
    config = ModelConfig(
        model_name=model_path,
        device="cpu"
    )
    
    model = PaperPalModel(config, load_model=True)
    console.print("[green]✓ Model loaded successfully\n")
    
    return model


def summarize_and_display(model: PaperPalModel, abstract: str, title: str = "Summary"):
    """Generate and display a summary."""
    console.print(Panel(
        f"[bold]Abstract:[/bold]\n{abstract.strip()}",
        title="Input",
        border_style="cyan"
    ))
    
    console.print("\n[yellow]Generating summary...[/yellow]\n")
    
    summary = model.summarize(abstract)
    
    console.print(Panel(
        summary,
        title=f"✨ {title}",
        border_style="green"
    ))
    
    # Show word counts
    abstract_words = len(abstract.split())
    summary_words = len(summary.split())
    compression = abstract_words / summary_words if summary_words > 0 else 0
    
    console.print(f"\n[dim]Abstract: {abstract_words} words | "
                 f"Summary: {summary_words} words | "
                 f"Compression: {compression:.1f}x[/dim]\n")


def interactive_mode(model: PaperPalModel):
    """Run in interactive mode."""
    console.print("\n[bold cyan]Interactive Mode[/bold cyan]")
    console.print("Paste an abstract and press Enter twice (empty line) to summarize.")
    console.print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        console.print("[yellow]Paste abstract:[/yellow]")
        lines = []
        
        while True:
            try:
                line = input()
                if line.strip().lower() in ['quit', 'exit']:
                    console.print("\n[cyan]Goodbye![/cyan]")
                    return
                if line.strip() == "" and lines:
                    break
                if line.strip():
                    lines.append(line)
            except EOFError:
                return
        
        abstract = " ".join(lines)
        
        if abstract.strip():
            summarize_and_display(model, abstract)
        else:
            console.print("[yellow]Empty input, please try again.[/yellow]\n")


def test_with_samples(model: PaperPalModel):
    """Test with sample abstracts."""
    console.print("[bold cyan]Testing with Sample Abstracts[/bold cyan]\n")
    
    titles = ["BERT", "Transformer", "GPT-3"]
    
    for i, (abstract, title) in enumerate(zip(SAMPLE_ABSTRACTS, titles), 1):
        console.print(f"\n[bold]Example {i}: {title}[/bold]")
        console.print("="*60)
        summarize_and_display(model, abstract, f"Summary - {title}")
        
        if i < len(SAMPLE_ABSTRACTS):
            console.print("\n")


def main():
    args = parse_args()
    
    console.print("\n[bold cyan]═══ PaperPal Model Tester ═══[/bold cyan]\n")
    
    # Load model
    model = load_model(args.model)
    
    # Different modes
    if args.interactive:
        interactive_mode(model)
    
    elif args.abstract:
        summarize_and_display(model, args.abstract, "Custom Abstract")
    
    elif args.file:
        if not Path(args.file).exists():
            console.print(f"[red]Error: File not found: {args.file}[/red]")
            sys.exit(1)
        
        with open(args.file, 'r') as f:
            abstract = f.read()
        
        summarize_and_display(model, abstract, f"From {Path(args.file).name}")
    
    else:
        test_with_samples(model)
    
    console.print("\n[green]✓ Testing complete![/green]\n")


if __name__ == "__main__":
    main()