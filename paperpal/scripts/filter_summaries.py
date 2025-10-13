"""Filter dataset to only include papers with valid summaries"""

import pandas as pd
from pathlib import Path

def filter_split(input_file, output_file):
    """Filter a split to only include papers with summaries"""
    df = pd.read_json(input_file, lines=True)
    
    print(f"Original: {len(df)} papers")
    
    # Filter for valid summaries
    df_filtered = df[
        df['summary'].notna() & 
        (df['summary'].str.strip() != '')
    ].copy()
    
    print(f"After filter: {len(df_filtered)} papers with summaries")
    
    # Save
    df_filtered.to_json(output_file, orient='records', lines=True)
    print(f"✓ Saved to {output_file}\n")
    
    return len(df_filtered)

def main():
    base_dir = Path('data/processed')
    
    for split in ['train', 'val', 'test']:
        input_file = base_dir / f'papers_{split}_with_summaries.jsonl'
        output_file = base_dir / f'papers_{split}_filtered.jsonl'
        
        if input_file.exists():
            print(f"Processing {split}...")
            count = filter_split(input_file, output_file)
            
            if count == 0:
                print(f"⚠️  Warning: {split} has 0 papers after filtering!")
        else:
            print(f"⚠️  {input_file} not found, skipping...")

if __name__ == '__main__':
    main()