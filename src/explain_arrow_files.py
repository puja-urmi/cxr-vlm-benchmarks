#!/usr/bin/env python3
"""
Script to explain and examine the Arrow files in the OpenI dataset
"""

import os
import json
import pandas as pd
from datasets import load_from_disk
import pyarrow as pa
import pyarrow.parquet as pq

def explain_arrow_format():
    """Explain what Arrow files are and their benefits"""
    explanation = """
    APACHE ARROW FILES (.arrow) - EXPLANATION
    ========================================
    
    What are Arrow files?
    - Apache Arrow is a columnar in-memory analytics format
    - Designed for fast data processing and analytics
    - Used by HuggingFace datasets library for efficient storage
    
    Why Arrow format for datasets?
    1. FAST LOADING: Much faster than CSV or JSON
    2. MEMORY EFFICIENT: Columnar storage reduces memory usage
    3. TYPE PRESERVATION: Maintains data types (images, text, numbers)
    4. COMPRESSION: Efficient compression reduces file size
    5. CROSS-LANGUAGE: Can be read by Python, R, Java, etc.
    
    Your OpenI dataset structure:
    - data-00000-of-00005.arrow: Shard 1 of 5 (contains subset of data)
    - data-00001-of-00005.arrow: Shard 2 of 5
    - data-00002-of-00005.arrow: Shard 3 of 5
    - data-00003-of-00005.arrow: Shard 4 of 5
    - data-00004-of-00005.arrow: Shard 5 of 5
    - dataset_info.json: Metadata about the dataset
    - state.json: Dataset state information
    
    Why multiple shards?
    - Large datasets are split into smaller files for efficiency
    - Enables parallel processing
    - Easier to handle in memory
    """
    print(explanation)

def examine_dataset_metadata():
    """Examine the metadata files"""
    data_dir = "/Users/pujasaha/Documents/openi/data/openi_dataset"
    
    print("\n" + "="*60)
    print("DATASET METADATA EXAMINATION")
    print("="*60)
    
    # Read dataset_info.json
    info_file = os.path.join(data_dir, "dataset_info.json")
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        
        print("\nDATASET INFO:")
        print("-" * 30)
        print(f"Dataset size: {dataset_info.get('dataset_size', 'Unknown')} bytes")
        print(f"Number of examples: {dataset_info.get('splits', {}).get('train', {}).get('num_examples', 'Unknown')}")
        
        features = dataset_info.get('features', {})
        print(f"\nDATASET FEATURES:")
        for feature_name, feature_info in features.items():
            print(f"  - {feature_name}: {feature_info.get('_type', 'Unknown type')}")
    
    # Read state.json
    state_file = os.path.join(data_dir, "state.json")
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state_info = json.load(f)
        
        print(f"\nDATASET STATE:")
        print("-" * 30)
        print(f"Version: {state_info.get('_data_files', [{}])[0].get('filename', 'Unknown')}")

def examine_arrow_file_details():
    """Examine the actual Arrow files"""
    data_dir = "/Users/pujasaha/Documents/openi/data/openi_dataset"
    
    print("\n" + "="*60)
    print("ARROW FILES EXAMINATION")
    print("="*60)
    
    arrow_files = [f for f in os.listdir(data_dir) if f.endswith('.arrow')]
    arrow_files.sort()
    
    total_size = 0
    for i, arrow_file in enumerate(arrow_files):
        file_path = os.path.join(data_dir, arrow_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        total_size += file_size
        
        print(f"\nShard {i+1}: {arrow_file}")
        print(f"  Size: {file_size:.2f} MB")
        
        # Try to read Arrow file directly
        try:
            table = pa.ipc.open_file(file_path).read_all()
            print(f"  Rows in this shard: {len(table)}")
            print(f"  Columns: {table.num_columns}")
            if i == 0:  # Show column names for first shard only
                print(f"  Column names: {table.column_names}")
        except Exception as e:
            print(f"  Error reading Arrow file: {e}")
    
    print(f"\nTotal dataset size: {total_size:.2f} MB")

def load_and_examine_sample():
    """Load the dataset and show a sample"""
    print("\n" + "="*60)
    print("DATASET SAMPLE EXAMINATION")
    print("="*60)
    
    try:
        # Load the dataset from disk
        dataset = load_from_disk("/Users/pujasaha/Documents/openi/data/openi_dataset")
        
        print(f"\nDataset loaded successfully!")
        print(f"Dataset type: {type(dataset)}")
        print(f"Dataset structure: {dataset}")
        
        if hasattr(dataset, 'num_rows'):
            print(f"Number of rows: {dataset.num_rows}")
        
        # Show first example
        if len(dataset) > 0:
            print(f"\nFirst example structure:")
            first_example = dataset[0]
            for key, value in first_example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: '{value[:100]}...' (truncated)")
                else:
                    print(f"  {key}: {type(value).__name__} - {str(value)[:50]}")
                    
        # Show column information
        print(f"\nDataset columns:")
        for feature_name, feature_type in dataset.features.items():
            print(f"  - {feature_name}: {feature_type}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

def main():
    """Main function to run all examinations"""
    print("OPENI DATASET ARROW FILES EXPLANATION")
    print("=" * 60)
    
    explain_arrow_format()
    examine_dataset_metadata()
    examine_arrow_file_details()
    load_and_examine_sample()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
    Your OpenI dataset is now stored locally in Apache Arrow format, which means:
    
    ✅ FASTER ACCESS: No need to download from internet each time
    ✅ EFFICIENT STORAGE: Compressed and optimized format
    ✅ TYPE SAFETY: All data types are preserved
    ✅ READY FOR ANALYSIS: Can be loaded instantly for your project
    
    Next steps for your project:
    1. Use the analyze_openi_dataset.py script to explore the data
    2. The data will load much faster now since it's local
    3. You can build your report generation model on this data
    """)

if __name__ == "__main__":
    main()