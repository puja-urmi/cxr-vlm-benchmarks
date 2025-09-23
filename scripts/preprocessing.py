#!/usr/bin/env python3
"""
Day 1 - Data Preprocessing for OpenI Dataset
Goal: Clean dataset and create train/test splits by end of today

Based on EDA findings:
- 3,851 total samples
- 13.3% missing findings (514 cases)
- 516 very short findings (<50 chars)
- Target: ~2,500-3,000 clean samples for training
"""

import pandas as pd
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import os
import json

def load_dataset():
    """Load the cached OpenI dataset"""
    print("🔄 Loading OpenI dataset from cache...")
    dataset = load_from_disk("/Users/pujasaha/Documents/openi/data/openi_dataset")
    df = dataset.to_pandas()
    print(f"✅ Loaded {len(df)} samples")
    return df

def clean_text_data(df):
    """Clean and filter the text data based on EDA findings"""
    print("\n🧹 CLEANING TEXT DATA...")
    print("=" * 40)
    
    initial_count = len(df)
    print(f"Starting with: {initial_count} samples")
    
    # 1. Remove samples with missing findings or impressions
    print("\n1️⃣ Removing missing findings/impressions...")
    before = len(df)
    df = df.dropna(subset=['findings', 'impression'])
    after = len(df)
    print(f"   Removed {before - after} samples with missing text")
    print(f"   Remaining: {after} samples")
    
    # 2. Calculate text lengths
    print("\n2️⃣ Calculating text lengths...")
    df['findings_length'] = df['findings'].astype(str).apply(len)
    df['impression_length'] = df['impression'].astype(str).apply(len)
    
    # 3. Remove very short findings (< 50 chars) and impressions (< 10 chars)
    print("\n3️⃣ Removing very short texts...")
    before = len(df)
    df = df[(df['findings_length'] >= 50) & (df['impression_length'] >= 10)]
    after = len(df)
    print(f"   Removed {before - after} samples with short texts")
    print(f"   Remaining: {after} samples")
    
    # 4. Remove duplicates based on findings text
    print("\n4️⃣ Removing duplicate findings...")
    before = len(df)
    df = df.drop_duplicates(subset=['findings'], keep='first')
    after = len(df)
    print(f"   Removed {before - after} duplicate findings")
    print(f"   Remaining: {after} samples")
    
    # 5. Basic text cleaning
    print("\n5️⃣ Basic text cleaning...")
    # Strip whitespace and normalize
    df['findings'] = df['findings'].str.strip()
    df['impression'] = df['impression'].str.strip()
    
    # Remove any remaining empty strings after stripping
    df = df[(df['findings'].str.len() > 0) & (df['impression'].str.len() > 0)]
    
    final_count = len(df)
    print(f"\n✅ CLEANING COMPLETE!")
    print(f"   Started with: {initial_count} samples")
    print(f"   Final clean dataset: {final_count} samples")
    print(f"   Cleaned {initial_count - final_count} samples ({((initial_count - final_count)/initial_count)*100:.1f}%)")
    
    return df

def create_train_test_split(df):
    """Create train/validation/test splits"""
    print("\n📊 CREATING TRAIN/TEST SPLITS...")
    print("=" * 40)
    
    # Select only the columns we need for text generation
    text_df = df[['uid', 'findings', 'impression', 'MeSH', 'findings_length', 'impression_length']].copy()
    
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(
        text_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=None  # Can't stratify on continuous variables
    )
    
    # Second split: 10% val, 10% test from the 20% temp
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42
    )
    
    print(f"📈 Split Summary:")
    print(f"   Training set: {len(train_df)} samples ({len(train_df)/len(text_df)*100:.1f}%)")
    print(f"   Validation set: {len(val_df)} samples ({len(val_df)/len(text_df)*100:.1f}%)")
    print(f"   Test set: {len(test_df)} samples ({len(test_df)/len(text_df)*100:.1f}%)")
    print(f"   Total: {len(train_df) + len(val_df) + len(test_df)} samples")
    
    return train_df, val_df, test_df

def analyze_splits(train_df, val_df, test_df):
    """Analyze the characteristics of each split"""
    print("\n📋 SPLIT ANALYSIS...")
    print("=" * 40)
    
    splits = {'Train': train_df, 'Validation': val_df, 'Test': test_df}
    
    for name, split_df in splits.items():
        print(f"\n{name} Set:")
        print(f"  Samples: {len(split_df)}")
        print(f"  Avg findings length: {split_df['findings_length'].mean():.0f} chars")
        print(f"  Avg impression length: {split_df['impression_length'].mean():.0f} chars")
        print(f"  Findings range: {split_df['findings_length'].min()}-{split_df['findings_length'].max()} chars")
        print(f"  Impression range: {split_df['impression_length'].min()}-{split_df['impression_length'].max()} chars")

def save_processed_data(train_df, val_df, test_df, output_dir="processed_data"):
    """Save the processed datasets"""
    print(f"\n💾 SAVING PROCESSED DATA...")
    print("=" * 40)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV files
    train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
    val_df.to_csv(f"{output_dir}/val_data.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data.csv", index=False)
    
    # Save combined for easy loading
    all_data = {
        'train': train_df.to_dict('records'),
        'validation': val_df.to_dict('records'), 
        'test': test_df.to_dict('records')
    }
    
    with open(f"{output_dir}/processed_dataset.json", 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Save metadata
    metadata = {
        'processing_date': pd.Timestamp.now().isoformat(),
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'avg_findings_length': float(train_df['findings_length'].mean()),
        'avg_impression_length': float(train_df['impression_length'].mean()),
        'processing_steps': [
            'Removed missing findings/impressions',
            'Removed short texts (findings < 50 chars, impressions < 10 chars)',
            'Removed duplicate findings',
            'Basic text cleaning and normalization',
            '80/10/10 train/val/test split'
        ]
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved processed data to '{output_dir}/' directory:")
    print(f"   📄 train_data.csv ({len(train_df)} samples)")
    print(f"   📄 val_data.csv ({len(val_df)} samples)")
    print(f"   📄 test_data.csv ({len(test_df)} samples)")
    print(f"   📄 processed_dataset.json (all splits)")
    print(f"   📄 metadata.json (processing info)")

def main():
    """Main preprocessing pipeline for Day 1"""
    print("🚀 DAY 1 - OPENI DATA PREPROCESSING")
    print("=" * 50)
    print("Goal: Clean dataset and create train/test splits")
    print("Deadline: End of today")
    print("=" * 50)
    
    # Step 1: Load data
    df = load_dataset()
    
    # Step 2: Clean data
    clean_df = clean_text_data(df)
    
    # Step 3: Create splits
    train_df, val_df, test_df = create_train_test_split(clean_df)
    
    # Step 4: Analyze splits
    analyze_splits(train_df, val_df, test_df)
    
    # Step 5: Save processed data
    save_processed_data(train_df, val_df, test_df)
    
    print(f"\n🎉 DAY 1 COMPLETE!")
    print("=" * 50)
    print("✅ Data successfully cleaned and split")
    print("✅ Ready for model training tomorrow")
    print(f"✅ {len(clean_df)} clean samples available")
    print("\n📅 TOMORROW (Day 2): Set up HuggingFace Transformers")
    print("📅 WEDNESDAY (Day 3): Train your first model")
    print("\n💪 Great job! You're on track for the 1-week deadline!")

if __name__ == "__main__":
    main()