#!/usr/bin/env python3
"""
OpenI Dataset Analysis Script with Local Caching
Author: puja-urmi
Description: Comprehensive analysis of the OpenI dataset with local storage for efficiency
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set the local data path
LOCAL_DATA_PATH = "/Users/pujasaha/Documents/openi/data"
DATASET_CACHE_PATH = os.path.join(LOCAL_DATA_PATH, "openi_dataset")

def load_and_cache_dataset():
    """Load OpenI dataset and cache it locally for future use"""
    
    print("=" * 80)
    print("OPENI DATASET LOADING AND CACHING")
    print("=" * 80)
    
    try:
        # Check if dataset is already cached locally
        if os.path.exists(DATASET_CACHE_PATH) and os.listdir(DATASET_CACHE_PATH):
            print("� Found local dataset cache, loading from disk...")
            try:
                # Load from local cache
                cached_dataset = Dataset.load_from_disk(DATASET_CACHE_PATH)
                print("✅ Dataset loaded successfully from local cache!")
                print(f"📂 Cache location: {DATASET_CACHE_PATH}")
                print(f"📊 Loaded {len(cached_dataset)} examples from cache")
                
                # Return in the expected format
                return {'train': cached_dataset}
                
            except Exception as cache_error:
                print(f"⚠️  Error loading from cache: {cache_error}")
                print("🔄 Will download fresh dataset...")
        else:
            print("🌐 No local cache found, downloading dataset from HuggingFace...")
        
        # Download the dataset
        print("📥 Downloading OpenI dataset...")
        ds = load_dataset("ykumards/open-i")
        print("✅ Dataset downloaded successfully!")
        
        # Ensure the cache directory exists
        os.makedirs(DATASET_CACHE_PATH, exist_ok=True)
        
        # Save dataset locally for future use
        print("💾 Saving dataset to local cache...")
        
        # Determine which split to cache
        if 'train' in ds:
            ds['train'].save_to_disk(DATASET_CACHE_PATH)
            print(f"✅ Training split saved to: {DATASET_CACHE_PATH}")
        elif len(ds) > 0:
            # If no 'train' split, save the first available split
            first_split = list(ds.keys())[0]
            ds[first_split].save_to_disk(DATASET_CACHE_PATH)
            print(f"✅ {first_split} split saved to: {DATASET_CACHE_PATH}")
        
        # Display cache information
        cache_size = sum(os.path.getsize(os.path.join(DATASET_CACHE_PATH, f)) 
                        for f in os.listdir(DATASET_CACHE_PATH) 
                        if os.path.isfile(os.path.join(DATASET_CACHE_PATH, f)))
        print(f"💾 Cache size: {cache_size / 1024**2:.2f} MB")
        
        return ds
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print("This might be due to network issues or dataset access problems.")
        return None

def analyze_dataset_structure(ds):
    """Analyze and display dataset structure"""
    
    print("\n" + "="*50)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*50)
    
    print(f"📋 Available splits: {list(ds.keys())}")
    
    for split_name, split_data in ds.items():
        print(f"\n📊 {split_name.upper()} SPLIT:")
        print(f"  - Number of examples: {len(split_data):,}")
        print(f"  - Features: {list(split_data.features.keys())}")
        
        # Detailed feature analysis
        for feature_name, feature_info in split_data.features.items():
            print(f"  - {feature_name}: {feature_info}")

def analyze_sample_data(ds):
    """Analyze sample data from the dataset"""
    
    print("\n" + "="*50)
    print("SAMPLE DATA ANALYSIS")
    print("="*50)
    
    # Use the first available split
    split_name = list(ds.keys())[0]
    split_data = ds[split_name]
    
    print(f"\n🔍 Analyzing samples from '{split_name}' split...")
    
    # Show first few examples
    num_samples = min(3, len(split_data))
    
    for i in range(num_samples):
        sample = split_data[i]
        print(f"\n--- SAMPLE {i+1} ---")
        
        for key, value in sample.items():
            if isinstance(value, str):
                # Truncate long text for display
                display_value = value[:200] + "..." if len(value) > 200 else value
                print(f"{key}: {display_value}")
            elif isinstance(value, list):
                print(f"{key}: List with {len(value)} items")
                if value and len(str(value[0])) < 100:
                    print(f"  First few items: {value[:3]}")
            else:
                print(f"{key}: {value}")
    
    return split_data

def generate_statistics(split_data):
    """Generate detailed statistics about the dataset"""
    
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    # Convert to pandas for easier analysis
    try:
        df = split_data.to_pandas()
        
        print(f"📈 Total examples: {len(df)}")
        print(f"📊 Number of features: {len(df.columns)}")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Analyze each column
        print("\n🔍 FEATURE ANALYSIS:")
        for col in df.columns:
            print(f"\n  {col}:")
            
            if df[col].dtype == 'object':
                # Text/string analysis
                non_null = df[col].notna().sum()
                print(f"    - Non-null values: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
                
                if non_null > 0:
                    lengths = df[col].dropna().astype(str).str.len()
                    print(f"    - Average length: {lengths.mean():.1f} characters")
                    print(f"    - Length range: {lengths.min()} - {lengths.max()}")
                    
                    # Show unique values if reasonable
                    unique_vals = df[col].nunique()
                    print(f"    - Unique values: {unique_vals}")
                    
                    if unique_vals < 20:
                        print(f"    - Sample values: {list(df[col].dropna().unique()[:5])}")
            else:
                # Numeric analysis
                print(f"    - Data type: {df[col].dtype}")
                print(f"    - Non-null values: {df[col].notna().sum()}/{len(df)}")
                if df[col].notna().sum() > 0:
                    print(f"    - Min: {df[col].min()}, Max: {df[col].max()}")
                    print(f"    - Mean: {df[col].mean():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error generating statistics: {str(e)}")
        return None

def create_visualizations(df, output_dir="analysis_output"):
    """Create visualizations for the dataset"""
    
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Text length distribution (if applicable)
        text_columns = df.select_dtypes(include=['object']).columns
        
        if len(text_columns) > 0:
            fig, axes = plt.subplots(len(text_columns), 1, figsize=(10, 4*len(text_columns)))
            if len(text_columns) == 1:
                axes = [axes]
            
            for i, col in enumerate(text_columns):
                if df[col].notna().sum() > 0:
                    lengths = df[col].dropna().astype(str).str.len()
                    axes[i].hist(lengths, bins=50, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Text Length Distribution: {col}')
                    axes[i].set_xlabel('Length (characters)')
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/text_length_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved text length distribution to {output_dir}/text_length_distribution.png")
        
        # Missing data visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', ax=ax)
            ax.set_title('Missing Data by Feature')
            ax.set_xlabel('Features')
            ax.set_ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/missing_data.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved missing data analysis to {output_dir}/missing_data.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        return False

def generate_report(ds, df, output_dir="analysis_output"):
    """Generate a comprehensive report"""
    
    print("\n" + "="*50)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate detailed report
    report = {
        "dataset_name": "OpenI Dataset",
        "analysis_summary": {
            "total_splits": len(ds.keys()),
            "splits": list(ds.keys()),
            "total_examples": sum(len(split) for split in ds.values()),
        },
        "recommendations": [
            "✅ Dataset successfully loaded and analyzed",
            "🔍 Consider preprocessing text data for consistency",
            "📊 Explore relationships between different features",
            "🎯 Plan model architecture based on data characteristics",
            "📝 Prepare data preprocessing pipeline"
        ]
    }
    
    if df is not None:
        report["detailed_analysis"] = {
            "features": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "missing_data": df.isnull().sum().to_dict()
        }
    
    # Save report as JSON
    with open(f"{output_dir}/dataset_analysis_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown report
    markdown_report = f"""# OpenI Dataset Analysis Report

## Overview
- **Dataset**: OpenI (Open-i medical reports dataset)
- **Purpose**: Medical report generation and analysis
- **Total Splits**: {len(ds.keys())}
- **Available Splits**: {', '.join(ds.keys())}

## Key Findings
{chr(10).join(['- ' + rec for rec in report['recommendations']])}

## Next Steps for Your Project
1. **Data Preprocessing**: Clean and prepare text data
2. **Feature Engineering**: Extract relevant features for report generation
3. **Model Selection**: Choose appropriate models for your specific use case
4. **Evaluation Metrics**: Define how to measure success
5. **Pipeline Development**: Build end-to-end processing pipeline

## Files Generated
- `dataset_analysis_report.json`: Detailed analysis in JSON format
- `text_length_distribution.png`: Visualization of text lengths
- `missing_data.png`: Missing data analysis
- `dataset_analysis_report.md`: This report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(f"{output_dir}/dataset_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    
    print(f"✅ Comprehensive report saved to {output_dir}/")
    print(f"📄 Files created:")
    print(f"  - dataset_analysis_report.json")
    print(f"  - dataset_analysis_report.md")
    
    return report

def main():
    """Main function to run the complete analysis"""
    
    # Load and cache the dataset
    ds = load_and_cache_dataset()
    if ds is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Analyze dataset structure
    analyze_dataset_structure(ds)
    
    # Analyze sample data
    split_data = analyze_sample_data(ds)
    
    # Generate statistics
    df = generate_statistics(split_data)
    if df is None:
        print("❌ Failed to generate statistics. Exiting.")
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis_output')
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Save comprehensive report
    report = generate_report(ds, df, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📁 Dataset cached at: {DATASET_CACHE_PATH}")
    print(f"📊 Analysis results saved to: {output_dir}")
    print("\nFor future runs, the dataset will load much faster from the local cache!")
    print("="*80)
    if df is not None:
        create_visualizations(df)
    
    # Generate comprehensive report
    report = generate_report(ds, df)
    
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*80)
    print("📂 Check the 'analysis_output' folder for detailed results")
    print("📊 Review the generated reports and visualizations")
    print("🚀 You're now ready to proceed with your OpenI project!")

if __name__ == "__main__":
    main()