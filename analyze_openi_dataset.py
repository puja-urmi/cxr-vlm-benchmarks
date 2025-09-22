# analyze_openi_dataset.py

"""
Script to load and analyze the Open-i dataset for report generation.
"""

from datasets import load_dataset

# Load the Open-i dataset
ds = load_dataset("ykumards/open-i")

# Print basic information about the dataset
print("Dataset structure:")
print(ds)

# Show a sample from the training split (if available)
if 'train' in ds:
    print("\nSample from training split:")
    print(ds['train'][0])
else:
    print("\nNo training split found in the dataset.")
