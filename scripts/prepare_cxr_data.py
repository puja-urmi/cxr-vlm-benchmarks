#!/usr/bin/env python3
"""
Prepare OpenI test set for MAIRA-2 evaluation
"""
import pandas as pd
import os

def main():
    # Path to processed test set
    test_path = "processed_data/test_data.csv"
    if not os.path.exists(test_path):
        print(f"Test set not found at {test_path}")
        return

    df = pd.read_csv(test_path)
    print(f"Loaded {len(df)} test samples")

    # MAIRA-2 expects text generation: findings → impression
    # We'll create a list of dicts: { 'input': findings, 'target': impression }
    eval_data = [
        {"input": row["findings"], "target": row["impression"]}
        for _, row in df.iterrows()
    ]

    # Save as JSON for easy loading
    import json
    with open("processed_data/maira2_eval_test.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Saved MAIRA-2 eval data to processed_data/maira2_eval_test.json")

if __name__ == "__main__":
    main()
