import pandas as pd
import os
from PIL import Image
from transformers import pipeline

def main():
    # Paths
    reports_csv = '/home/psaha03/scratch/complete_dataset/indiana_test.csv'
    projections_csv = '/home/psaha03/scratch/complete_dataset/indiana_projections_complete.csv'
    image_dir = '/home/psaha03/scratch/complete_dataset/images'
    output_csv = '/home/psaha03/scratch/results/medgemma_4bit/medgemma_generated_reports.csv'

    # Load data
    reports = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)

    # Merge on 'uid' (adjust if your key is different)
    if 'uid' not in projections.columns:
        raise ValueError("'uid' column not found in projections CSV.")
    if 'uid' not in reports.columns:
        raise ValueError("'uid' column not found in reports CSV.")
    if 'filename' not in projections.columns:
        raise ValueError("'filename' column not found in projections CSV.")

    df = pd.merge(projections, reports[['uid', 'impression']], on='uid', how='inner')

    # Add image path
    df['image_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))

    # Use a pipeline as a high-level helper
    from transformers import pipeline
    
    # Initialize MedGemma pipeline
    print("Loading MedGemma pipeline...")
    pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it")
    print("MedGemma pipeline loaded successfully")

    # For saving results
    results = []
    generated_list = []
    reference_list = []

    # Evaluate only the first 5 cases
    for idx, row in df.head(50).iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        try:
            # Open the image
            image = Image.open(img_path).convert('RGB')
            
            # Create messages for MedGemma
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": img_path},  # Using local file path
                        {"type": "text", "text": "Describe this chest X-ray in detail. Provide a radiological report."}
                    ]
                },
            ]
            
            # Process with MedGemma
            response = pipe(text=messages)
            generated_text = response[0]["generated_text"]
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
        results.append({
            'uid': row['uid'],
            'image_path': img_path,
            'generated_report': generated_text,
            'ground_truth_report': row['impression']
        })
        generated_list.append(generated_text)
        reference_list.append(row['impression'])
        print(f"UID: {row['uid']}")
        print("Generated:", generated_text)
        print("Ground Truth:", row['impression'])
        print("-" * 40)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # Compute ROUGE scores
    try:
        from evaluate import load as load_metric
        rouge = load_metric('rouge')
        rouge_result = rouge.compute(predictions=generated_list, references=reference_list, use_stemmer=True)
        print("\nROUGE scores:")
        for k, v in rouge_result.items():
            print(f"{k}: {v:.4f}")
    except ImportError:
        print("Install the 'evaluate' package to compute ROUGE scores: pip install evaluate")
    except Exception as e:
        print(f"Error computing ROUGE: {e}")

if __name__ == "__main__":
    main()
