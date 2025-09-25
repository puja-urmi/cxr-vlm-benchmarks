import pandas as pd
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    # Paths
    reports_csv = '/Users/pujasaha/Documents/iu_xray/complete_dataset/indiana_reports_complete.csv'
    projections_csv = '/Users/pujasaha/Documents/iu_xray/complete_dataset/indiana_projections_complete.csv'
    image_dir = '/Users/pujasaha/Documents/iu_xray/complete_dataset/images'
    output_csv = '/Users/pujasaha/Documents/iu_xray/complete_dataset/cxr_generated_results.csv'

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

    # Load model and processor
    processor = AutoProcessor.from_pretrained("nathansutton/generate-cxr")
    model = AutoModelForVision2Seq.from_pretrained("nathansutton/generate-cxr")

    # For saving results
    results = []
    generated_list = []
    reference_list = []

    # Evaluate
    for idx, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
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
