import pandas as pd
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

def main():
    # Paths
    reports_csv = '/home/psaha03/scratch/complete_dataset/indiana_test.csv'
    projections_csv = '/home/psaha03/scratch/complete_dataset/indiana_projections_complete.csv'
    image_dir = '/home/psaha03/scratch/complete_dataset/images'
    output_csv = '/home/psaha03/scratch/results/cxr_llava_v2/cxr_llava_v2_reports.csv'

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

    # Load CXR-LLAVA-v2 model and processor
    print("Loading CXR-LLAVA-v2 model...")
    model = AutoModel.from_pretrained("ECOFRI/CXR-LLAVA-v2", trust_remote_code=True, torch_dtype="auto")
    processor = AutoProcessor.from_pretrained("ECOFRI/CXR-LLAVA-v2")

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
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
            
        # Prepare a prompt asking for a chest X-ray report
        prompt = "Describe the findings in this chest X-ray. Generate a detailed radiological report."
        
        # Process image and text with CXR-LLAVA-v2
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        
        # Generate text from the model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode the generated text
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)
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
    print(f"Saved CXR-LLAVA-v2 reports to {output_csv}")

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
