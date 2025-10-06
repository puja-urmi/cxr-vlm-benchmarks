import pandas as pd
import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def main():
    # Paths
    reports_csv = '/home/psaha03/scratch/complete_dataset/indiana_test.csv'
    projections_csv = '/home/psaha03/scratch/complete_dataset/indiana_projections_complete.csv'
    image_dir = '/home/psaha03/scratch/complete_dataset/images'
    output_csv = '/home/psaha03/scratch/results/rad_dino/rad_dino_features.csv'

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

    # Load rad-dino model and processor
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino")
    model = AutoModel.from_pretrained("microsoft/rad-dino")

    # # Load model and processor from local fine-tuned directory
    # model_dir = '/home/psaha03/scratch/models/20_epoch/fine_tuned_generate_cxr'
    # processor = AutoProcessor.from_pretrained(model_dir)
    # model = AutoModelForVision2Seq.from_pretrained(model_dir)

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
        # Process image with rad-dino model
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # rad-dino returns embeddings, not text directly
        # Extract the image features (embeddings) from the model output
        image_features = outputs.last_hidden_state.mean(dim=1)  # Pool features
        
        # Save the embedding vector for later use with a decoder
        embedding_path = f"/home/psaha03/scratch/results/rad_dino/embeddings/{row['uid']}_embedding.pt"
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        
        torch.save(image_features, embedding_path)
        
        # For CSV reporting, we'll store the path to the embedding file
        results.append({
            'uid': row['uid'],
            'image_path': img_path,
            'embedding_path': embedding_path,
            'ground_truth_report': row['impression']
        })
        
        # No generated text at this stage, only embeddings
        reference_list.append(row['impression'])
        print(f"UID: {row['uid']}")
        print(f"Saved embedding to: {embedding_path}")
        print("Ground Truth:", row['impression'])
        print("-" * 40)

    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved embedding paths to {output_csv}")
    print(f"\nEmbeddings were extracted for {len(results)} images")
    print(f"Use the decoder script to generate reports from these embeddings")

if __name__ == "__main__":
    main()
