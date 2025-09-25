
import pandas as pd
import os
import shutil


import pandas as pd
import os
import shutil

# Paths
image_dir = '/home/psaha03/scratch/original_dataset/images_normalized'
reports_path = '/home/psaha03/scratch/original_dataset/indiana_reports.csv'
projections_path = '/home/psaha03/scratch/original_dataset/indiana_projections.csv'

# Output paths
output_dir = '/home/psaha03/scratch/complete_dataset'
output_image_dir = os.path.join(output_dir, 'images')
os.makedirs(output_image_dir, exist_ok=True)

# Load data
reports_df = pd.read_csv(reports_path)
projections_df = pd.read_csv(projections_path)

# Filter for complete reports (no missing values in key columns)
required_cols = ['uid', 'MeSH', 'Problems', 'image', 'indication', 'comparison', 'findings', 'impression']
complete_reports_df = reports_df.dropna(subset=required_cols)
print(f"Complete reports: {len(complete_reports_df)}")

# Save new reports CSV
complete_reports_csv = os.path.join(output_dir, 'indiana_reports_complete.csv')
complete_reports_df.to_csv(complete_reports_csv, index=False)
print(f"Saved complete reports to {complete_reports_csv}")

# Get valid uids
valid_uids = set(complete_reports_df['uid'])

# Filter projections by uid
if 'uid' in projections_df.columns:
    filtered_projections_df = projections_df[projections_df['uid'].isin(valid_uids)]
    complete_projections_csv = os.path.join(output_dir, 'indiana_projections_complete.csv')
    filtered_projections_df.to_csv(complete_projections_csv, index=False)
    print(f"Saved filtered projections to {complete_projections_csv}")
else:
    print("No 'uid' column found in projections_df. Please check the CSV header.")


# Copy images by matching uid in filenames
import re
image_files = os.listdir(image_dir)
copied = 0
for uid in valid_uids:
    pattern = re.compile(rf"{uid}.*\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)
    matches = [f for f in image_files if pattern.search(f)]
    if matches:
        for img_name in matches:
            src = os.path.join(image_dir, img_name)
            dst = os.path.join(output_image_dir, img_name)
            shutil.copy2(src, dst)
            copied += 1
    else:
        print(f"Warning: No image file found for uid {uid}")
print(f"Copied {copied} images to {output_image_dir}")