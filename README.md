

# X-ray Report Generation using Vision Language Model

This repository contains tools and scripts for generating, fine-tuning, and evaluating radiology reports on the Indiana University Chest X-ray (CXR) dataset using different vision language models (such as BLIP, MAIRA 2, MedGemma etc.)


## Project Overview

- **Data**: Indiana University Chest X-ray (CXR) images, projections, and reports.
- **Goal**: Automate and evaluate radiology report generation using state-of-the-art vision-to-text models and metrics.
- **CXR**: CXR stands for Chest X-Ray, a common medical imaging abbreviation.


## Workflow

1. **Download and Analyze Data**
	- Use `scripts/download_data.sh` to download the dataset.
	- The notebook `original_eda.ipynb` shows analysis of real data.

2. **Preprocess Data**
	- Run `scripts/preprocess_data.py` to filter and organize images, projections, and reports.
	- Outputs are saved in `complete_dataset/`.
	- See `processed_eda.ipynb` for dataset completeness checks.

3. **Split Data**
	- Data is split into train:val:test = 1967:246:246.

4. **Fine-tune Vision-to-Text Model (Optional)**
	- Use `scripts/fine_tune_all.py` to fine-tune a vision-to-text model (e.g., `nathansutton/generate-cxr`) on your training data.
	- Adjust epochs, batch size, and model path as needed.

5. **Generate Reports using Pretrained or Fine-tuned Model**
	- Use `scripts/generate_cxr.py` to generate reports on the test dataset from images using the `nathansutton/generate-cxr` model or our fine-tuned model.
	- Results are saved as `generated_*.csv` in the output directory.

6. **Evaluate Reports**
	- Use `scripts/evaluate_cxr.py` to compare generated reports with ground truth using BERTScore, ROUGE, and other metrics.



## Requirements

Install dependencies in your environment:
```bash
pip install -r requirements.txt
pip install bert-score nltk rouge-score
```
(For GREEN Score, see their [GitHub](https://github.com/rajpurkarlab/green) for setup.)


## Running on a Cluster

Use `scripts/job.sh` as a template for SLURM job submission. Update paths as needed for your environment. For evaluation, see `eval_job.sh`.


## Notes

- The project uses the Indiana dataset, not Open-i.
- For pyarrow errors, always load the Arrow module before activating your environment.
- For semantic evaluation, BERTScore and GREEN Score are recommended.
- "CXR" = Chest X-Ray (medical abbreviation).


*Maintained by puja-urmi*
