
# X-ray Report Generation

This repository contains tools and scripts for generating and evaluating radiology reports using the Indiana University Chest X-ray dataset.

## Project Overview

- **Data**: Indiana University Chest X-ray images, projections, and reports.
- **Goal**: Automate and evaluate radiology report generation using state-of-the-art models and metrics.

## Workflow

1. **Download Data**
	- Use `scripts/download_data.sh` to download the dataset.

2. **Preprocess Data**
	- Run `scripts/preprocess_data.py` to filter and organize images, projections, and reports.
		- Outputs are saved in `complete_dataset/`.

3. **Generate Reports**
	- Use `scripts/generate_cxr.py` to generate reports from images using the `nathansutton/generate-cxr` model.
		- Results are saved as `cxr_generated_results.csv` in the output directory.

4. **Evaluate Reports**
	- Use `scripts/evaluate_cxr.py` to compare generated reports with ground truth using BERTScore and other metrics.
	- (Optional) For clinical evaluation, set up and use the GREEN Score metric.



## Jupyter Notebooks: Step-by-Step Usage

1. **Explore Raw Data**
   - After downloading the data, open and run `notebooks/original_eda.ipynb` to explore the raw dataset.
2. **Preprocess Data**
   - Run the preprocessing script:
     ```bash
     python scripts/preprocess_data.py
     ```
3. **Explore Processed Data**
   - After preprocessing, open and run `notebooks/processed_eda.ipynb` to explore the cleaned/filtered dataset.

## Requirements

(Install dependencies in your environment):
```bash
pip install -r scripts/requirements.txt
pip install bert-score nltk rouge-score
```
(For GREEN Score, see their [GitHub](https://github.com/rajpurkarlab/green) for setup.)

## Running on a Cluster

Use `scripts/job.sh` as a template for SLURM job submission. Update paths as needed for your environment.

## Notes

- The project uses the Indiana dataset, not Open-i.
- For pyarrow errors, always load the Arrow module before activating your environment.
- For semantic evaluation, BERTScore and GREEN Score are recommended.

---

*Maintained by puja-urmi*