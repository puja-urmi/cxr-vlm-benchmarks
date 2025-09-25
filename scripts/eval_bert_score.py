import pandas as pd
from bert_score import score

# Load ground truth and generated reports
gt = pd.read_csv('/home/psaha03/scratch/complete_dataset/indiana_reports_complete.csv')
gen = pd.read_csv('/home/psaha03/scratch/complete_dataset/cxr_generated_results.csv')

# Make sure the reports are aligned (matching order or by ID)
references = gt['Report'].tolist()
candidates = gen['Generated_Report'].tolist()  # Adjust column name as needed

# Compute BERTScore
P, R, F1 = score(candidates, references, lang='en', rescale_with_baseline=True)
print(f'BERTScore F1: {F1.mean().item():.4f}')