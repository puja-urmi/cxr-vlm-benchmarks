
import pandas as pd
from bert_score import score

# Load generated and ground truth reports from the same CSV
df = pd.read_csv('/home/psaha03/scratch/results/ft10/generated_report_ft10.csv')
candidates = df['generated_report'].astype(str).tolist()
references = df['ground_truth_report'].astype(str).tolist()

# Compute BERTScore
P, R, F1 = score(candidates, references, lang='en', rescale_with_baseline=True)
print(f'BERTScore F1: {F1.mean().item():.4f}')