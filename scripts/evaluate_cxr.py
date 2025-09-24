#!/usr/bin/env python3
"""
Evaluate MAIRA-2 model on OpenI test set
"""
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load MAIRA-2 model and tokenizer
print("Loading MAIRA-2 model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/maira-2", trust_remote_code=True)
model.eval()

# Load prepared test data
with open("processed_data/maira2_eval_test.json") as f:
    eval_data = json.load(f)
print(f"Loaded {len(eval_data)} test samples for evaluation.")

# Run inference and compare outputs
def generate_impression(findings):
    inputs = tokenizer(findings, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64)
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated

results = []
for i, sample in enumerate(eval_data):
    input_text = sample["input"]
    target_text = sample["target"]
    generated_text = generate_impression(input_text)
    results.append({
        "input": input_text,
        "generated": generated_text,
        "target": target_text
    })
    if i < 5:
        print(f"Sample {i+1}")
        print("Findings:", input_text)
        print("Generated Impression:", generated_text)
        print("Ground Truth:", target_text)
        print("-" * 40)

# Save all results for further analysis
to_save = "processed_data/maira2_eval_results.json"
with open(to_save, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved all evaluation results to {to_save}")
