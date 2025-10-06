import pandas as pd
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import load as load_metric
import argparse

def main(args):
    # Load CSV with embedding paths
    embeddings_csv = args.embeddings_csv
    output_csv = args.output_csv
    decoder_model_name = args.decoder_model

    print(f"Loading embeddings from: {embeddings_csv}")
    print(f"Using decoder model: {decoder_model_name}")
    
    df = pd.read_csv(embeddings_csv)
    
    # Check if required columns exist
    required_cols = ['uid', 'embedding_path', 'ground_truth_report']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Load decoder model
    print("Loading decoder model...")
    tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
    decoder_model = AutoModelForCausalLM.from_pretrained(decoder_model_name)
    
    results = []
    generated_list = []
    reference_list = []
    
    print("Generating reports from embeddings...")
    for idx, row in df.iterrows():
        uid = row['uid']
        embedding_path = row['embedding_path']
        ground_truth = row['ground_truth_report']
        
        if not os.path.exists(embedding_path):
            print(f"Embedding file not found: {embedding_path}")
            continue
        
        try:
            # Load the embedding
            image_embedding = torch.load(embedding_path)
            
            # Use a prompt to condition the decoder (optional)
            prompt = "Generate a chest X-ray report based on the image:"
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            # Combine embedding with prompt (implementation depends on decoder architecture)
            # Note: This is a simplified example; the actual implementation depends on decoder architecture
            # This assumes a decoder that can take embeddings as conditioning input
            outputs = decoder_model.generate(
                input_ids=input_ids,
                encoder_hidden_states=image_embedding.unsqueeze(0),  # Add batch dimension
                max_length=150,
                num_return_sequences=1,
                temperature=0.7
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'uid': uid,
                'generated_report': generated_text,
                'ground_truth_report': ground_truth
            })
            
            generated_list.append(generated_text)
            reference_list.append(ground_truth)
            
            print(f"UID: {uid}")
            print("Generated:", generated_text)
            print("Ground Truth:", ground_truth)
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing embedding {embedding_path}: {e}")
    
    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved generated reports to {output_csv}")
    
    # Compute evaluation metrics
    try:
        # ROUGE scores
        print("\nComputing evaluation metrics...")
        rouge = load_metric('rouge')
        rouge_result = rouge.compute(predictions=generated_list, references=reference_list, use_stemmer=True)
        print("\nROUGE scores:")
        for k, v in rouge_result.items():
            print(f"{k}: {v.mid.fmeasure:.4f}")
        
        # BLEU score
        try:
            bleu = load_metric('sacrebleu')
            bleu_result = bleu.compute(predictions=generated_list, references=[[ref] for ref in reference_list])
            print(f"\nBLEU score: {bleu_result['score']:.4f}")
        except Exception as bleu_err:
            print(f"Error computing BLEU: {bleu_err}")
        
        # BERTScore
        try:
            bertscore = load_metric('bertscore')
            bertscore_result = bertscore.compute(predictions=generated_list, references=reference_list, lang="en")
            print(f"\nBERTScore F1: {sum(bertscore_result['f1'])/len(bertscore_result['f1']):.4f}")
        except Exception as bert_err:
            print(f"Error computing BERTScore: {bert_err}")
            
    except Exception as e:
        print(f"Error computing evaluation metrics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate reports from rad-dino embeddings')
    parser.add_argument('--embeddings_csv', type=str, 
                        default='/home/psaha03/scratch/results/rad_dino/rad_dino_features.csv',
                        help='Path to CSV with embedding paths')
    parser.add_argument('--output_csv', type=str,
                        default='/home/psaha03/scratch/results/rad_dino/generated_reports_from_rad_dino.csv',
                        help='Path to save generated reports')
    parser.add_argument('--decoder_model', type=str,
                        default='gpt2',  # Replace with appropriate medical text decoder
                        help='Hugging Face model name for the decoder')
    
    args = parser.parse_args()
    main(args)