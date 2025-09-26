import pandas as pd
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from torch.utils.data import Dataset

train_path = '/home/psaha03/scratch/complete_dataset/indiana_train.csv'
val_path = '/home/psaha03/scratch/complete_dataset/indiana_val.csv'
projections_path = '/home/psaha03/scratch/complete_dataset/indiana_projections_complete.csv'
image_dir = '/home/psaha03/scratch/complete_dataset/images'

MODEL_NAME = "nathansutton/generate-cxr"

class IndianaCXRDataset(Dataset):
	def __init__(self, df, processor, image_dir):
		self.df = df
		self.processor = processor
		self.image_dir = image_dir

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		img_path = os.path.join(self.image_dir, row['filename'])
		image = Image.open(img_path).convert('RGB')
		inputs = self.processor(images=image, text=row['impression'], return_tensors="pt", padding="max_length", truncation=True, max_length=256)
		# Flatten batch dimension
		item = {k: v.squeeze(0) for k, v in inputs.items()}
		item["labels"] = item["input_ids"]
		return item

def main():
	# Load data
	train_df = pd.read_csv(train_path)
	val_df = pd.read_csv(val_path)
	projections_df = pd.read_csv(projections_path)

	# Merge to get filenames for each uid
	train_df = pd.merge(train_df, projections_df[['uid', 'filename']], on='uid', how='inner')
	val_df = pd.merge(val_df, projections_df[['uid', 'filename']], on='uid', how='inner')

	processor = AutoProcessor.from_pretrained(MODEL_NAME)
	model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME)

	train_dataset = IndianaCXRDataset(train_df, processor, image_dir)
	val_dataset = IndianaCXRDataset(val_df, processor, image_dir)

	training_args = Seq2SeqTrainingArguments(
		output_dir="./results",
		per_device_train_batch_size=2,
		per_device_eval_batch_size=2,
		num_train_epochs=20,
		logging_dir="./logs",
		fp16=torch.cuda.is_available(),
	)

	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		tokenizer=processor,
	)

	trainer.train()
	model.save_pretrained("./fine_tuned_generate_cxr")
	processor.save_pretrained("./fine_tuned_generate_cxr")
	print("Fine-tuning complete. Model saved to ./fine_tuned_generate_cxr")

if __name__ == "__main__":
	main()
