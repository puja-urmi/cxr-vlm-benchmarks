# MedGemma Chest X-Ray Report Generation

This directory contains scripts for generating chest X-ray reports using Google's MedGemma 4B-IT model.

## Overview

MedGemma is a multimodal medical large language model that can process both images and text inputs to generate medical reports. It's particularly well-suited for radiology report generation from chest X-rays.

## Requirements

This module uses the consolidated environment setup from the main repository.

Key dependencies for MedGemma include:
```
torch>=2.0.0
transformers>=4.30.0
accelerate
bitsandbytes  # For 4-bit quantization
```

Install all required dependencies with:
```bash
# From the main repository directory
pip install -r requirements.txt
```

For detailed environment setup, see [ENVIRONMENT.md](../ENVIRONMENT.md) in the main directory.

## Usage

Run the script to generate reports from chest X-ray images:

```bash
python generate_cxr.py
```

By default, this will:
- Load the test dataset from `/home/psaha03/scratch/complete_dataset/`
- Process images with MedGemma 4B-IT
- Generate reports for each image
- Save the reports to `/home/psaha03/scratch/results/medgemma_4bit/medgemma_generated_reports.csv`
- Calculate evaluation metrics including ROUGE scores

## How It Works

1. The script loads chest X-ray images
2. It creates a prompt for each image asking for a radiological report
3. The MedGemma model processes the image and prompt together
4. The model generates a detailed radiological report
5. Results are saved to a CSV file

## Model Information

### Google MedGemma 4B-IT

MedGemma is a medical multimodal model designed specifically for healthcare applications. Key features:

- 4 billion parameters
- Fine-tuned on medical image-text pairs
- Supports 4-bit quantization for efficient inference
- Can handle both visual and textual inputs simultaneously

## Tips for Best Results

1. **Prompt Engineering**: Experiment with different prompts to guide MedGemma
2. **Quantization**: Use 4-bit quantization for efficient inference on consumer hardware
3. **Batch Processing**: For large datasets, consider implementing batch processing
4. **GPU Requirements**: While 4-bit quantization reduces memory requirements, a GPU is still recommended