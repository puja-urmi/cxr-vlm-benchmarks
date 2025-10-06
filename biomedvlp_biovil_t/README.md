# BiomedVLP-BioViL-T for Chest X-Ray Analysis

This directory contains scripts for analyzing chest X-rays using Microsoft's BiomedVLP-BioViL-T model.

## Overview

BiomedVLP-BioViL-T is a vision-language model trained on biomedical data. It generates embeddings that capture the semantic content of medical images, which can be used for various downstream tasks such as retrieval, classification, or report generation.

## Requirements

This module uses the consolidated environment setup from the main repository.

Key dependencies for BiomedVLP-BioViL-T include:
```
torch
transformers
scikit-learn
```

Install all required dependencies with:
```bash
# From the main repository directory
pip install -r requirements.txt
```

For detailed environment setup, see [ENVIRONMENT.md](../ENVIRONMENT.md) in the main directory.

## Usage

Run the script to process chest X-ray images:

```bash
python generate_cxr.py
```

By default, this will:
- Load the test dataset from `/home/psaha03/scratch/complete_dataset/`
- Process images with BiomedVLP-BioViL-T
- Extract embeddings for each image
- Save the results to a CSV file

## How It Works

1. The script loads chest X-ray images
2. It processes each image through the BiomedVLP-BioViL-T model
3. The model generates embeddings that capture the semantic content of the images
4. These embeddings can be used for downstream tasks like retrieval or classification

## Model Information

### Microsoft BiomedVLP-BioViL-T

BiomedVLP-BioViL-T is a biomedical vision-language model designed for understanding medical images:

- Trained on biomedical data
- Generates embeddings that capture semantic content of images
- Can be used for tasks like retrieval, classification, or zero-shot prediction
- Works with the `trust_remote_code=True` parameter in Hugging Face

## Comparison with Other Models

Unlike MedGemma which is a generative multimodal model, BiomedVLP-BioViL-T is primarily an embedding model that creates vector representations of images. Here are the key differences:

1. **Model Purpose**:
   - BiomedVLP-BioViL-T: Creates embeddings for retrieval and classification
   - MedGemma: Generates text directly from images

2. **Memory Requirements**:
   - BiomedVLP-BioViL-T: Lower memory requirements
   - MedGemma: Higher memory requirements (benefits from 4-bit quantization)

3. **Output Format**:
   - BiomedVLP-BioViL-T: Vector embeddings
   - MedGemma: Natural language text

## References

- [Microsoft BiomedVLP-BioViL-T on Hugging Face](https://huggingface.co/microsoft/BiomedVLP-BioViL-T)
- [BioViL Paper](https://arxiv.org/abs/2204.09817)