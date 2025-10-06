# CXR-LLAVA-v2 for Chest X-Ray Report Generation

This directory contains scripts for generating chest X-ray reports using the CXR-LLAVA-v2 model.

## Overview

CXR-LLAVA-v2 is a multimodal large language model specifically fine-tuned for chest X-ray interpretation. It combines visual understanding capabilities with medical knowledge to generate detailed radiological reports from X-ray images.

## Requirements

This module uses the consolidated environment setup from the main repository.

Key dependencies for CXR-LLAVA-v2 include:
```
torch>=2.0.0
transformers>=4.30.0
sentencepiece>=0.1.99
einops>=0.6.1
peft>=0.4.0
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
- Process images with CXR-LLAVA-v2
- Generate detailed radiological reports
- Save the reports to `/home/psaha03/scratch/results/cxr_llava_v2/cxr_llava_v2_reports.csv`

## How It Works

1. The script loads chest X-ray images
2. A prompt is prepared asking for a radiological report
3. The image and prompt are processed together by CXR-LLAVA-v2
4. The model generates a detailed radiological report
5. Results are saved to a CSV file with evaluation metrics

## Model Information

### ECOFRI/CXR-LLAVA-v2

CXR-LLAVA-v2 is a specialized medical vision-language model:

- Built on the LLAVA (Large Language and Vision Assistant) architecture
- Fine-tuned specifically for chest X-ray interpretation
- Combines visual understanding with medical knowledge
- Generates natural language descriptions and findings from chest X-rays

## Comparison with Other Models

CXR-LLAVA-v2 has some distinct characteristics compared to other models in this benchmark:

1. **Specialized Focus**: 
   - Unlike general medical VLMs, CXR-LLAVA-v2 is specifically optimized for chest X-rays

2. **Architecture**:
   - Based on the LLAVA architecture, which combines a vision encoder with an LLM

3. **Instruction Following**:
   - Designed to follow specific instructions in prompts related to radiological findings

4. **Output Quality**:
   - Generates structured reports with appropriate medical terminology

## References

- [CXR-LLAVA-v2 on Hugging Face](https://huggingface.co/ECOFRI/CXR-LLAVA-v2)
- [LLAVA: Large Language and Vision Assistant](https://llava-vl.github.io/)