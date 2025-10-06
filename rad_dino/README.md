# RAD-DINO Chest X-Ray Report Generation Pipeline

This directory contains scripts for generating chest X-ray reports using Microsoft's RAD-DINO model. The process is split into two steps:

1. Feature extraction using RAD-DINO
2. Report generation using a decoder model

## Requirements

This module uses the consolidated environment setup from the main repository.

Key dependencies for RAD-DINO include:
```
torch
transformers
timm
evaluate
```

Install all required dependencies with:
```bash
# From the main repository directory
pip install -r requirements.txt
```

For detailed environment setup, see [ENVIRONMENT.md](../ENVIRONMENT.md) in the main directory.

## Feature Extraction with RAD-DINO

The `generate_cxr.py` script extracts visual features from chest X-ray images using Microsoft's RAD-DINO model. This model provides powerful medical image representations that capture relevant radiological features.

### Usage:

```bash
python generate_cxr.py
```

By default, this will:
- Load the test dataset from `/home/psaha03/scratch/complete_dataset/`
- Process images with RAD-DINO
- Save embeddings to `/home/psaha03/scratch/results/rad_dino/embeddings/`
- Generate a CSV file with paths to embeddings at `/home/psaha03/scratch/results/rad_dino/rad_dino_features.csv`

### How it works:

1. The script loads chest X-ray images
2. Processes them through the RAD-DINO model
3. Extracts the image embeddings (feature vectors)
4. Saves these embeddings to disk for later use
5. Creates a CSV mapping patient UIDs to embedding file paths

## Report Generation with Decoder Model

The `decode_embeddings.py` script takes the RAD-DINO embeddings and generates human-readable reports using a text decoder model.

### Usage:

```bash
python decode_embeddings.py --embeddings_csv path/to/embeddings.csv --decoder_model model_name
```

Arguments:
- `--embeddings_csv`: Path to the CSV file with embedding paths (default: results from the extraction step)
- `--output_csv`: Path to save generated reports (default: `/home/psaha03/scratch/results/rad_dino/generated_reports_from_rad_dino.csv`)
- `--decoder_model`: Hugging Face model name for the decoder (default: `gpt2`)

### How it works:

1. The script loads the RAD-DINO embeddings for each image
2. Passes these embeddings to a decoder model
3. The decoder model generates a natural language report
4. Results are saved to a CSV file
5. ROUGE scores are calculated to evaluate the quality of the generated reports

## Workflow

The complete workflow consists of:

1. **Extract Features**:
   - Run `generate_cxr.py` to extract RAD-DINO features from images

2. **Generate Reports**:
   - Run `decode_embeddings.py` to generate reports from the extracted features

3. **Evaluate Results**:
   - Examine the ROUGE scores
   - Review the generated reports compared to ground truth

## Model Information

### RAD-DINO

RAD-DINO is a self-supervised vision model pre-trained on a large corpus of radiological images. It's particularly effective at extracting meaningful features from chest X-rays without requiring labeled data.

- Model: `microsoft/rad-dino`
- Type: Vision encoder
- Output: High-dimensional embedding vectors

### Decoder Model

The decoder transforms the visual embeddings into human-readable text reports. You can experiment with different decoder architectures:

- Basic option: GPT-2 (general purpose text model)
- Better options for medical text: medical domain-specific models like Clinical-T5, Med-PaLM, etc.

## Tips for Best Results

1. **Choose the right decoder**: Medical-domain specific decoders will likely perform better than general-purpose text models
2. **Fine-tune the decoder**: Fine-tuning on a dataset of radiology reports can significantly improve results
3. **Experiment with prompts**: Different prompting strategies can guide the decoder toward more accurate reporting
4. **Post-processing**: Consider adding post-processing steps to improve report formatting and content