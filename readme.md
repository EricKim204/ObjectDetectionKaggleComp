# CLIP Image Classification - Kaggle Competition Winner

Fine-tuning Vision Transformers (CLIP) for image classification, achieving **first place** in the university Kaggle competition.

## Overview

This project is a refined recreation of the notebook I built when I won my university’s image classification competition - Original Notebook link: https://colab.research.google.com/drive/1SrOHL5HgtcpHeZ43WVgTVEDugFBE1DY4?usp=sharing

I lost access to the original files and other various scripts due to my university email account being expired...

This is a custom classification training script using OpenAI’s CLIP (Contrastive Language–Image Pre‑training) model by applying transfer learning and modern training techniques.

### Key Features

- **CLIP Model Fine-tuning**: Utilizes pre-trained CLIP vision transformers
- **Data Augmentation**: RandAugment with optimized hyperparameters
- **Cosine Learning Rate Schedule**: Smooth convergence with warmup
- **Pseudo-Labeling**: Semi-supervised learning for improved performance
- **GPU Acceleration**: CUDA-enabled 

## Architecture

The model architecture consists of:

1. **Backbone**: CLIP Vision Transformer (ViT-Large-Patch14)
2. **Classification Head**: Fine-tuned for specific class predictions

## Results

- **Competition Rank**: 1st Place
- **Model**: `openai/clip-vit-large-patch14`
- **Image Size**: 64x64 pixels
- **Training Strategy**: Two-stage training with pseudo-labeling

### Training Details

**Stage 1**: Initial Fine-tuning
- Epochs: 4
- Learning Rate: 3e-6
- Batch Size: 8
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR

**Stage 2**: Pseudo-Label Refinement
- High-confidence predictions (>0.9) added to training set
- Epochs: 2
- Learning Rate: 1.5e-6 (reduced)

## Methodology

### Data Augmentation

**RandAugment** with magnitude=7, as recommended in the CLIP fine-tuning literature:
- Random horizontal flips
- Random rotations
- Color jittering
- Random cropping

### Learning Rate Schedule

Cosine annealing learning rate schedule provides:
- Smooth convergence
- Better generalization
- Prevents overfitting in later epochs

### Pseudo-Labeling Strategy

1. Train initial model on labeled data
2. Generate predictions on unlabeled test data
3. Select high-confidence predictions (>90% confidence)
4. Add pseudo-labeled samples to training set
5. Retrain model with enhanced dataset

This semi-supervised approach improved model robustness and accuracy.

## Hyperparameter Tuning

Key hyperparameters explored:

| Parameter | Tested Values | Final Choice |
|-----------|--------------|--------------|
| Model | ViT-Base, ViT-Large | ViT-Large |
| Learning Rate | 1e-6, 3e-6, 5e-6 | 3e-6 |
| Image Size | 32x32, 64x64, 224x224 | 64x64 |
| Batch Size | 4, 8, 16 | 8 |
| RandAugment Magnitude | 5, 7, 9 | 7 |

## Acknowledgments

- OpenAI for the CLIP model
- Hugging Face for the Transformers library
- Monash University Kaggle competition organizers
