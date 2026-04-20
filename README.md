# Vision Transformer from Scratch

This repository contains a clean, educational implementation of the Vision Transformer (ViT) architecture, as described in the paper [**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020).

The implementation is designed to be as faithful as possible to the original paper.

## Overview

The Vision Transformer (ViT) applies the standard Transformer architecture directly to images. Instead of using convolutional layers, it breaks an image into fixed-size patches, flattens them, and linearly projects them into a sequence of embeddings, which are then processed by a series of Transformer Encoder blocks.

## Repository Structure

- `model.py`: The core ViT architecture (Patch Embedding, Attention, Transformer Blocks, Classification Head).
- `train.py`: Training script for CIFAR-10 with AMP, AdamW, and a warmup + cosine LR schedule.
- `inference.py`: Script to load a trained checkpoint and run predictions.
- `inference.ipynb`: Jupyter Notebook for interactive visualization, confidence analysis, and patch inspection.
- `requirements.txt`: List of dependencies.
- `checkpoints/`: Directory where trained models are saved.

## Architecture Details

The implementation follows the **Pre-Norm** variant described in §3 and Appendix B of the paper:

1. **Patch Embedding**: A 2D convolution (`kernel=stride=patch_size`) linearly projects flattened patches into the hidden dimension `D`.
2. **[CLS] Token**: A learnable parameter prepended to the patch sequence (§3, Eq. 1).
3. **Position Embedding**: Learnable 1D embeddings added to all tokens, including the [CLS] token (§3, Eq. 1).
4. **Transformer Encoder** (repeated `L` times):
   - Layer Normalization applied *before* each sub-block (Pre-Norm).
   - Multi-Head Self-Attention (MHSA) with residual connection.
   - MLP block (two linear layers, GELU activation, dropout) with residual connection.
5. **Classification Head** (§3):
   - **Pre-training / training from scratch** (`head_type='pretrain'`): MLP with one hidden layer and `Tanh` activation — as specified in the paper.
   - **Fine-tuning** (`head_type='finetune'`): Single linear layer — used when adapting a pre-trained checkpoint to a new dataset.
6. **Weight Initialization** (Appendix B): All weight matrices initialized with truncated normal `N(0, 0.02)`. All biases initialized to zero. The [CLS] token and position embeddings are also initialized with `N(0, 0.02)`.

### Default Configuration (ViT-Tiny for CIFAR-10)

| Hyperparameter | Value |
|---|---|
| Image size | 32×32 |
| Patch size | 4×4 |
| Hidden dim (`D`) | 192 |
| Layers (`L`) | 12 |
| Heads | 3 |
| MLP dim | 768 |
| Dropout | 0.1 |
| Parameters | ~5.4M |

For the standard **ViT-Base** configuration from the paper (`image_size=224, patch_size=16, hidden_dim=768, num_layers=12, num_heads=12, mlp_dim=3072`).

## Training

To train the model on CIFAR-10:

```bash
python train.py
```

The training script uses:
- **Optimizer**: AdamW with `β₁=0.9`, `β₂=0.999`, `weight_decay=0.05` (Appendix B).
- **LR Schedule**: Linear warmup for the first 10 epochs, followed by cosine annealing decay (Appendix B).
- **Mixed Precision**: AMP (`torch.cuda.amp`) for faster training on GPU.

## Inference

Once you have a trained checkpoint (`checkpoints/vit_best.pth`):

```bash
python inference.py
```

For a more interactive experience, use the provided Jupyter Notebook:

```bash
jupyter notebook inference.ipynb
```

The notebook includes:
- Visual inspection of test samples (Ground Truth vs. Prediction).
- Confidence probability charts for individual images.
- **Patch Visualization**: See how the model decomposes an image into a grid of patches.
- Per-class accuracy breakdown.

### Usage in Code

```python
import torch
from model import VisionTransformer

# Initialize with the pre-training head (default, used for training from scratch)
model = VisionTransformer(num_classes=10, head_type='pretrain')
model.load_state_dict(torch.load("checkpoints/vit_best.pth"))
model.eval()

# Run prediction
with torch.no_grad():
    logits = model(image_tensor)  # image_tensor: (B, 3, 32, 32)
    prediction = logits.argmax(dim=-1)
```

To fine-tune a pre-trained checkpoint on a new dataset, swap the head:

```python
# Load pre-trained weights, then replace the head for fine-tuning
model = VisionTransformer(num_classes=10, head_type='pretrain')
model.load_state_dict(torch.load("checkpoints/vit_best.pth"))

# Reinitialize as fine-tuning head
model.head = torch.nn.Linear(192, new_num_classes)
```

## Results

After 100 epochs of training on CIFAR-10:
- **Parameters**: ~5.4M
- **Training Time**: ~2 minutes/epoch on RTX 5060 Ti
