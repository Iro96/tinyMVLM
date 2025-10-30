# TinyVLM — Minimal Vision-Language Model (CLIP-Style)

**TinyVLM** is a lightweight, educational implementation of a **CLIP-style contrastive training loop** built with  
a **ViT-Tiny** image encoder (via `timm`) and a **MiniLM** text encoder (via `transformers`).  
It trains image–text alignment models similar to [CLIP](https://openai.com/research/clip) using modest resources.

---

## Features
- Contrastive multimodal training (image ↔ text)
- Vision backbone: [ViT-Tiny](https://huggingface.co/docs/timm)
- Text backbone: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Supports **Flickr30k** (via `datasets`) or **custom local TSV/CSV** data
- Modular PyTorch implementation — easy to extend or modify
- Learnable logit scaling parameter (like CLIP)
- Lightweight enough to run on a single GPU or CPU

---

## Installation

```bash
git clone https://github.com/yourusername/tiny_vlm.git
cd tiny_vlm
pip install torch torchvision timm transformers datasets pillow
````

(Optional: you can omit `datasets` if using only local data.)

---

## Usage

### 1. Train on Flickr30k

```bash
python tiny_vlm.py --dataset AnyModal/flickr30k --batch_size 64 --epochs 3 --out_dir ./checkpoints
```

or with fewer samples and workers:

```bash
python tiny_vlm.py \
  --dataset AnyModal/flickr30k \
  --batch_size 16 \
  --num_workers 1 \
  --samples 7000 \
  --epochs 3 \
  --out_dir ./checkpoints
```

---

### 2. Train on a Local Dataset

Create a `pairs.tsv` file with **tab-separated** columns:

```
path/to/image1.jpg    a cat sitting on the sofa
path/to/image2.png    a car parked on the street
```

Then run:

```bash
python tiny_vlm.py --dataset local --local_path ./pairs.tsv --batch_size 32 --epochs 5
```

Optional arguments:

* `--image_root ./images` — prefix for relative image paths
* `--embed_dim 256` — shared embedding dimension
* `--lr 2e-5` — learning rate

---

## Key Arguments

| Argument             | Description                                   | Default                |
| -------------------- | --------------------------------------------- | ---------------------- |
| `--dataset`          | `"AnyModal/flickr30k"` or `"local"`           | `flickr30k`            |
| `--local_path`       | Path to TSV/CSV with `image_path<TAB>caption` | `None`                 |
| `--image_root`       | Base directory for image paths                | `.`                    |
| `--image_model_name` | timm model name                               | `vit_tiny_patch16_224` |
| `--model_text`       | HuggingFace text model                        | `all-MiniLM-L6-v2`     |
| `--embed_dim`        | Shared projection dimension                   | `256`                  |
| `--epochs`           | Number of training epochs                     | `3`                    |
| `--batch_size`       | Mini-batch size                               | `32`                   |
| `--lr`               | Learning rate                                 | `2e-5`                 |
| `--out_dir`          | Checkpoint output directory                   | `./checkpoints`        |

---

## Model Overview

TinyVLM follows the **CLIP** paradigm:

1. Encode images using a small ViT backbone.
2. Encode text using a compact transformer.
3. Project both embeddings into a shared latent space.
4. Train contrastively using an **InfoNCE** loss:

[
\mathcal{L} = \frac{1}{2} [CE(\text{img→txt}) + CE(\text{txt→img})]
]

A learnable `logit_scale` parameter controls similarity temperature.

---

## File Structure

```
tiny_vlm.py           # Main training script and model definition
checkpoints/          # Saved model weights (created automatically)
pairs.tsv             # Example local dataset file (optional)
README.md             # This file
```

---

## Checkpoints

Model checkpoints are automatically saved after each epoch (or every `--save_every` epochs):

```
checkpoints/tiny_vlm_epoch1.pt
checkpoints/tiny_vlm_epoch2.pt
...
```

Each checkpoint contains:

* `model_state_dict`
* `optimizer_state_dict`
* `epoch`

---

## Quick Test

After training, the script prints a **similarity matrix** for the first few image–text pairs:

```
Similarity matrix (first 8x8):
[[1.00, 0.23, 0.18, ...],
 [0.20, 0.95, 0.15, ...],
 ...]
```

Diagonal dominance → good alignment between image–text pairs

---

## Example: Custom Models

You can swap out encoders easily:

```bash
python tiny_vlm.py \
  --model_text sentence-transformers/all-MiniLM-L12-v2 \
  --image_model_name resnet18
```

---

## References

* [CLIP: Learning Transferable Visual Models From Natural Language Supervision (OpenAI, 2021)](https://arxiv.org/abs/2103.00020)
* [timm library](https://github.com/huggingface/pytorch-image-models)
* [HuggingFace Transformers](https://huggingface.co/docs/transformers)
* [Flickr30k Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

---

## License

[MIT License © 2025](https://github.com/Iro96/tinyMVLM/edit/main/LICENSE)
Created for research, learning, and lightweight multimodal prototyping.
