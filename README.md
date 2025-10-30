# TinyMVLM: Minimal Vision-Language Model (CLIP-Style)

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
git clone https://github.com/Iro96/tinyMVLM.git
cd tinyMVLM
pip install -r requirements.txt
````

(Optional: you can omit `datasets` if using only local data.)

---

## Usage

### 1. Train on Flickr30k

```bash
python tiny_mvlm.py --dataset AnyModal/flickr30k --batch_size 64 --epochs 3 --out_dir ./checkpoints
```

or with fewer samples and workers:

```bash
python tiny_mvlm.py \
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
python tiny_mvlm.py --dataset local --local_path ./pairs.tsv --batch_size 32 --epochs 5
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


$$
\mathcal{L} = 
\frac{1}{2} 
\Bigg[
-\frac{1}{N} \sum_{i=1}^{N} 
\log
\frac{
\exp\left(\text{sim}(v_i, t_i) / \tau\right)
}{
\sum_{j=1}^{N} \exp\left(\text{sim}(v_i, t_j) / \tau\right)
}
\-\
\frac{1}{N} \sum_{i=1}^{N}
\log
\frac{
\exp\left(\text{sim}(t_i, v_i) / \tau\right)
}{
\sum_{j=1}^{N} \exp\left(\text{sim}(t_i, v_j) / \tau\right)
}
\Bigg]
$$

Where:

- $v_i$ = image embedding for sample $i$
- $t_i$ = text embedding for sample i
- $\text{sim}(v_i, t_j) = v_i^\top t_j$ is cosine-scaled similarity
- $\tau = e^{-\mathrm{logit\_scale}}$ is a learnable `logit_scale` parameter controls similarity temperature.
- $N$ = batch size


This loss aligns matched image–text pairs while pushing apart mismatched ones, averaged over both directions (image→text and text→image).

---

## File Structure

```
tiny_mvlm.py           # Main training script and model definition
checkpoints/          # Saved model weights (created automatically)
pairs.tsv             # Example local dataset file (optional)
README.md             # This file
```

---

## Checkpoints

Model checkpoints are automatically saved after each epoch (or every `--save_every` epochs):

```
checkpoints/tiny_mvlm_epoch1.pt
checkpoints/tiny_mvlm_epoch2.pt
...
```

Each checkpoint contains:

* `model_state_dict`
* `optimizer_state_dict`
* `epoch`

---

## Testing

### 1. Quick Test

After training, the script prints a **similarity matrix** for the first few image–text pairs:

```
Similarity matrix (first 8x8):
[[1.00, 0.23, 0.18, ...],
 [0.20, 0.95, 0.15, ...],
 ...]
```

Diagonal dominance → good alignment between image–text pairs

### 2. Test full pretrained model
```bash
python test_tiny_mvlm.py --checkpoint ./checkpoints/tiny_mvlm_epoch*.pt --image ./path/to/img --captions "text" "text" "text"
```

Example ouput:

<div align="center">
  <img width="50%" height="50%" alt="image" src="https://github.com/Iro96/tinyMVLM/blob/main/assets/cars_crash.png">
  <img width="50%" height="50%" alt="image" src="https://github.com/user-attachments/assets/4a2e6c4e-6d57-41da-b8db-bb30f07c4109" />
</div>

---

## Example: Custom Models

You can swap out encoders easily:

```bash
python tiny_mvlm.py \
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
