#!/usr/bin/env python3
# tiny_vlm.py
"""
Tiny VLM (Vision-Language Model) — CLIP-style contrastive training with
ViT-Tiny (timm) + MiniLM (HuggingFace).

Usage examples:
  # Train on Flickr30k (downloads dataset)
  python tiny_vlm.py --dataset AnyModal/flickr30k --batch_size 64 --epochs 3 --out_dir ./checkpoints

  # Train using a local TSV/CSV with columns: image_path\tcaption
  python tiny_vlm.py --dataset local --local_path ./pairs.tsv --batch_size 32 --epochs 5
"""

import os
import math
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from PIL import Image
import timm

# Optional dataset library for Flickr30k / COCO
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except Exception:
    HF_DATASETS_AVAILABLE = False


# ---------------------------
# Config / Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="AnyModal/flickr30k",
                   choices=["AnyModal/flickr30k", "local"], help="Use 'flickr30k' or 'local'")
    p.add_argument("--local_path", type=str, default=None,
                   help="If dataset=local, path to TSV/CSV with image_path<TAB>caption")
    p.add_argument("--image_root", type=str, default=".",
                   help="If local dataset, prefix for image paths")
    p.add_argument("--model_text", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HuggingFace text encoder (small).")
    p.add_argument("--image_model_name", type=str, default="vit_tiny_patch16_224",
                   help="timm image model name.")
    p.add_argument("--embed_dim", type=int, default=256, help="Shared projection dimension")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (auto detect if None)")
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=1)
    return p.parse_args()


# ---------------------------
# Dataset wrappers
# ---------------------------
class LocalPairsDataset(Dataset):
    def __init__(self, pairs_path: str, image_root: str, tokenizer, transform):
        # Expect TSV/CSV with two columns: image_path, caption
        self.pairs = []
        with open(pairs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    img, cap = line.split("\t", 1)
                else:
                    img, cap = line.split(",", 1)
                self.pairs.append((img.strip(), cap.strip()))
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            img_path, caption = self.pairs[idx]
            full_path = os.path.join(self.image_root, img_path)
            image = Image.open(full_path).convert("RGB")
            image = self.transform(image)
            text_inputs = self.tokenizer(
                caption, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
            )
            return image, text_inputs["input_ids"].squeeze(0), text_inputs["attention_mask"].squeeze(0), caption
        except Exception as e:
            print(f"Warning: Skipping problematic sample at index {idx}. Error: {e}")
            return None


class HFFlickr30kDataset(Dataset):
    """
    Works with AnyModal/flickr30k or similar datasets.
    Handles various caption field names: alt_text, caption, sentence.
    """
    def __init__(self, split: str, tokenizer, transform, max_samples: Optional[int] = None):
        self.ds = load_dataset("AnyModal/flickr30k", split=split, streaming=False)
        if max_samples is not None:
            self.ds = self.ds.select(range(min(max_samples, len(self.ds))))

        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        # Detect caption field automatically
        if "alt_text" in ex:
            caption = ex["alt_text"]
        elif "caption" in ex:
            caption = ex["caption"]
        elif "sentence" in ex:
            caption = ex["sentence"][0] if isinstance(ex["sentence"], list) else ex["sentence"]
        else:
            raise KeyError(f"Cannot find a caption field in example keys: {list(ex.keys())}")

        if "image" not in ex:
            raise KeyError(f"Cannot find 'image' field in example keys: {list(ex.keys())}")

        img = ex["image"]
        image = self.transform(img.convert("RGB"))
        text_inputs = self.tokenizer(
            caption, truncation=True, padding="max_length", max_length=32, return_tensors="pt"
        )
        return image, text_inputs["input_ids"].squeeze(0), text_inputs["attention_mask"].squeeze(0), caption


# Collate function
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, input_ids, attention_masks, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return images, input_ids, attention_masks, captions


# ---------------------------
# Model: TinyVLM
# ---------------------------
class TinyVLM(nn.Module):
    def __init__(self, text_model_name: str, image_model_name: str, embed_dim: int = 256, device: str = "cpu"):
        super().__init__()
        self.device = device

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Image encoder (timm)
        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0, global_pool="avg")
        img_feat_dim = self.image_encoder.num_features if hasattr(self.image_encoder, "num_features") else 768
        txt_feat_dim = self.text_encoder.config.hidden_size

        # Projection heads
        self.image_proj = nn.Sequential(
            nn.Linear(img_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(txt_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = torch.sum(last_hidden * mask, dim=1)
        lengths = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        pooled = summed / lengths
        return pooled

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder.forward_features(images) if hasattr(self.image_encoder, "forward_features") else self.image_encoder(images)
        if feats.ndim > 2:
            feats = feats.mean(dim=1)
        return feats

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        img_feats = self.encode_image(images)
        txt_feats = self.encode_text(input_ids, attention_mask)
        img_emb = self.image_proj(img_feats)
        txt_emb = self.text_proj(txt_feats)
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        logit_scale = self.logit_scale.exp()
        return img_emb, txt_emb, logit_scale


# ---------------------------
# Loss: Contrastive InfoNCE
# ---------------------------
def contrastive_loss(img_emb, txt_emb, logit_scale):
    logits_per_image = logit_scale * img_emb @ txt_emb.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(img_emb.size(0), dtype=torch.long, device=img_emb.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# ---------------------------
# Training Loop
# ---------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    steps = 0
    for i, batch in enumerate(dataloader, 1):
        if batch is None:
            continue
        images, input_ids, attention_mask, _ = batch
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)

        optimizer.zero_grad()
        img_emb, txt_emb, logit_scale = model(images, input_ids, attention_mask)
        loss = contrastive_loss(img_emb, txt_emb, logit_scale)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        if i % 50 == 0:
            print(f"Epoch {epoch} Iter {i}: loss={loss.item():.4f}")

    avg_loss = total_loss / max(1, steps)
    print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
    return avg_loss


# ---------------------------
# Utilities
# ---------------------------
def save_checkpoint(model, optimizer, epoch, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"tiny_vlm_epoch{epoch}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, path)
    print("Saved checkpoint:", path)


def compute_embeddings(model, images, input_ids, attention_mask, device):
    model.eval()
    with torch.no_grad():
        images, input_ids, attention_mask = images.to(device), input_ids.to(device), attention_mask.to(device)
        img_emb, txt_emb, _ = model(images, input_ids, attention_mask)
    return img_emb.cpu(), txt_emb.cpu()


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading text tokenizer & model:", args.model_text)
    tokenizer = AutoTokenizer.from_pretrained(args.model_text)
    print("Loading TinyVLM model...")
    model = TinyVLM(args.model_text, args.image_model_name, args.embed_dim, device)
    model.to(device)

    # Dataset
    if args.dataset == "AnyModal/flickr30k":
        if not HF_DATASETS_AVAILABLE:
            raise RuntimeError("datasets library not available. Install it with `pip install datasets`.")
        print("Loading Flickr30k (HuggingFace)...")
        ds = HFFlickr30kDataset("train", tokenizer, transform, max_samples=args.samples)
    else:
        if not args.local_path:
            raise ValueError("local_path must be provided for local dataset")
        ds = LocalPairsDataset(args.local_path, args.image_root, tokenizer, transform)

    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, dataloader, optimizer, epoch, device)
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, args.out_dir)

    print("Training finished. Final checkpoint saved.")
    save_checkpoint(model, optimizer, args.epochs, args.out_dir)

    # Quick test
    # Find first non-empty batch
    batch = None
    for b in dataloader:
        if b is not None:
            batch = b
            break
    if batch is None:
        print("No valid batch for quick test; skipping similarity print.")
        return
    images, input_ids, attention_mask, _ = batch
    img_emb, txt_emb = compute_embeddings(model, images, input_ids, attention_mask, device)
    sims = (img_emb @ txt_emb.t()).numpy()
    print("Similarity matrix (first 8x8):")
    print(sims[:8, :8])


if __name__ == "__main__":
    main()
