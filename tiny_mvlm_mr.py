"""
TinyMVLM-MR (Merging-Aware, Research Logging Edition)
-------------------------------------------------
CLIP-style contrastive model merging pretrained MiniLM (text) and ViT-Tiny (image)
using model-merging techniques (feature alignment, task-vector fusion, norm preservation).

Automatically logs training statistics to a JSON file for later analysis / visualization.

Example:
  python tiny_mvlm.py --dataset AnyModal/flickr30k --epochs 3
"""

import os, math, json, argparse, time
from pathlib import Path
from typing import Optional
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from PIL import Image
import timm

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except Exception:
    HF_DATASETS_AVAILABLE = False


# ---------------------------
# Args
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="AnyModal/flickr30k", choices=["AnyModal/flickr30k", "local"])
    p.add_argument("--local_path", type=str, default=None)
    p.add_argument("--image_root", type=str, default=".")
    p.add_argument("--model_text", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--image_model_name", type=str, default="vit_tiny_patch16_224")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="./checkpoints")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    return p.parse_args()


# ---------------------------
# Dataset
# ---------------------------
class LocalPairsDataset(Dataset):
    def __init__(self, pairs_path, image_root, tokenizer, transform):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
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

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        full = os.path.join(self.image_root, img_path)
        image = self.transform(Image.open(full).convert("RGB"))
        t = self.tokenizer(caption, truncation=True, padding="max_length", max_length=16, return_tensors="pt")
        return image, t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0), caption


class HFFlickr30kDataset(Dataset):
    def __init__(self, split, tokenizer, transform, max_samples=None):
        ds = load_dataset("AnyModal/flickr30k", split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.data = []
        for ex in ds:
            cap = ex.get("alt_text") or ex.get("caption") or (ex["sentence"][0] if "sentence" in ex else None)
            img = ex.get("image")
            if img and cap:
                self.data.append((img, cap))
        self.tokenizer, self.transform = tokenizer, transform

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img, cap = self.data[idx]
        img = self.transform(img.convert("RGB"))
        t = self.tokenizer(cap, truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        return img, t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0), cap


def collate_fn(batch):
    imgs, ids, masks, caps = zip(*batch)
    return torch.stack(imgs), torch.stack(ids), torch.stack(masks), caps


# ---------------------------
# Model
# ---------------------------
class MergedVLM(nn.Module):
    def __init__(self, text_model_name, image_model_name, embed_dim=256, alpha=0.5, beta=0.5):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0, global_pool="avg")

        txt_dim = self.text_encoder.config.hidden_size
        img_dim = self.image_encoder.num_features
        self.text_proj = nn.Linear(txt_dim, embed_dim)
        self.image_proj = nn.Linear(img_dim, embed_dim)
        self.base_proj = nn.Linear(embed_dim, embed_dim)
        self.alpha, self.beta = alpha, beta
        self.merged_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_text = nn.LayerNorm(txt_dim)
        self.norm_img = nn.LayerNorm(img_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self._init_merge()

    def _init_merge(self):
        with torch.no_grad():
            base_w = self.base_proj.weight
            text_w = self.text_proj.weight
            image_w = self.image_proj.weight

            # Pad weights to the maximum dimension for merging
            max_dim = max(base_w.shape[1], text_w.shape[1], image_w.shape[1])
        
            def pad_to_max(w, max_d):
                padding = max_d - w.shape[1]
                if padding > 0:
                    return F.pad(w, (0, padding))
                return w

            base_padded = pad_to_max(base_w, max_dim)
            text_padded = pad_to_max(text_w, max_dim)
            image_padded = pad_to_max(image_w, max_dim)

            dt = text_padded - base_padded
            di = image_padded - base_padded
        
            merged = base_padded + self.alpha * dt + self.beta * di
        
            # Truncate back to the merged_proj's dimension
            merged_final = merged[:, :self.merged_proj.weight.shape[1]]
        
            merged_final /= merged_final.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self.merged_proj.weight.copy_(merged_final)

    def encode_text(self, ids, mask):
        out = self.text_encoder(input_ids=ids, attention_mask=mask, return_dict=True)
        last = out.last_hidden_state
        mask = mask.unsqueeze(-1).to(last.dtype)
        pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        return self.norm_text(pooled)

    def encode_image(self, imgs):
        f = self.image_encoder.forward_features(imgs)
        if f.ndim > 2: f = f.mean(1)
        return self.norm_img(f)

    def forward(self, imgs, ids, mask):
        t_feat = self.encode_text(ids, mask)
        v_feat = self.encode_image(imgs)
        t_emb = F.normalize(self.merged_proj(self.text_proj(t_feat)), dim=-1)
        v_emb = F.normalize(self.merged_proj(self.image_proj(v_feat)), dim=-1)
        return v_emb, t_emb, self.logit_scale.exp()


# ---------------------------
# Training
# ---------------------------
def contrastive_loss(v_emb, t_emb, scale):
    logits_i = scale * v_emb @ t_emb.t()
    logits_t = logits_i.t()
    labels = torch.arange(v_emb.size(0), device=v_emb.device)
    li = F.cross_entropy(logits_i, labels)
    lt = F.cross_entropy(logits_t, labels)
    return (li + lt) / 2


def train_one_epoch(model, dl, opt, epoch, device):
    model.train()
    total = 0
    for i, (imgs, ids, mask, _) in enumerate(dl, 1):
        imgs, ids, mask = imgs.to(device), ids.to(device), mask.to(device)
        opt.zero_grad()
        v, t, s = model(imgs, ids, mask)
        loss = contrastive_loss(v, t, s)
        loss.backward()
        opt.step()
        total += loss.item()
        if i % 50 == 0:
            print(f"Epoch {epoch} step {i}: loss={loss.item():.4f}")
    avg = total / len(dl)
    print(f"Epoch {epoch} avg loss={avg:.4f}")
    return avg


def save_ckpt(model, opt, epoch, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, f"merged_vlm_epoch{epoch}.pt")
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch
    }, path)
    print("Saved:", path)
    return path


# ---------------------------
# Main
# ---------------------------
def main():
    a = parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tok = AutoTokenizer.from_pretrained(a.model_text)
    model = MergedVLM(a.model_text, a.image_model_name, a.embed_dim, a.alpha, a.beta).to(device)

    # Dataset
    if a.dataset == "AnyModal/flickr30k":
        if not HF_DATASETS_AVAILABLE:
            raise RuntimeError("Install datasets library.")
        ds = HFFlickr30kDataset("train", tok, tfm, max_samples=a.samples)
    else:
        ds = LocalPairsDataset(a.local_path, a.image_root, tok, tfm)

    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers, collate_fn=collate_fn)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    # ---- Training + Logging ----
    start_time = time.time()
    epoch_losses = []
    for e in range(1, a.epochs + 1):
        avg_loss = train_one_epoch(model, dl, opt, e, device)
        epoch_losses.append(avg_loss)
        if e % a.save_every == 0:
            ckpt_path = save_ckpt(model, opt, e, a.out_dir)

    total_time = round(time.time() - start_time, 2)
    final_ckpt = save_ckpt(model, opt, a.epochs, a.out_dir)

    # ---- Save Research Status JSON ----
    status = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_text": a.model_text,
        "image_model": a.image_model_name,
        "embed_dim": a.embed_dim,
        "alpha": a.alpha,
        "beta": a.beta,
        "dataset": a.dataset,
        "samples": a.samples,
        "epochs": a.epochs,
        "batch_size": a.batch_size,
        "lr": a.lr,
        "weight_decay": a.weight_decay,
        "avg_losses": epoch_losses,
        "final_loss": epoch_losses[-1],
        "final_logit_scale": float(model.logit_scale.exp().item()),
        "training_time_sec": total_time,
        "final_checkpoint": final_ckpt
    }

    summary_path = os.path.join(a.out_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(f"\nTraining complete. Summary saved to: {summary_path}\n")


if __name__ == "__main__":
    main()
