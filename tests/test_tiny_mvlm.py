#!/usr/bin/env python3
# test_tiny_mvlm.py
"""
Test TinyMVLM pretrained model (.pt checkpoint)

Usage:
  python test_tiny_vlm.py \
      --checkpoint ./checkpoints/tiny_vlm_epoch3.pt \
      --image test.jpg \
      --captions "A man riding a horse on the beach." "A small dog playing in the park." \
      --device auto
"""

import torch
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
import argparse
from tiny_mvlm import TinyVLM


def parse_args():
    p = argparse.ArgumentParser(description="TinyVLM Inference Script")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--image", type=str, required=True, help="Path to test image")
    p.add_argument("--captions", type=str, nargs="+", required=True, help="Text captions to compare")
    p.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--image_model", type=str, default="vit_tiny_patch16_224")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--device", type=str, default="auto", help="'cpu', 'cuda', or 'auto'")
    return p.parse_args()


def main():
    args = parse_args()
    device = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Using device: {device}")

    # Load model
    model = TinyVLM(args.text_model, args.image_model, args.embed_dim, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    # Image preprocessing (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load and encode image
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        img_feats = model.encode_image(image)
        img_emb = model.image_proj(img_feats)
        img_emb = torch.nn.functional.normalize(img_emb, dim=-1)

    # Encode captions
    text_inputs = tokenizer(
        args.captions,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        txt_feats = model.encode_text(text_inputs["input_ids"], text_inputs["attention_mask"])
        txt_emb = model.text_proj(txt_feats)
        txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)

    # Compute similarity (cosine similarity)
    sims = (img_emb @ txt_emb.T * model.logit_scale.exp()).squeeze(0).cpu().numpy()

    # Show results
    print("\n=== Image–Text Similarities ===")
    for cap, sim in zip(args.captions, sims):
        print(f"Caption: {cap}")
        print(f" → Similarity: {sim:.4f}\n")

    best_idx = sims.argmax()
    print(f"Best match: '{args.captions[best_idx]}' (score={sims[best_idx]:.4f})")


if __name__ == "__main__":
    main()
