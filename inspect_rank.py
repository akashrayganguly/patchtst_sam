"""
Sanity-check that channel attention isn't rank-collapsed.

Usage:
    python inspect_rank.py --checkpoint ./checkpoints/<setting>/checkpoint.pth \
                           --data_path dk1.csv --enc_in 3 --channel_attn 1

What it does:
  1. Rebuilds the model with the same args you trained with.
  2. Enables `store_attn=True` on every encoder layer so attention weights
     are cached on the forward pass.
  3. Runs one validation batch.
  4. For each layer, computes the (approximate) rank of the attention
     matrix averaged over batch and heads.

What to look for:
  - Channel attention over M=3 channels has a maximum possible rank of 3.
  - Healthy training: rank ≈ 2–3 across layers.
  - Rank-collapsed: rank ≈ 1 (one channel attends to everything, all
    other rows of the softmax are near-identical). This is the failure
    mode SAM is supposed to prevent. If you see it with SAM on, raise
    rho; if you see it with SAM off, that's the expected SAMformer
    motivation — turn SAM on.
"""

import argparse
import os
import sys
import torch
import numpy as np

# Reuse the project's argument parser via Exp_Main + a synthetic args.
# Easiest path: import the same modules the training script imports.
from data_provider.data_factory import data_provider
from models import PatchTST


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--root_path', type=str, default='./dataset/')
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--data', type=str, default='custom')
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--target', type=str, default='OffshoreWindPower')
    p.add_argument('--freq', type=str, default='h')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--seq_len', type=int, default=336)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--enc_in', type=int, default=3)
    p.add_argument('--c_out', type=int, default=3)
    p.add_argument('--channel_attn', type=int, default=1)
    p.add_argument('--n_heads', type=int, default=1)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--d_ff', type=int, default=512)
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--padding_patch', type=str, default='end')
    p.add_argument('--revin', type=int, default=1)
    p.add_argument('--affine', type=int, default=0)
    p.add_argument('--subtract_last', type=int, default=0)
    p.add_argument('--decomposition', type=int, default=0)
    p.add_argument('--kernel_size', type=int, default=25)
    p.add_argument('--individual', type=int, default=0)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--fc_dropout', type=float, default=0.2)
    p.add_argument('--head_dropout', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--rank_tol', type=float, default=1e-2,
                   help='Singular values below tol * sigma_max are treated as zero')
    return p.parse_args()


def enable_store_attn(model):
    """Walk into the PatchTST encoder layers and turn on store_attn."""
    # model.model is the PatchTST_backbone (no decomposition case)
    backbone = model.model
    for layer in backbone.backbone.encoder.layers:
        layer.store_attn = True
    return backbone


def attn_rank(A, tol):
    """
    A: [bs, n_heads, q_len, q_len]  (softmaxed attention)
    Returns mean approximate rank across batch and heads.
    """
    A = A.detach().float().cpu()
    bs, h, q, _ = A.shape
    ranks = []
    for b in range(bs):
        for hi in range(h):
            s = torch.linalg.svdvals(A[b, hi])
            r = (s > tol * s.max()).sum().item()
            ranks.append(r)
    return float(np.mean(ranks)), float(np.std(ranks)), int(q)


def main():
    args = parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model with same shape as training
    model = PatchTST.Model(args).float().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    backbone = enable_store_attn(model)

    # One batch from val set
    _, loader = data_provider(args, flag='val')
    batch_x, batch_y, *_ = next(iter(loader))
    batch_x = batch_x.float().to(device)

    with torch.no_grad():
        _ = model(batch_x)

    print(f"\nChannel attention rank check  (channel_attn={args.channel_attn}, M={args.enc_in})")
    print(f"Max possible rank per attn matrix = q_len = {args.enc_in if args.channel_attn else 'patch_num'}")
    print("-" * 60)

    for i, layer in enumerate(backbone.backbone.encoder.layers):
        if not hasattr(layer, 'attn') or layer.attn is None:
            print(f"Layer {i}: no attn cached (did store_attn fire?)")
            continue
        mean_r, std_r, q_len = attn_rank(layer.attn, args.rank_tol)
        bar = "#" * int(round(mean_r * 4))
        print(f"Layer {i}: q_len={q_len}  rank ≈ {mean_r:5.2f} ± {std_r:.2f}   {bar}")

    print("-" * 60)
    print("Healthy (M=3 channels): rank ≈ 2.5–3.0 per layer.")
    print("Collapsed:               rank ≈ 1.0 (raise rho, check RevIN is on).")


if __name__ == '__main__':
    main()
