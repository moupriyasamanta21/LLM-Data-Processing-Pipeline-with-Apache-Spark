"""
evaluate.py
-----------
Loads a handful of DeepSpeed checkpoints saved during a training run (e.g.
early/mid/late steps), reconstructs a plain single-process PyTorch model
from each checkpoint's model state, and measures:

  1. Perplexity on the held-out validation blocks (data/processed/val_blocks.pt)
     = exp(mean cross-entropy loss over all validation blocks), the standard
     definition.
  2. A short greedy text continuation from a fixed prompt, decoded with the
     char-level vocab from data/processed/meta.json.

Results are written to results/evaluation.json.

Note on loading DeepSpeed checkpoints outside DeepSpeed: a ZeRO stage-1/2
checkpoint's `mp_rank_00_model_states.pt` file already contains the *full*,
un-partitioned model `state_dict` (only optimizer state is partitioned across
ranks in stage 1/2), so we can load it directly into a plain nn.Module for
evaluation without needing DeepSpeed or a distributed process group. This
is the standard way to run inference/eval after ZeRO-1/2 training.
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig


def load_model_from_ckpt(ckpt_dir, tag, cfg):
    model = GPT(cfg)
    state_path = os.path.join(ckpt_dir, tag, "mp_rank_00_model_states.pt")
    blob = torch.load(state_path, map_location="cpu", weights_only=False)
    state_dict = blob["module"] if "module" in blob else blob
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def compute_perplexity(model, val_blocks, batch_size=32):
    total_loss = 0.0
    total_batches = 0
    n = val_blocks.shape[0]
    for i in range(0, n, batch_size):
        batch = val_blocks[i:i + batch_size]
        x = batch[:, :-1]
        y = batch[:, 1:]
        _, loss = model(x, targets=y)
        total_loss += loss.item()
        total_batches += 1
    mean_loss = total_loss / total_batches
    ppl = float(torch.exp(torch.tensor(mean_loss)))
    return mean_loss, ppl


@torch.no_grad()
def generate_sample(model, stoi, itos, prompt, max_new_tokens=200, temperature=0.8, top_k=20, seed=1337):
    torch.manual_seed(seed)
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    ids = out[0].tolist()
    text = "".join(itos[str(i)] for i in ids)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/zero1_eval",
                         help="Directory containing DeepSpeed checkpoint tags (e.g. step50, step150, step300).")
    parser.add_argument("--tags", type=str, default="step50,step150,step300")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--out", type=str, default="results/evaluation.json")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]
    vocab_size = meta["vocab_size"]
    block_size = meta["block_size"]

    val_blocks = torch.load(os.path.join(args.data_dir, "val_blocks.pt"))

    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=0.0,
    )

    results = []
    for tag in args.tags.split(","):
        tag = tag.strip()
        model = load_model_from_ckpt(args.ckpt_dir, tag, cfg)
        mean_loss, ppl = compute_perplexity(model, val_blocks)
        sample = generate_sample(model, stoi, itos, args.prompt)
        entry = {
            "checkpoint_tag": tag,
            "val_mean_cross_entropy": mean_loss,
            "val_perplexity": ppl,
            "prompt": args.prompt,
            "generated_sample": sample,
        }
        results.append(entry)
        print(f"[{tag}] val_loss={mean_loss:.4f} val_ppl={ppl:.3f}")
        print(f"  sample: {sample!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ckpt_dir": args.ckpt_dir, "prompt": args.prompt, "results": results}, f, indent=2)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
