"""
train.py
--------
Distributed training entry point. Loads the preprocessed char-level blocks,
builds the small GPT model from model.py, hands it to deepspeed.initialize()
with whichever ZeRO stage config is passed in, and runs a fixed number of
steps, logging loss and (at the end) wall-clock time, throughput and peak
resident memory for this rank.

Intended launch (from repo root, venv activated):

    python -m torch.distributed.run --nproc_per_node=<N> src/train.py \
        --ds_config ds_configs/zero1.json --steps 200

Each rank reads the same train_blocks.pt file and takes a different
deterministic shard of the blocks (striped by rank), so ranks see disjoint
data per step -- this is real (not simulated) data-parallel training, not
N copies of the same shard.
"""
import argparse
import json
import os
import resource
import time

import deepspeed
import torch

from model import GPT, GPTConfig


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--ckpt_dir", type=str, default=None,
                         help="If set, save DeepSpeed checkpoints here at ckpt_steps.")
    parser.add_argument("--ckpt_steps", type=str, default="",
                         help="Comma-separated step numbers at which to save a checkpoint.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--metrics_out", type=str, default=None,
                         help="If set (rank 0 only), write a JSON file with timing/loss/memory stats.")
    parser.add_argument("--local_rank", type=int, default=-1)  # accepted for deepspeed launcher compat
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = get_args()
    torch.manual_seed(args.seed)

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    block_size = meta["block_size"]

    train_blocks = torch.load(os.path.join(args.data_dir, "train_blocks.pt"))

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.1,
    )
    model = GPT(cfg)
    n_params = model.num_params()

    engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=args.ds_config,
    )

    rank = engine.local_rank if engine.local_rank is not None else 0
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    micro_bsz = engine.train_micro_batch_size_per_gpu()
    device = engine.device

    # Deterministic, disjoint sharding: rank r sees blocks[r::world_size].
    shard = train_blocks[global_rank::world_size]
    n_shard = shard.shape[0]

    ckpt_steps = set(int(s) for s in args.ckpt_steps.split(",") if s.strip())

    g = torch.Generator().manual_seed(args.seed + global_rank)

    def get_batch(step_idx):
        ix = torch.randint(0, n_shard, (micro_bsz,), generator=g)
        batch = shard[ix]  # (B, block_size+1)
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        return x, y

    losses = []
    t_start = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(step)
        _, loss = engine(x, targets=y)
        engine.backward(loss)
        engine.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if global_rank == 0 and (step % args.log_every == 0 or step == 1):
            print(f"[rank0] step {step}/{args.steps} loss {loss_val:.4f}", flush=True)

        if step in ckpt_steps and args.ckpt_dir is not None:
            tag = f"step{step}"
            engine.save_checkpoint(args.ckpt_dir, tag=tag)
            if global_rank == 0:
                print(f"[rank0] saved checkpoint at {tag}", flush=True)

    torch.distributed.barrier() if torch.distributed.is_initialized() else None
    elapsed = time.time() - t_start

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    tokens_this_rank = args.steps * micro_bsz * block_size
    tokens_per_sec_this_rank = tokens_this_rank / elapsed if elapsed > 0 else float("nan")

    print(
        f"[rank {global_rank}] DONE steps={args.steps} elapsed_sec={elapsed:.3f} "
        f"peak_rss_mb={peak_rss_kb / 1024:.1f} tokens_per_sec={tokens_per_sec_this_rank:.1f} "
        f"final_loss={losses[-1]:.4f} first_loss={losses[0]:.4f}",
        flush=True,
    )

    if args.metrics_out and global_rank == 0:
        result = {
            "ds_config": args.ds_config,
            "world_size": world_size,
            "steps": args.steps,
            "micro_batch_size_per_gpu": micro_bsz,
            "block_size": block_size,
            "n_params": n_params,
            "elapsed_sec_rank0": elapsed,
            "peak_rss_mb_rank0": peak_rss_kb / 1024,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "loss_trace_every_log": losses[:: args.log_every] + [losses[-1]],
        }
        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
