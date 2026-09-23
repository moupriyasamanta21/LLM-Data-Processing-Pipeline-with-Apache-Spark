"""
preprocess.py
-------------
Turns the raw tiny-shakespeare text into fixed-length blocks of character-level
token ids that the training loop in src/train.py can load directly.

Pipeline (run with local-mode Spark so the same code path would scale to a
real cluster read of many text shards, even though here it's one file):

  1. Read data/raw/input.txt into Spark as a one-row-per-line text DataFrame.
  2. Collect the distinct characters that appear anywhere in the file and
     build a deterministic char <-> id vocabulary (sorted so the mapping is
     reproducible across runs).
  3. Encode the full text (with '\n' put back between lines, since Spark's
     text reader drops line terminators) into a single 1-D array of ids.
  4. Chunk that id stream into non-overlapping windows of length
     block_size + 1. Each row of the output is one such window.
  5. Split rows 90/10 into train / validation and save each split as a single
     torch tensor file under data/processed/.

Framing note (documented per project convention): each saved block has
block_size + 1 tokens. The training loop is responsible for slicing
tokens[:-1] as the model input and tokens[1:] as the shifted next-token
target -- we do NOT pre-shift into separate input_ids/labels tensors here,
we store one contiguous window and shift in the dataloader/train step.
"""
import argparse
import json
import os
import time

from pyspark.sql import SparkSession
import torch


def build_vocab(chars):
    chars = sorted(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/input.txt")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("mini-zero-llm-preprocess")
        .config("spark.driver.memory", "4g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Read raw text as one Spark row per line (this is the part that would
    # parallelize across many shards/files on a real cluster).
    lines_df = spark.read.text(args.input)
    lines = lines_df.rdd.map(lambda row: row[0]).collect()
    # Spark's text source strips the trailing '\n' from every line; put it
    # back so re-joining reproduces the original character stream faithfully.
    full_text = "\n".join(lines)

    n_chars_total = len(full_text)
    unique_chars = lines_df.rdd.flatMap(lambda row: list(row[0])).distinct().collect()
    # '\n' never appears inside a Spark text row (it's the row delimiter), so
    # add it back explicitly before building the vocab.
    unique_chars = set(unique_chars) | {"\n"}
    stoi, itos = build_vocab(unique_chars)
    vocab_size = len(stoi)

    # Encode the whole stream to ids using a Spark job over line boundaries,
    # keeping the '\n' separators, then flatten back to one id sequence.
    def encode_line(line):
        ids = [stoi[c] for c in line]
        ids.append(stoi["\n"])  # restore the separator this line was split on
        return ids

    encoded_lines = lines_df.rdd.map(lambda row: encode_line(row[0])).collect()
    all_ids = []
    for ids in encoded_lines:
        all_ids.extend(ids)
    # Drop the extra trailing '\n' introduced after the very last line, which
    # doesn't exist in the source file if it wasn't newline-terminated.
    if not full_text.endswith("\n") and all_ids and all_ids[-1] == stoi["\n"]:
        all_ids = all_ids[:-1]

    n_tokens = len(all_ids)
    assert n_tokens == n_chars_total, f"token count {n_tokens} != char count {n_chars_total}"

    block_len = args.block_size + 1
    n_blocks = n_tokens // block_len
    usable = n_blocks * block_len
    ids_tensor = torch.tensor(all_ids[:usable], dtype=torch.long)
    blocks = ids_tensor.view(n_blocks, block_len)

    # Deterministic shuffle before the split so train/val aren't just
    # "first 90% of the play" / "last 10%" (which would leak very little
    # cross-scene structure into val but is still worth decorrelating).
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_blocks, generator=g)
    blocks = blocks[perm]

    n_val = int(n_blocks * args.val_fraction)
    n_train = n_blocks - n_val
    # .clone() (not .contiguous(), which is a no-op on an already-contiguous
    # slice and would silently keep the *whole* permuted tensor's storage
    # attached to each split) so each saved file only holds its own rows.
    train_blocks = blocks[:n_train].clone()
    val_blocks = blocks[n_train:].clone()

    torch.save(train_blocks, os.path.join(args.outdir, "train_blocks.pt"))
    torch.save(val_blocks, os.path.join(args.outdir, "val_blocks.pt"))

    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "block_size": args.block_size,
        "block_len_stored": block_len,
        "n_chars_total": n_chars_total,
        "n_blocks_total": n_blocks,
        "n_train_blocks": n_train,
        "n_val_blocks": n_val,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "source_file": args.input,
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print("=== preprocess.py (Spark local[*]) run complete ===")
    print(f"raw chars               : {n_chars_total}")
    print(f"vocab_size              : {vocab_size}")
    print(f"block_size (input len)  : {args.block_size}  (stored block_len={block_len})")
    print(f"total blocks            : {n_blocks}")
    print(f"train blocks            : {n_train}")
    print(f"val blocks              : {n_val}")
    print(f"wall_clock_seconds      : {elapsed:.2f}")
    print(f"train tensor shape      : {tuple(train_blocks.shape)}")
    print(f"val tensor shape        : {tuple(val_blocks.shape)}")

    spark.stop()


if __name__ == "__main__":
    main()
