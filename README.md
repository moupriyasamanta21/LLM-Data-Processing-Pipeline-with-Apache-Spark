# mini-zero-llm

A small, personally-built-and-run project exploring the systems side of LLM
training: a Spark preprocessing step, a from-scratch character-level GPT in
PyTorch, and distributed training with DeepSpeed's ZeRO memory optimizer,
compared across stages 0/1/2. Built to understand these systems hands-on,
not to reimplement any specific course assignment.

**Scope note (read this first):** everything here runs CPU-only, on one
16-core machine, with a small model (~818k params) and short training runs
(150-300 steps). This is a personal learning project, not a claim of
large-scale or GPU-class results. Every number below is from an actual run
on this machine (see `results/`) -- nothing is estimated.

## What's in it

- `src/preprocess.py` -- PySpark (local mode) job that reads the raw text,
  builds a character-level vocabulary, encodes the text, and chunks it into
  fixed-length blocks, split 90/10 into train/val. Each saved block is
  `block_size + 1 = 129` tokens long; `train.py` slices `tokens[:-1]` as
  input and `tokens[1:]` as the shifted next-token target (documented in
  the script, not pre-shifted at save time).
- `src/model.py` -- a small decoder-only transformer from scratch (token +
  positional embeddings, causal self-attention + MLP blocks, final LayerNorm,
  tied input/output embeddings -- the standard GPT-2/nanoGPT default).
- `src/train.py` -- loads the preprocessed blocks, builds the model, trains
  it under `deepspeed.initialize()` with a given ZeRO config, one shard of
  data per rank (real data-parallel, not N copies of the same batch).
- `ds_configs/zero{0,1,2,3}.json` -- DeepSpeed configs, one per ZeRO stage.
- `src/evaluate.py` -- loads saved checkpoints, computes perplexity on the
  held-out validation blocks, and generates a short sample continuation.

## How to run it

```bash
source .venv/bin/activate

# 1. preprocess (real Spark local[*] job)
python src/preprocess.py

# 2. sanity-check the model
python src/model.py

# 3. train under a given ZeRO stage, N ranks
python -m torch.distributed.run --nproc_per_node=4 src/train.py \
  --ds_config ds_configs/zero1.json --steps 300 \
  --ckpt_dir checkpoints/my_run --ckpt_steps 100,200,300

# 4. evaluate saved checkpoints
python src/evaluate.py --ckpt_dir checkpoints/my_run --tags step100,step200,step300
```

## Real measured results

### Preprocessing (actual Spark run)

1,115,393 raw characters, vocab_size = 65 unique characters, block_size =
128. Chunked into 8,646 total blocks -> **7,782 train / 864 validation**
blocks (90/10 split). Wall clock for the Spark job: ~10s on this machine.

### Model

`n_layer=4, n_head=4, n_embd=128, block_size=128` -> **818,048 total
parameters** (measured via `model.num_params()`, embeddings tied).

### ZeRO stage comparison (4 ranks, 300 steps, CPU, same model/batch size)

| ZeRO stage | wall-clock (s) | aggregate tokens/sec | peak RSS (MB, max over ranks) | loss (first -> final, mean over ranks) |
|---|---|---|---|---|
| 0 | 87.6 | 28,058 | 607 | 4.22 -> 2.46 |
| 1 | 94.3 | 26,070 | 620 | 4.22 -> 2.46 |
| 2 | 90.5 | 27,177 | 606 | 4.22 -> 2.46 |

Full raw per-rank logs are in `results/logs/zero{0,1,2}.log`;
`results/benchmark_results.json` has the parsed numbers. At this tiny model
size (818k params) on CPU, the three stages perform within noise of each
other -- ZeRO's memory-partitioning benefits only really show up at model
sizes where optimizer/gradient state actually strains memory, which this toy
model doesn't. That's an honest, expected result at this scale, not a bug.

**ZeRO stage 3**: smoke-tested at 2 ranks / 10 steps and it ran without
crashing or diverging. The full 4-rank / 300-step benchmark run for stage 3
was **not completed** (deprioritized for time) -- so it is intentionally left
out of the table above rather than reported with made-up numbers.

### Evaluation (checkpoints from a 150-step ZeRO-1 run, 2 ranks)

Perplexity on the held-out validation blocks, at 3 checkpoints:

| checkpoint | val cross-entropy | val perplexity |
|---|---|---|
| step 30 | 3.057 | 21.26 |
| step 90 | 2.639 | 14.00 |
| step 150 | 2.525 | 12.49 |

Perplexity drops monotonically as expected. Full results (including a
generated sample per checkpoint) are in `results/evaluation.json`. Sample
generated continuation from the step-150 checkpoint, prompt `"ROMEO:"`,
greedy/top-k sampling:

```
ROMEO:

E:
An thedeneane in rolound e we thonse ane thilou wist ic wis teriale
myonsenime cose angr prerancey mou ber s ocorl teme s tofren gre d, hag
thoris annonint tisans thet touraners he

He hes, tode
```

It's not real English yet -- 150 steps on a tiny char-level model is
nowhere near enough to learn spelling, just rough letter/punctuation
statistics and some name-like fragments (`ROMEO`, `He`). That's the honest,
expected result at this step count; the point of this evaluation was to
verify the perplexity-over-checkpoints pipeline works end to end, not to
produce Shakespeare.

## Honest limitations / what wasn't done

- ZeRO stage 3: smoke-tested only, not benchmarked at full scale (see above).
- No GPU was available on this machine; everything above is CPU-only.
- The model/training runs are intentionally short (150-300 steps) to keep
  iteration fast on CPU -- this is a systems/pipeline demonstration, not an
  attempt at a well-trained language model.
