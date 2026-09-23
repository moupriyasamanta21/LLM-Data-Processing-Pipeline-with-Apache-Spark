"""
benchmark.py
------------
Actually launches `torch.distributed.run` for each ZeRO stage config,
with the same fixed step count / micro-batch size / rank count for every
stage, so the resulting timing numbers are comparable. Captures real
stdout from each rank (each rank prints its own elapsed time, peak RSS and
tokens/sec at the end of src/train.py), plus the rank-0 metrics JSON that
train.py writes, and assembles results/benchmark_results.json and
results/benchmark_results.md.

No numbers here are estimated: every field in the output comes from parsing
the actual subprocess output of a real run on this machine.
"""
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONE_RE = re.compile(
    r"\[rank (\d+)\] DONE steps=(\d+) elapsed_sec=([\d.]+) "
    r"peak_rss_mb=([\d.]+) tokens_per_sec=([\d.]+) "
    r"final_loss=([\d.]+) first_loss=([\d.]+)"
)


def run_stage(stage, world_size, steps, log_every, log_path):
    cfg_path = os.path.join(REPO_ROOT, "ds_configs", f"zero{stage}.json")
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        os.path.join(REPO_ROOT, "src", "train.py"),
        "--ds_config", cfg_path,
        "--steps", str(steps),
        "--log_every", str(log_every),
        "--metrics_out", os.path.join(REPO_ROOT, "results", f"raw_zero{stage}.json"),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(max(1, 16 // world_size))

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0

    with open(log_path, "w") as f:
        f.write("=== CMD ===\n" + " ".join(cmd) + "\n\n")
        f.write("=== STDOUT ===\n" + proc.stdout + "\n\n")
        f.write("=== STDERR (tail) ===\n" + "\n".join(proc.stderr.splitlines()[-80:]) + "\n")

    ranks = []
    for m in DONE_RE.finditer(proc.stdout + proc.stderr):
        r = {
            "rank": int(m.group(1)),
            "steps": int(m.group(2)),
            "elapsed_sec": float(m.group(3)),
            "peak_rss_mb": float(m.group(4)),
            "tokens_per_sec": float(m.group(5)),
            "final_loss": float(m.group(6)),
            "first_loss": float(m.group(7)),
        }
        ranks.append(r)

    success = proc.returncode == 0 and len(ranks) == world_size

    entry = {
        "stage": stage,
        "world_size": world_size,
        "steps": steps,
        "returncode": proc.returncode,
        "success": success,
        "benchmark_wall_clock_sec": wall,  # includes process spawn/import overhead
        "per_rank": ranks,
        "log_file": os.path.relpath(log_path, REPO_ROOT),
    }
    if success:
        entry["max_elapsed_sec_over_ranks"] = max(r["elapsed_sec"] for r in ranks)
        entry["max_peak_rss_mb_over_ranks"] = max(r["peak_rss_mb"] for r in ranks)
        entry["aggregate_tokens_per_sec"] = sum(r["tokens_per_sec"] for r in ranks)
        entry["mean_final_loss"] = sum(r["final_loss"] for r in ranks) / len(ranks)
        entry["mean_first_loss"] = sum(r["first_loss"] for r in ranks) / len(ranks)
    else:
        entry["error_tail"] = "\n".join(proc.stderr.splitlines()[-15:])
    return entry


def main():
    world_size = int(os.environ.get("BENCH_WORLD_SIZE", "4"))
    steps = int(os.environ.get("BENCH_STEPS", "300"))
    log_every = 50
    stages = [0, 1, 2, 3]

    os.makedirs(os.path.join(REPO_ROOT, "results", "logs"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)

    all_results = []
    for stage in stages:
        print(f"=== running ZeRO stage {stage} : world_size={world_size} steps={steps} ===", flush=True)
        log_path = os.path.join(REPO_ROOT, "results", "logs", f"zero{stage}.log")
        entry = run_stage(stage, world_size, steps, log_every, log_path)
        all_results.append(entry)
        if entry["success"]:
            print(
                f"  OK: wall={entry['max_elapsed_sec_over_ranks']:.2f}s "
                f"agg_tok/s={entry['aggregate_tokens_per_sec']:.1f} "
                f"peak_rss={entry['max_peak_rss_mb_over_ranks']:.1f}MB "
                f"loss {entry['mean_first_loss']:.3f}->{entry['mean_final_loss']:.3f}",
                flush=True,
            )
        else:
            print(f"  FAILED (returncode={entry['returncode']}). See {log_path}", flush=True)
            print("  error tail:\n" + entry.get("error_tail", ""), flush=True)

    out_json = os.path.join(REPO_ROOT, "results", "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump({"world_size": world_size, "steps": steps, "runs": all_results}, f, indent=2)

    # Human-readable markdown table.
    lines = []
    lines.append(f"# ZeRO stage benchmark results\n")
    lines.append(f"Measured on this machine: world_size={world_size} ranks, {steps} training steps per run, "
                  f"CPU only. All numbers below come from parsing real stdout of real subprocess runs "
                  f"(see `results/logs/zero*.log` for full raw output).\n")
    lines.append("| ZeRO stage | status | wall-clock (s) | aggregate tokens/sec | peak RSS (MB, max over ranks) | loss (first -> final, mean over ranks) |")
    lines.append("|---|---|---|---|---|---|")
    for entry in all_results:
        if entry["success"]:
            lines.append(
                f"| {entry['stage']} | OK | {entry['max_elapsed_sec_over_ranks']:.2f} | "
                f"{entry['aggregate_tokens_per_sec']:.1f} | {entry['max_peak_rss_mb_over_ranks']:.1f} | "
                f"{entry['mean_first_loss']:.3f} -> {entry['mean_final_loss']:.3f} |"
            )
        else:
            lines.append(f"| {entry['stage']} | FAILED (rc={entry['returncode']}) | - | - | - | - |")
    out_md = os.path.join(REPO_ROOT, "results", "benchmark_results.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\nWrote", out_json)
    print("Wrote", out_md)


if __name__ == "__main__":
    main()
