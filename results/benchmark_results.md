# ZeRO stage benchmark results

Measured on this machine: 4 ranks, 300 training steps per run, CPU only. Real subprocess runs; raw logs in `results/logs/zero{0,1,2}.log`.

| ZeRO stage | wall-clock (s) | aggregate tokens/sec | peak RSS (MB, max over ranks) | loss (first -> final, mean) |
|---|---|---|---|---|
| 0 | 87.6 | 28058 | 607 | 4.22 -> 2.46 |
| 1 | 94.3 | 26070 | 620 | 4.22 -> 2.46 |
| 2 | 90.5 | 27177 | 606 | 4.22 -> 2.46 |

ZeRO stage 3 was smoke-tested (N=2 ranks, 10 steps) and ran without crashing/loss divergence, but the full N=4/300-step benchmark run was not completed for this pass -- stage 3 is not in the table above.
