---
status: reference
updated: 2026-09-26T21:29:45-04:00
scope: Seed protocol
---

# Seed protocol

A **training run seed** controls initialization and action sampling. It is not a single board repeated throughout training. Reusing the same run seed across reward conditions creates a paired comparison; different run seeds test robustness.

## Recurrent reproduction pipeline

Each training episode receives `(1 << 63) | (run_seed << 32) | episode_counter`. Run seeds use 31 bits, counters use 32 bits, and exhaustion raises an error. Completed episode seeds are logged in `data/episodes.csv`; active environments and their seeds are checkpointed. The upper half of uint64 is reserved for training, so evaluation seed ranges cannot overlap it.

| Purpose | Seeds |
| --- | --- |
| Initial paired training run | 71 |
| V2 default / planned replication | 72 / 73 |
| Periodic validation during training | 700000–700099 |
| Independent comparison after the pair | 800000–800299 |
| Long-continuation comparison (default) | 810000–810999 |
| Engineering checks | 710000 onward |
| Reserved final test after model selection | 900000–909999 |

The same periodic games are reused to show progress. Once comparison seeds influence model choice, they become validation data. Keep the final range unopened until the method and checkpoint selection are fixed. Evaluation clears LSTM memory and per-game environment history between games. Distinct initial seeds do not force two policies to encounter identical later tile draws; their moves change trajectories.

## Historical MLP pipeline

Earlier MaskablePPO runs used seeds 42 and 43, followed by article-inspired runs with 52 and 53. The latter used 120000–120099 for periodic evaluation and 500000–500299 for comparison. Their training episode seeds were generated from an environment RNG and were neither logged individually nor checked against evaluation ranges. Those comparison ranges have already been inspected and are no longer untouched tests.

A particular integer such as 42 has no special learning advantage. Replication uses multiple fresh training seeds while keeping validation rules fixed.

Engine revision 2 changes some random-call ordering. Identical seeds across revisions do not imply identical games; record the engine revision and evaluate compared policies together.
