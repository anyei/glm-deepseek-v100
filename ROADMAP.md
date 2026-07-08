# Roadmap (glm-deepseek-v100 fork)

This is the prioritized, actionable plan for the fork. It is the top-level
"what next" document. The detailed status, measurements, and rationale live
in the backing documents and are treated as **backlog / reference**:

- [VOLTA.md](VOLTA.md) — V100/Volta internals and every measured number
  behind the perf items below (the "Pending (the performance phase)"
  section is the raw source for §2 here).
- [VALIDATION.md](VALIDATION.md) — the validation playbook; §9 is the
  merge gate every change below must pass, §8 is the release-gate log.
- [RUNNING.md](RUNNING.md) — how the Docker image and the two-GPU compose
  stack are run today (DeepSeek-only; §1 below fixes that).

Last updated: 2026-07-08.

## Where we are

Correctness is **done and shipped**. GLM 5.2 on CUDA is validated (CPU-oracle
logit match + the 100-case official-continuation release gate), on by default,
and merged to `main` (merge commit `c962770`). DeepSeek V4 Flash continues to
work on the same streaming path. What remains is the **performance phase** and
turning the merged work into something actually deployed.

Warm GLM decode is **0.45 t/s** today (from 0.16 at the start of the perf
phase, a 2.8x gain). A decode token is ~2.0 s device-bound expert reads +
~0.8 s GPU compute/drain — so the wins left are in *bytes read per token* and
*compute*, not IO plumbing (the disk path is already at ~1.9–2.0 GB/s against a
~2.45 GB/s ceiling and is insensitive to worker count).

## 1. Near-term: make the merge real (ops)

Small, high-value, do first. None of this is blocked on the perf work.

- [ ] **Rebuild the Docker image from `main`.** Everything deployed
  (`ds4:sm70-ipc`, the benchmark image) was built from the branch. Verify the
  new binary with a content check (see VALIDATION.md §1 — `docker build | tail`
  silently eats the exit code, so grep the shipped binary for a known string).
- [ ] **Wire a GLM service into docker-compose / RUNNING.md.** Both only
  describe DeepSeek serving today. Add the validated best-decode config:
  26 GB expert budget + 26 GB peer cache, host L2 off. Caveat: the peer cache
  wants GPU1, which the prod llama-server container currently owns — so this
  needs prod stopped or shrunk (documented in VOLTA.md).
- [ ] **Delete the merged `glm5.2` branch** (local + origin) — it is 0 commits
  ahead of `main`.
- [ ] **Decide on upstreaming.** The GLM CUDA port is Metal-only upstream;
  decide whether to offer it back to DwarfStar.

## 2. Performance phase (the real work)

Ordered by expected impact. Every change here is numerics-affecting or
numerics-adjacent, so each goes through the VALIDATION.md §9 gate
(soak + fixture score + old-vs-new binary A/B).

- [ ] **Distributed GLM across both V100s — the key experiment.** GPU IPC and
  distributed sessions still *decline* GLM and fall back to TCP/single-GPU
  (VOLTA.md:167). A two-GPU layer split gives each GPU half the layers' expert
  pool + its own cache — but it *competes with the peer expert cache* for GPU1,
  which is today's best config. The open, measurable question: does
  "split layers across 2 GPUs" beat "1 GPU + 26 GB peer cache"? This is the one
  lever that could plausibly move decode well past 0.5 t/s without kernel work.
  Run the A/B before committing to either.
- [ ] **Reduce bytes per token.** The only attack on the 2.0 s read time.
  Two sub-levers:
  - **Cache admission policy** — which experts earn the ~52 GB of combined
    GPU-tier slots (currently fills naively from the miss stream).
  - **Quantization choices** — smaller routed experts shrink the 7.1 GiB
    per-token working set directly.
- [ ] **Fast-path kernels (attacks the ~0.8 s compute slice).** flash / staged
  KV / batched attention / split-group8 decode / batched low-rank QK are still
  stubs routing to scalar equivalents (VOLTA.md:159). Routed-MoE measures
  3.3 ms/layer. Real, but secondary until reads shrink.

## 3. Standing rule for every change above

Nontrivial runtime changes go through **VALIDATION.md §9**:

1. **Soak** — full 100-case fixture with the feature on, stall watchdog armed
   (the peer-tier hang only surfaced ~8 min into sustained load).
2. **Fixture scoring** — the same run is the quality datapoint; compare to the
   recorded band (§8 for GLM Q2) or the Flash baseline.
3. **Binary A/B** — for changes that must not alter numerics (IO, caching,
   scheduling), `compare_scores.py old.tsv new.tsv` must show delta ≈ 0.
