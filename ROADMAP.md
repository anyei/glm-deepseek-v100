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
The first three landed 2026-07-08 (commit `505c5ea`).

- [x] **Rebuild the Docker image from `main`.** Done — `ds4:sm70-ipc`
  (+ `ds4:main-<sha>`) rebuilt from `main`, content-checked (the cleanup symbol
  `cuda_stream_selected_stage_free_slots` is in the shipped binary). Also fixed
  a real `.dockerignore` bug: `*.gguf` only matched the context root, so the
  245 GB `gguf/` tree was copied into every build context (now ~87 MB).
- [x] **Wire a GLM service into docker-compose / RUNNING.md.** Done — opt-in
  `glm` compose profile (single ds4-server on GPU0 + 26 GB GPU1 peer cache over
  NVLink, the validated best-decode config; `docker compose up glm`). Default
  `up` still brings up the DeepSeek pair. RUNNING.md documents it + the
  GPU1-must-be-free caveat.
- [x] **Delete the merged `glm5.2` branch** — done, local + origin.
- [ ] **Decide on upstreaming.** The GLM CUDA port is Metal-only upstream;
  decide whether to offer it back to DwarfStar. (Your call — not a code task.)
  Feasibility assessed 2026-07-09: **clean and self-contained.**
  `ds4_cuda_glm.inc` has zero Volta-specific code (no `__CUDA_ARCH__`/`sm_70`/
  FP16-GEMM) and zero dependency on the fork's cache-tier/IPC additions — it
  rides only the base (upstream) streaming path. And it implements an API
  upstream already ships for Metal (`ds4_gpu_glm_*`: 32 decls in `ds4_gpu.h`,
  125 impls in `ds4_metal.m`), so a PR is "add the CUDA impl of existing
  hooks" — `ds4_cuda_glm.inc` + its `#include` + the caps wiring + drop the
  CUDA engine-open refusal, plus the validation evidence (§8). The Volta perf
  work (FP16 GEMM, cache tiers, NVLink IPC) is separable and would be a
  distinct, more niche PR. Open questions are non-technical: does upstream want
  GLM-on-CUDA, are the fixtures shareable, the AI-assisted-dev disclosure,
  maintenance.

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
  - [x] **Cache admission policy research gate** — closed 2026-07-11. A
    context-2K/192-token passive-peer trace forced 1,608 exact-LRU evictions,
    but every candidate tied or increased bytes; the best reduction was 0.00%,
    versus the required 20%. Keep exact LRU.
  - [ ] **Quantization choices** — smaller routed experts remain the available
    byte lever. Replacing Q2_K down tensors with IQ2_XXS would theoretically
    reduce routed payload 8.33% (72.562 -> 66.516 GiB), but the required HF
    safetensors and imatrix are absent and current free disk is insufficient.
- [ ] **Fast-path kernels (attacks the ~0.8 s compute slice).** flash / staged
  KV / batched attention / split-group8 decode / batched low-rank QK are still
  stubs routing to scalar equivalents (VOLTA.md:159). Routed-MoE measures
  3.3 ms/layer. Real, but secondary until reads shrink.

## 3. Code quality (deferred from the /simplify pass)

A 4-agent `/simplify` review of the fork's V100/GLM code (2026-07-08) applied
the safe, behavior-preserving dedups — **done and merged to `main`** (commit
`885edc2`: MoE forwarder alias, `ensure_i32`→`ensure_bytes`, shared
staging-teardown/lap helpers, cached getenvs). That commit cleared both gates:
a high-effort `/code-review` (no real bugs; one strict-aliasing nit fixed) and
the §4 binary A/B in its fast deterministic-equivalence form (old vs new
`--dump-logprobs` over a 37-token prefill + 48 decode steps came back
byte-identical, same SHA256 — see VALIDATION.md §9).

The items below are the findings that were **deferred**, because each touches
numeric or correctness-critical code and must go through the §4 gate — they
can't just be committed blind on numerically-validated shipping code.

- [ ] **Unify the triplicated selected-expert compaction + staging** (top
  structural finding, flagged by 3 of 4 agents). The compact-id remap and the
  stage/validate protocol are copy-pasted across `ds4_cuda.cu`
  (`begin_selected_load`, `prepare_selected_batch`) and `ds4_cuda_glm.inc`
  (`glm_cuda_stream_stage_selected`, `glm_routed_moe_launch`), and the copies
  have **already drifted** — the `.inc` cache-hit predicate requires
  `compact_count != 0`, the inline one in `routed_moe_launch` omits it; the
  `.inc` compaction has an all-padding fallback the others lack. Extract one
  `build_compaction()` + one `cuda_stream_selected_cache_hit()` predicate and
  reconcile the differences deliberately. Correctness-critical (the contract is
  shared with the kernels) → full §4 gate.
- [ ] **Dedup the device-kernel numeric copies** (numerics-affecting → §4 gate,
  expect a fixture re-run, not a delta-0 A/B). The ~11 open-coded block
  reductions → `glm_block_reduce_sum/max` helpers; `glm_rope_offset_kernel`
  re-inlines yarn math the file's own `glm_dev_rope_corr`/`glm_dev_rope_cs`
  helpers already provide (and duplicates `rope_tail_kernel`); the new
  `dev_iq2_xxs_dot_block_f32` duplicates the existing `dev_iq2_xxs_dot_f32`
  (the sibling Q2_K arm already reuses the shared dot); the `dev_q{4,5,6}_K_dot_f32`
  carry an always-`nb==1` loop. "Algebraically equal" ≠ bit-identical, so these
  need the soak/fixture, not just the A/B.
- [ ] **Consolidate env-var parsing** (behavior-adjacent — changes accepted
  input syntax). ~11 near-duplicate `strtoul/strtoull` parse+clamp blocks with
  inconsistent validation (some skip whitespace, the IPC pair accepts trailing
  garbage, ad-hoc clamp ceilings). Route every knob through one bounded helper
  per module (`cuda_parse_mib_env` / `dist_parse_positive_u32` already exist).

The two efficiency findings below are perf, not cleanup — they belong to §2's
hot-path work but were surfaced here:

- [x] **Peer-owner routed MoE research gate** — stopped 2026-07-10. The isolated
  probe returned exact per-slot rows and measured 26–36% below peer-copy plus
  GPU0 MoE, but the real replacement regressed median steady decode 4.04 -> 3.99
  t/s (-1.2%) because synchronous control/device-switch overhead consumed the
  gain. Logprob A/B was byte-identical. Replacement code was reverted; retain
  only `DS4_CUDA_PEER_OWNER_PROBE` for research and do not begin mixed ownership.

- [x] **Per-layer allocator churn in the staging path** — completed 2026-07-10.
  `begin_selected_load` / `begin_compact_load` /
  `glm_cuda_stream_stage_selected` now share process/session scratch vectors,
  clear logical lengths, and retain capacity. A 9,288-row before/after expert
  trace matched cache classification and bytes exactly. The
  O(n_total_expert) initialization remains for a later generation-tag change
  only if profiling shows it matters.
- [x] **Peer cache masked local-cache growth** — fixed 2026-07-10. Cache growth
  compared a requested local target against combined local+peer capacity, so a
  tiny seed allocation (about 3 slots at context 256) appeared to satisfy the
  701-slot local target once 3,944 peer slots existed. Compare against
  `local_capacity` instead. Three 96-token runs improved median steady decode
  3.77 -> 4.25 t/s (+12.7%); the trace and architecture CSV preserve the old
  geometry/result. The benchmark parser now prefers an explicit runtime
  allocation line rather than inferring slots from the requested plan.
- [x] **Forced D2H sync every decode layer even on cache hit** — implemented
  2026-07-10. GLM's early selected-expert staging already reads and records the
  authoritative host IDs; both routed-MoE launchers now trust that record and
  avoid the second blocking `ds4_gpu_tensor_read(selected)`. Prefill/cache-miss
  paths retain the required readback, and
  `DS4_CUDA_STREAM_SELECTED_ID_CROSSCHECK=1` restores a diagnostic GPU-vs-host
  comparison. CUDA and DeepSeek regression smokes pass; a GLM runtime fixture
  remains blocked because the local GLM model was removed.

## 4. Standing rule for every change above

Nontrivial runtime changes go through **VALIDATION.md §9**:

1. **Soak** — full 100-case fixture with the feature on, stall watchdog armed
   (the peer-tier hang only surfaced ~8 min into sustained load).
2. **Fixture scoring** — the same run is the quality datapoint; compare to the
   recorded band (§8 for GLM Q2) or the Flash baseline.
3. **Binary A/B** — for changes that must not alter numerics (IO, caching,
   scheduling), `compare_scores.py old.tsv new.tsv` must show delta ≈ 0.
