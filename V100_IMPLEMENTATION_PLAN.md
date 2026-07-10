# Two-V100 Streaming Implementation Plan

Derived from [V100_ARCHITECTURE_ANALYSIS.md](V100_ARCHITECTURE_ANALYSIS.md).
This plan targets the current host: two 32 GiB V100-SXM2 GPUs over NVLink, an
i9-10850K, 46 GiB RAM, and one approximately 2.5 GB/s NVMe.

The target architecture is a single GPU0-owned transformer graph with exclusive
expert-cache ownership across GPU0/GPU1 and owner-side routed-MoE compute. The
plan is deliberately gated: the current passive peer cache remains the release
path until a small prototype proves that peer compute wins.

## 0. Definition of success

### Primary workload

Interactive/agent decode with:

- DeepSeek V4 Flash IQ2XXS/Q2K model;
- GLM 5.2 routed-Q2 model;
- contexts at 1K, 16K, and the largest production context that fits safely;
- short and long prompts;
- at least 96 generated tokens for performance measurements.

### Primary metrics

1. Steady decode tokens/s.
2. First-token latency.
3. SSD bytes/token and read GB/s.
4. GPU0/GPU1 expert-cache hit rates.
5. GPU0/GPU1 routed-MoE milliseconds/layer.
6. NVLink bytes/token and transfer milliseconds.
7. Prefill tokens/s at 256, 2K, 16K, and 32K prompt lengths.
8. Peak VRAM per device and host `MemAvailable`.
9. Numerical score/delta under the existing validation fixtures.

### Acceptance rule

Do not merge a complexity increase for noise-level gains. A performance phase
must provide either:

- at least 10% steady-decode improvement on its target model with no material
  regression on the other model; or
- a measured reduction of at least 20% in SSD bytes/token that is expected to
  compound with a later phase; or
- a correctness/reliability improvement with no statistically meaningful
  throughput loss.

Use at least three measured runs after one warm-up. Report median and range.
Short seven-token samples are smoke tests, not optimization evidence.

## 1. Phase 0 — make the host benchmarkable

### 1.1 Storage remediation

**Status: remediated for DeepSeek on 2026-07-10.** `/mnt/full-models` now has
about 171 GiB free after copying and byte-comparing the 86,720,111,200-byte
DeepSeek model onto it. Its SHA-256 is
`31598c67c8b8744d3bcebcd19aa62253c6dc43cef3b8adf9f593656c9e86fd8c`.
A cold O_DIRECT 16 MiB/QD4 probe measured 3.14 GB/s; a deterministic QD1
scattered 4 MiB probe measured 582 MiB/s. The GLM file formerly in this tree is
no longer present, so its sweep remains blocked on model availability.

Current `/mnt/full-models` free space was about 6.8 GiB (98% used). Before
creating new model layouts:

- free or relocate enough data to leave at least 15–20% NVMe free;
- target at least 80–100 GiB free for traces, worktrees, and a DeepSeek expert
  sidecar;
- for a GLM sidecar, provision a larger/additional NVMe rather than trying to
  fit it beside the 245 GiB GGUF on this partition;
- run `fstrim` if appropriate for the filesystem/device and verify mount
  behavior;
- rerun `speed-bench/io_probe.c` after cleanup.

**Gate:** cold 16 MiB/QD4 probe remains around 2.45 GB/s and the scattered
expert-pattern probe is recorded.

### 1.2 Memory remediation

**Status: host headroom is sufficient for DeepSeek baselines.** About 21 GiB is
available and the GPUs are normally idle. Approximately 23 GiB of old dormant
swap remains in use, but the observed benchmark delta was only 25 pages
(100 KiB), not active memory pressure. The harness rechecks state before every
process and rejects more than 16 MiB during a one-second preflight or 256 MiB
over a complete process. The observed 117 MiB during a model warm-up was only
about 0.05% of its backing-device reads while 21 GiB RAM remained available;
the exact-zero rule incorrectly classified that low-rate traffic as active
pressure. Do not run `swapoff -a` without enough RAM.

- stop unrelated model processes;
- verify swap-in/swap-out (`vmstat 1`) is zero during benchmarks;
- reboot or deliberately reset the environment if old swapped pages make runs
  noisy;
- do not enable `DS4_CUDA_HOST_EXPERT_CACHE_GB` on this host during canonical
  tests;
- record configured DIMM speed with privileged DMI access.

**Gate:** no active swapping and at least 8 GiB host headroom throughout a run.

### 1.3 Canonical benchmark harness

**Status: implemented.** `speed-bench/v100_bench.sh` covers single-GPU,
passive-peer, and distributed profiles; `speed-bench/v100_compare.py` reports
median/range and old-vs-new deltas. Functional smokes passed for all three
profiles. The harness also refuses canonical runs on a busy/swapping host unless
explicitly overridden.

The harness records:

- git SHA and binary hash;
- model path/hash/size;
- CUDA driver/toolkit and `sm_70` build;
- GPU clocks/power/temperature;
- exact environment variables and CLI;
- prompt and token counts;
- CSV output and stderr diagnostics;
- diskstats, VRAM, and process RSS samples;
- warm/cold status.

It must support:

```text
single GPU + local cache
single GPU + passive peer cache
two-process layer split
future owner-compute mode
```

Keep benchmark artifacts outside git unless they are summarized reference CSVs.

**Gate:** rerunning the same binary/config three times yields a narrow enough
range to distinguish a 10% change.

## 2. Phase 1 — establish truthful baselines

### 2.1 VRAM envelope sweep

**Status: provisional DeepSeek geometry probe completed; canonical sweep
blocked by storage remediation.** A 2026-07-10 one-token/context-16 probe (not a
performance measurement) confirmed why the live slot count must be recorded.
With the default 16 GiB runtime reserve, requested local budgets of
8/12/16/20/24/26 GiB planned 701/1308/1915/2522/3128/3432 slots but were
live-capped to only 415–553 slots after resident tensors loaded. With an 8 GiB
local request, peer requests of 8/16/24/26 GiB allocated
1213/2427/3640/3944 slots. All probes completed without OOM. Peak VRAM and
backing-device read bytes are now included in `v100_compare.py` summaries so the
canonical rerun records these resources alongside throughput. Raw probe runs
remain under ignored `speed-bench/local-runs/`.

Do not interpret the one-token rates or peak VRAM from this probe: the cache was
not filled. Both model filesystems remain 98–99% full, so the storage gate in
Phase 0 still blocks canonical performance results.

For each model/context, sweep the local expert budget upward until the live
memory cap or OOM margin is reached. Record actual allocated slots, not only the
requested GB value.

Suggested starting grid:

```text
DeepSeek: 8, 12, 16, 20, 24, 26 GB local
GLM:      16, 20, 24, 26 GB local
```

Then sweep passive peer cache sizes:

```text
0, 8, 16, 24, 26 GB peer
```

Do not combine the host L2 in canonical runs.

### 2.2 Architecture A/B

**Status: DeepSeek context-256 decode decision completed on 2026-07-10.** The
tracked results are in `speed-bench/v100_architecture.csv`. With 96 generated
tokens, the 8 GiB local-cache profile reached median 2.37 steady t/s (two clean
samples at 2.37–2.40 and one storage-latency outlier at 0.94). The first passive
peer result reached 3.77 t/s, but tracing later proved a grow-condition bug had
left only about 3 local slots beside 3,944 peer slots; the summary parser had
incorrectly reported the planned 701 local slots. Comparing grow targets against
`local_capacity` rather than combined capacity restored 701+3,944 slots. Three
clean fixed runs reached median 4.25 steady t/s (4.10–4.26), +12.7% over the
superseded peer run and +79% over the single-GPU median. The fixed run traded
-4.6% prefill and +13% first-token latency for that decode gain; backing reads
were essentially flat at 209.12 GiB. The old row remains in the CSV as
`peer-grow-bug`; `peer-fixed` is the production result.

The distributed layer split completed one full 96-token screening run at only
0.14 steady t/s, about 27x slower than passive peer, while reading about 980 GiB
across its two processes. Repeating that approximately 11-minute losing run
three times could not plausibly satisfy the +10% gate, so it was stopped as an
explicit no-go rather than consuming another 30–40 minutes. This row is marked
`warmup_screen`, not a three-sample canonical measurement. Passive peer remains
the production decode default. The GLM comparison is unavailable because its
local model was deliberately removed during storage remediation.

Benchmark on identical prompts:

1. one V100 + SSD;
2. GPU0 + passive GPU1 peer cache;
3. two-process layer split over CUDA IPC/NVLink.

DeepSeek already supports all three. GLM first compares 1 and 2; whole-layer
GLM distribution is a later independent experiment.

**Deliverable: completed for the available DeepSeek model.**
`speed-bench/v100_architecture.csv` plus this interpretation; broader context
lengths remain future workload characterization rather than a blocker for the
architecture decision.

**Decision:** passive peer cache remains production default unless layer split
wins steady decode by at least 10% without unacceptable first-token or prefill
regressions.

## 3. Phase 2 — instrumentation and low-risk hot-path cleanup

This phase should not alter kernel math.

### 3.1 Expert trace format

**Status: implemented (2026-07-10).** Set `DS4_CUDA_EXPERT_TRACE` to an output
path to enable the compact CSV; it is fully disabled by default and bounded to
one million rows (`DS4_CUDA_EXPERT_TRACE_MAX_ROWS`).
`DS4_CUDA_EXPERT_TRACE_MODEL_ID` places a caller-supplied model hash/name in the
header. Rows preserve router-slot IDs and report logical token epoch, layer,
expert, GPU/peer/host/SSD tier, final owner, hit/miss, victim ID/age, uniquely
accounted read bytes, batched-read latency, classify/peer-copy time, total stage
time, model bytes, and cache geometry. Batch timing is repeated on its member
rows, while read bytes are emitted only on the first row for each compact expert
so summing bytes does not overcount prefill deduplication. A CUDA `sm_70` build
and an end-to-end 9,288-row DeepSeek smoke passed. Raw traces remain outside
git.

Add an opt-in binary or compact CSV trace capturing:

```text
token index
layer
selected expert IDs
cache owner/tier
hit or miss
victim ID and age
bytes read
read latency
upload/peer-copy latency
```

Bound trace overhead and keep it disabled by default. Include model identity and
cache geometry in the header.

### 3.2 Remove allocator churn

**Status: implemented and smoke-gated (2026-07-10).** CUDA staging now owns one
process/session scratch object whose read descriptors, fill/direct lists,
expert-to-slot map, compact IDs, slot IDs, retry descriptors, and optional trace
buffers retain capacity across routed layers. DeepSeek built for `sm_70` and ran
end to end. A before/after trace comparison matched all 9,288 rows exactly for
call/token/layer/router slot, expert, tier, owner, hit/miss, victim/age, uniquely
accounted bytes, model identity, and cache geometry, with total read bytes
unchanged at 16,031,416,320. This directly verifies unchanged cache
classification and staging semantics; timing fields were intentionally excluded
from equality. A longer numerical/performance A/B remains required before a
release claim.

Hoist per-layer staging vectors into reusable process/session scratch storage:

- selected IDs;
- expert-to-slot maps;
- compact IDs;
- load descriptors;
- fill/direct lists;
- host-cache reservations.

Retain capacity and clear logical lengths. Replace full expert-map memset with a
generation-tag array if profiling shows it matters.

**Gate:** byte-identical logprob A/B; at least no throughput regression. Keep the
change only if allocation count and/or CPU staging time falls measurably.

### 3.3 Remove redundant decode D2H synchronization

**Status: implemented; GLM runtime validation pending model availability.**
GLM's early selected-load API already performs the necessary router-ID readback
and now leaves an authoritative host record tied to model/layer/count. Both the
generic and typed GLM routed-MoE launchers consume that record without repeating
the blocking D2H copy on every decode layer. Prefill or a missing record still
reads and stages safely. `DS4_CUDA_STREAM_SELECTED_ID_CROSSCHECK=1` re-enables
the old GPU readback as a diagnostic comparison and fails loudly on mismatch.
The CUDA `sm_70` build and DeepSeek regression smoke pass. The expected removal
is one blocking synchronization per GLM routed decode layer; measurement and
the full GLM fixture cannot run until that deleted model is restored.

Thread already-known host router IDs through the staging API instead of reading
selected IDs back from the GPU solely for validation. Preserve a diagnostic
cross-check mode that compares host/GPU IDs.

**Gate:** full fixture if staging semantics can change; otherwise the fast
byte-identical A/B plus a long soak. Confirm approximately one blocking sync per
routed layer is removed.

### 3.4 Hash the local GPU cache index

**Status: implemented and trace-gated (2026-07-10).** The GPU/peer expert cache
now has an `unordered_map` keyed by the full model pointer/size, layer, total
expert count, expert ID, all three tensor offsets, and expert geometry. Every
indexed hit is revalidated against authoritative slot metadata; stale entries
are erased and become misses. Victim keys are removed before reuse, eager batch
reservations are indexed immediately, invalidation clears the directory, and an
allocation failure disables the index and safely falls back to the original
scan. Cache teardown was also changed from an unsafe `memset` over C++ container
members to value assignment. A 9,288-row before/after DeepSeek trace matched
hit/miss, tier, owner, victim/age, and all 16,031,416,320 read bytes exactly.
A passive-peer smoke exercised 4,471 peer hits and 5,604 peer-owned rows with no
errors.

The host L2 already uses a hash index; the GPU expert cache still scans slots.
Add a host metadata index keyed by full model/layer/expert identity and verify
the slot record on every hit. Keep LRU metadata authoritative.

**Gate:** byte-identical A/B and cache hit/miss sequence identical to baseline.

## 4. Phase 3 — policy simulation before policy implementation

### 4.1 Offline simulator

**Status: baseline replay and initial policies implemented (2026-07-10).**
`speed-bench/expert-cache-sim.py` groups rows by staging call, deduplicates
router slots, and replays unique experts in the runtime's ascending compact-ID
order. It reports hits, misses, byte reads, evictions, layer counters, and owner
admission/residency. Exact LRU reproduced both validation traces exactly:
7,023/2,265 hits/misses and 16,031,416,320 bytes for local-only, and
6,938/1,834 with 12,980,846,592 bytes for passive peer. Alternatives are
explicitly marked untrusted by a nonzero exit unless this baseline matches.
Segmented LRU, TinyLFU admission, owner-balanced placement, equal per-layer
quotas, decode-protected/prefill-ephemeral regions, optional top-K replication,
capacity overrides, reuse-distance percentiles, phase counters, and per-layer
reports are available.

A context-256/96-generated-token passive-peer trace (181,632 rows) exposed and
then validated the local-grow fix above. With the corrected 701+3,944 geometry,
exact LRU reproduced 45,472 hits, 16,466 misses, and 116,544,503,808 bytes. No
candidate reduced bytes: segmented LRU, TinyLFU, decode protection,
owner-balanced placement and replication tied baseline, while equal per-layer
quotas regressed bytes by 1.38%. There were no evictions under the useful
policies and prefill used no global cache, so this workload has neither capacity
pressure nor prefill pollution to fix. The required 20% runtime-policy gate
fails decisively; do not add a runtime policy for this trace.

Create `speed-bench/expert-cache-sim.py` consuming the Phase 2 trace. Simulate
combined GPU capacities and report:

- hit rate by layer and tier;
- bytes/token;
- evictions/token;
- reuse-distance distribution;
- owner load balance;
- prefill pollution of subsequent decode.

Policies:

1. exact current LRU baseline;
2. segmented LRU;
3. TinyLFU admission + LRU eviction;
4. per-layer quotas;
5. decode-protected and prefill-ephemeral regions;
6. owner-balanced exclusive placement;
7. optional top-K replication.

The simulator must reproduce observed baseline hit counts before its
alternatives are trusted.

### 4.2 Select policy by bytes, not hit count alone

**Decision: no runtime policy change for the measured DeepSeek workload.** The
corrected representative peer trace predicts 0% byte reduction from the best
alternative, far below the 20% gate; per-layer quotas are actively worse. Keep
exact LRU and avoid adding policy metadata/CPU cost. Revisit only with a longer
or materially different trace (larger context, another model, or demonstrated
eviction pressure).

Experts currently have uniform geometry within a model variant, but the design
should calculate byte cost. Select the simplest policy achieving most of the
simulated reduction. Reject policies whose metadata/decision cost approaches
the expected savings.

**Gate to runtime implementation:** at least 20% simulated SSD-byte reduction
on a representative decode trace or a clear prefill-pollution fix.

### 4.3 Runtime policy

**Status: skipped by gate.** No measured policy qualifies for implementation.

Implement behind a diagnostic environment gate initially, then remove the gate
once one release path is selected. Do not retain permanent semantic variants.

**Gate:** soak, fixture score, old/new binary A/B, and measured bytes/token.
Cache policy must be numerically exact.

## 5. Phase 4 — minimal owner-compute prototype

This is the highest-risk/highest-upside phase. Keep it narrow.

### 5.1 Narrow peer-expert context

**Status: diagnostic prototype implemented (2026-07-10).**
`DS4_CUDA_PEER_OWNER_PROBE=1` creates a narrow peer context only when requested;
it retains peer activation, routing, quantized-intermediate and per-slot output
buffers plus timing events, then tears them down on exit. It does not alter
normal tensor ownership, cache placement, or model output.

In `ds4_cuda.cu`, introduce a peer-expert runtime containing:

- peer device ID;
- peer expert arenas and metadata;
- one upload stream and one compute stream on GPU1;
- inter-device events;
- compact selected IDs/weights on GPU1;
- activation input and per-selected-slot output buffers;
- explicit initialization and teardown.

Do not change generic `ds4_gpu_tensor` ownership in the prototype. Avoid a
project-wide multi-device abstraction until the feature proves useful.

### 5.2 First prototype scope

**Status: all-peer-hit DeepSeek decode microbenchmark complete.** With local
cache forced to one slot for test coverage, 773 of 774 routed calls had all six
selected experts in peer slots. The probe copied the activation and routing
weights to GPU1, ran the existing IQ2_XXS gate/up plus Q2_K down decode kernels
directly against peer-owned arenas, and returned six unreduced F32 slot rows.
The shipping path still ran normally on GPU0; this was duplicate diagnostic
compute only.

Choose one model and one simple path:

- DeepSeek Flash decode, `n_tokens == 1`;
- one routed-expert quant combination already supported by the generic CUDA MoE
  kernels;
- all selected routed experts computed on GPU1;
- shared expert remains on GPU0;
- SSD misses target GPU1 directly;
- GPU1 returns one F32 expert-output vector per selected router slot to GPU0;
- GPU0 reduces those slot outputs in canonical baseline order.

This all-peer prototype is simpler than dynamic partitioning and directly
measures launch/event/activation overhead. It may not be the final cache design.

### 5.3 Correctness mechanics

The probe requires `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6=1` so GPU0 retains its
per-slot baseline rows. Across three runs, all returned peer rows matched GPU0
exactly (`max_delta=0`, `RMS=0`). Any peer allocation, transfer, launch, event,
or comparison error fails the routed layer while the diagnostic is enabled.
Normal execution remains unaffected when the environment gate is absent.

- Reuse existing routed-MoE dot kernels where possible, but add an unreduced
  output mode that preserves one vector per router slot.
- Preserve router-slot identity across compaction and peer dispatch.
- Combine every routed slot in the baseline order; this is mandatory, not best
  effort, because cache ownership must not alter FP grouping.
- Add a diagnostic mode that runs baseline GPU0 MoE and peer MoE on the same
  input and reports max/RMS deltas before enabling end-to-end generation.
- Any peer CUDA error fails the layer; no partial fallback after compute begins.

### 5.4 Microbenchmarks

**Final gate result: no-go; retain passive peer.** Three
runs each measured 773 all-peer calls. Owner-side means were 0.195–0.206 ms:
activation/control 0.018–0.024 ms, compute 0.165–0.169 ms, and six-slot return
0.012–0.013 ms. The matching passive path measured GPU0 MoE plus peer-hit
classify/weight copies at 0.273–0.331 ms. Adding the measured 0.0055 ms
canonical join leaves a 26.4–36.1% reduction (35.6% median), above the 15%
isolated gate. Results are tracked in `speed-bench/v100_peer_owner_probe.csv`.
A separate shipping direct-sum run measured 0.346 ms for
peer-copy plus GPU0 MoE, so the comparison is not relying on a slower baseline
kernel. However, the actual all-peer replacement A/B consistently regressed
steady decode: passive peer measured 4.03/4.04/4.05 t/s (4.04 median), while
replacement measured 3.96/3.99/4.00 (3.99 median), about -1.2%. A fixed-prompt
8-token logprob dump was byte-identical, proving correctness, but synchronous
device switching and selected-ID/weight control readbacks consumed the isolated
kernel gain. Results are tracked in
`speed-bench/v100_peer_owner_replacement.csv`. The replacement code was reverted;
only the duplicate diagnostic probe remains. Stop owner-compute work and do not
start Phase 5 exclusive ownership on this host.

Measure per layer:

```text
activation GPU0->GPU1
peer routed-MoE compute
per-slot outputs GPU1->GPU0
GPU0 shared expert overlap
canonical slot-order join/reduction
```

Compare against:

```text
peer weight copy GPU1->GPU0
GPU0 routed-MoE compute
```

**Go/no-go gate:** owner-side compute reduces routed-MoE plus peer-transfer time
by at least 15% on peer hits and does not increase total token time. If it does
not, stop and retain passive peer caching.

## 6. Phase 5 — exclusive dual-GPU ownership

**Status: skipped.** Phase 4's end-to-end replacement regressed steady decode by
about 1.2%, so the prerequisite did not pass. The sections below remain design
notes only.

Proceed only if Phase 4 passes.

### 6.1 Exclusive slot directory

Extend cache metadata with owner device. A key has one canonical slot. Slot
state must include:

```text
empty
loading
resident
in use
retiring
```

Use event-protected generations so an evicted slot cannot be reused while a
compute stream still references it.

### 6.2 Partition selected experts

After routing:

- partition selected IDs by owner;
- issue misses to their selected owner;
- compute GPU0 and GPU1 groups concurrently into per-selected-slot outputs;
- transfer peer-owned slot outputs to GPU0;
- combine every slot in canonical router order independent of owner.

Initially choose ownership by capacity/load. Add the selected admission policy
only after baseline exclusive ownership is correct.

### 6.3 Avoid duplicate weight movement

A GPU1-resident expert is computed on GPU1. Never mirror it into GPU0 compact
buffers on a normal hit. The only cross-GPU payloads should be activations,
IDs/weights, per-slot outputs, and small control/event traffic.

### 6.4 Extend to GLM

After DeepSeek passes soak and fixture gates, adapt GLM's routed-MoE launch.
Keep attention/KV/indexer on GPU0. GLM's larger expert bytes make it the likely
larger beneficiary, but its current scalar kernels and staging duplication
increase implementation risk.

**Gate:** full GLM 100-case fixture, DeepSeek fixture, 100-case soak with peer
ownership, and old/new score comparison.

## 7. Phase 6 — expert-oriented disk layout

### 7.1 Sidecar format

**Status: v1 planner, resumable builder, and verifier implemented; full-size
DeepSeek construction and verification completed.** `gguf-tools/expert-sidecar.py` parses
GGUF v2/v3 metadata and tensor tables, recognizes both DS4 and HF-style routed
tensor names, computes quantized payload sizes from GGUF block geometry,
validates every layer's gate/up/down expert count, and lays out aligned
`gate | up | down` records. Format v1 has a 4 KiB identity header, 128-byte
per-expert directory records, a model and directory SHA-256, and a payload
SHA-256 per expert. Construction preallocates `OUTPUT.part`, durably checkpoints
payload before publishing directory entries, resumes at the first incomplete
entry, and uses a no-replace hard-link publication after the complete file is
synced. `--verify` rehashes both source GGUF ranges and sidecar records. A tiny
synthetic GGUF regression covers planning, building, complete-file recovery,
verification, and corruption detection.

The restored DeepSeek IQ2XXS file produced 43 layers, 11,008 experts,
77,913,391,104 payload bytes, and a 77,914,804,224-byte sidecar. Construction
completed in 8m57s and full source/sidecar verification completed in 7m33s on
2026-07-10. The result leaves about 98 GiB free (36% of the filesystem), above
the required storage margin. The GLM symlink remains dangling, so GLM format
and fixture validation are still pending.

Implement an offline tool that copies exact expert tensor bytes into an
O_DIRECT-friendly sidecar:

```text
header + model identity
layer directory
  expert 0: gate | up | down | padding
  expert 1: gate | up | down | padding
  ...
checksums/manifest
```

Requirements:

- no requantization;
- preserve each tensor's exact bytes;
- 4 KiB alignment minimum, with block-size experiments at 1/4/16 MiB;
- atomic creation via temporary file + rename;
- resumable construction for 80–245 GiB models;
- validation against GGUF tensor hashes;
- normal GGUF fallback.

### 7.2 Read coalescing

**Status: prototyped, measured, rejected, and reverted.** The CUDA prototype
validated the v1 identity/directory against the open GGUF, used O_DIRECT on the
sidecar, replaced each three-range gate/up/down miss with one contiguous read,
scattered its pinned staging payload directly to final GPU buffers, and fell
back to normal GGUF reads when absent or rejected. A 16-token old/new logprob
A/B was byte-identical.

The one-run context-256 gate smoke was decisively negative. Relative to normal
GGUF ranges, the sidecar reduced block-device reads only 13.2% (471,631 to
409,556), increased average request size 17.0% (448.67 to 525.10 KiB), increased
read service time 13.4% (869,482 to 985,784 ms), and increased bytes read from
201.80 to 205.09 GiB. Steady decode fell from 2.56 to 1.87 t/s (-27.0%); overall
generation fell from 1.79 to 1.42 t/s (-20.7%). The prototype therefore missed
both the 10% read-time and 20% I/O-count gates and was removed from the runtime.
Raw summary metrics are preserved in `speed-bench/v100_sidecar_probe.csv`; the
verified sidecar and offline tool remain experimental.

### 7.3 Benchmark

Measure real scattered expert traffic, not sequential `dd`. Compare:

- IOPS/token;
- average request size;
- GB/s;
- read seconds/token;
- end-to-end t/s.

**Gate:** at least 10% lower read time or 20% fewer I/O operations at unchanged
numerics. Otherwise keep the sidecar tool experimental and do not complicate
the release path.

## 8. Phase 7 — prefill and deployment profiles

### 8.1 Owner-compute prefill

Batch activation matrices and per-selected-slot outputs per layer. Deduplicate
selected experts across the chunk and retain router-slot identity for the final
GPU0 reduction. Tune chunk size against:

- peer activation bytes;
- unique expert count;
- temporary compact memory;
- overlap and GPU occupancy.

### 8.2 Compare with whole-layer distribution

Run controlled A/B at prompt lengths 256, 2K, 16K, and 32K:

```text
single-process passive peer cache
single-process owner-compute
two-process layer pipeline
```

Choose documented profiles:

- `interactive`: best decode/latency architecture;
- `long-prefill`: architecture that wins large prompt ingestion;
- `single-gpu`: compatibility profile.

These are deployment recipes, not permanent alternate numerical semantics.

### 8.3 Compose integration

Only after release validation:

- update `docker-compose.yml` with the winning profile;
- expose only stable capacity knobs;
- keep diagnostics out of normal compose configuration;
- document GPU exclusivity and VRAM margins.

## 9. Phase 8 — optional CPU-host research

Do not begin until both V100s are well utilized.

Prototype expert RPC only if spare hosts have:

- wired low-latency networking;
- local model shards or enough RAM for resident experts;
- measured quantized expert throughput competitive with SSD miss latency.

Batch every selected expert owned by one host into one request per layer. Test
prefill first. Reject complete CPU layer offload as a performance path unless a
benchmark disproves the expected bottleneck.

## 10. Validation matrix

| Change type | Required gate |
| --- | --- |
| Docs/instrumentation disabled by default | affected builds + smoke |
| Allocator/index/config plumbing | CUDA/CPU builds + byte-identical fast A/B |
| Cache policy or I/O scheduling | soak + fixture + old/new binary A/B |
| Peer owner compute/reduction | CPU oracle where available + full fixtures + soak |
| Quantization or kernel math | full scored fixture; no byte-identity assumption |
| Distributed transport | two-V100 IPC smoke, reconnect soak, request after reconnect |
| Release | [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) |

Every CUDA change builds in:

```sh
docker run --rm -v "$PWD":/src -w /src \
  nvidia/cuda:12.9.1-devel-ubuntu22.04 \
  bash -lc 'apt-get update -qq && apt-get install -y -qq make gcc g++ >/dev/null; \
            make cuda CUDA_ARCH=sm_70 -j"$(nproc)"'
```

Performance runs must leave both old and new binaries available and preserve
raw logs. Correctness before speed remains non-negotiable.

## 11. Proposed commit sequence

Keep each commit independently reviewable and gateable:

1. `bench: add reproducible two-V100 architecture harness`
2. `profile: add compact expert access trace`
3. `streaming: reuse selected-expert staging scratch`
4. `streaming: remove redundant decode selected-id sync`
5. `streaming: index GPU expert slots by key`
6. `bench: add offline expert-cache policy simulator`
7. `streaming: adopt validated cache admission policy`
8. `cuda: prototype peer-owner routed MoE for Flash decode`
9. `cuda: exclusive dual-GPU expert ownership`
10. `cuda: extend peer-owner routed MoE to GLM`
11. `tools: build aligned expert streaming sidecar`
12. `streaming: coalesce expert reads through sidecar`
13. `ops: publish validated interactive and long-prefill profiles`

Do not combine cache policy, peer compute, and disk-layout changes into one
commit. Their performance attribution and rollback paths must remain separate.

## 12. Immediate next action

The corrected passive-peer profile remains the release path at 4.25 steady
t/s, Phase 3 runtime policy is skipped, and Phase 4 owner compute is a measured
end-to-end no-go despite its isolated kernel win. Phase 5 is therefore skipped.
Phase 6 is complete as a measured no-go: the offline sidecar is valid, but CUDA
read coalescing missed both I/O gates and regressed decode sharply, so no runtime
path remains. The next eligible task is **Phase 7, prefill and deployment
profiles**. GLM fixture validation remains queued until that model is restored.

Keep the passive-peer decode profile unchanged. Phase 7 should begin with
controlled prefill-only measurements and must not publish deployment defaults
until a profile passes the correctness and performance gates. Keep the sidecar
artifact out of normal runtime configuration.
