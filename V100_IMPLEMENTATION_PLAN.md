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

Current `/mnt/full-models` free space is about 6.8 GiB (98% used). Before
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

- stop unrelated model processes;
- verify swap-in/swap-out (`vmstat 1`) is zero during benchmarks;
- reboot or deliberately reset the environment if old swapped pages make runs
  noisy;
- do not enable `DS4_CUDA_HOST_EXPERT_CACHE_GB` on this host during canonical
  tests;
- record configured DIMM speed with privileged DMI access.

**Gate:** no active swapping and at least 8 GiB host headroom throughout a run.

### 1.3 Canonical benchmark harness

Add `speed-bench/v100_ab.sh` or an equivalent small harness that records:

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

Benchmark on identical prompts:

1. one V100 + SSD;
2. GPU0 + passive GPU1 peer cache;
3. two-process layer split over CUDA IPC/NVLink.

DeepSeek already supports all three. GLM first compares 1 and 2; whole-layer
GLM distribution is a later independent experiment.

**Deliverable:** `speed-bench/v100_architecture.csv` plus a short interpretation
in the analysis document or `VOLTA.md`.

**Decision:** passive peer cache remains production default unless layer split
wins steady decode by at least 10% without unacceptable first-token or prefill
regressions.

## 3. Phase 2 — instrumentation and low-risk hot-path cleanup

This phase should not alter kernel math.

### 3.1 Expert trace format

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

Thread already-known host router IDs through the staging API instead of reading
selected IDs back from the GPU solely for validation. Preserve a diagnostic
cross-check mode that compares host/GPU IDs.

**Gate:** full fixture if staging semantics can change; otherwise the fast
byte-identical A/B plus a long soak. Confirm approximately one blocking sync per
routed layer is removed.

### 3.4 Hash the local GPU cache index

The host L2 already uses a hash index; the GPU expert cache still scans slots.
Add a host metadata index keyed by full model/layer/expert identity and verify
the slot record on every hit. Keep LRU metadata authoritative.

**Gate:** byte-identical A/B and cache hit/miss sequence identical to baseline.

## 4. Phase 3 — policy simulation before policy implementation

### 4.1 Offline simulator

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

Experts currently have uniform geometry within a model variant, but the design
should calculate byte cost. Select the simplest policy achieving most of the
simulated reduction. Reject policies whose metadata/decision cost approaches
the expected savings.

**Gate to runtime implementation:** at least 20% simulated SSD-byte reduction
on a representative decode trace or a clear prefill-pollution fix.

### 4.3 Runtime policy

Implement behind a diagnostic environment gate initially, then remove the gate
once one release path is selected. Do not retain permanent semantic variants.

**Gate:** soak, fixture score, old/new binary A/B, and measured bytes/token.
Cache policy must be numerically exact.

## 5. Phase 4 — minimal owner-compute prototype

This is the highest-risk/highest-upside phase. Keep it narrow.

### 5.1 Narrow peer-expert context

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

- Reuse existing routed-MoE dot kernels where possible, but add an unreduced
  output mode that preserves one vector per router slot.
- Preserve router-slot identity across compaction and peer dispatch.
- Combine every routed slot in the baseline order; this is mandatory, not best
  effort, because cache ownership must not alter FP grouping.
- Add a diagnostic mode that runs baseline GPU0 MoE and peer MoE on the same
  input and reports max/RMS deltas before enabling end-to-end generation.
- Any peer CUDA error fails the layer; no partial fallback after compute begins.

### 5.4 Microbenchmarks

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

Modify staging so one expert miss can be served by one contiguous read when the
sidecar is present. Upload subranges to final owner buffers without extra
copies where alignment permits.

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

The next coding task should be **Phase 0.3, the canonical benchmark harness**,
followed by the Phase 1 architecture/cache sweep. It produces the baseline
needed to decide whether owner-side compute, cache policy, or disk layout is
actually the first implementation win for DeepSeek on this exact host.

Do not start by writing peer kernels. First make the current three
architectures reproducibly comparable and capture expert traces that can prove
a cache policy before placing it in the hot path.
