# Best-Case Architecture Analysis for the Two-V100 Host

Status: design analysis, not an implementation claim.
Hardware snapshot: 2026-07-10.
Primary objective: maximize sustained interactive decode for DeepSeek V4 Flash
and GLM 5.2 without weakening numerical validation, while retaining useful
prefill and CPU-only reference paths.

Related documents:

- [VOLTA.md](VOLTA.md) — current V100 implementation and measurements.
- [VALIDATION.md](VALIDATION.md) — correctness and performance gates.
- [ROADMAP.md](ROADMAP.md) — existing backlog.
- [V100_IMPLEMENTATION_PLAN.md](V100_IMPLEMENTATION_PLAN.md) — phased plan
  derived from this analysis.

## 1. Executive decision

The best long-term architecture for this machine is **not** whole-layer
streaming, a CPU layer worker, or a second V100 used only as passive storage.
It is:

> **One process, GPU0 owning the sequential transformer graph, and a unified
> expert cache distributed exclusively across both V100s. Each selected routed
> expert is computed on the GPU that owns its weights. GPU1 returns per-slot
> expert outputs over NVLink, never the expert weights; GPU0 reduces slots
> in canonical router order so cache placement cannot change numerics.**

This is intra-layer MoE expert parallelism, not the existing distributed
whole-layer mode.

The strongest decode candidate available in the code today is:

> **One CUDA process on GPU0, the largest safe local expert cache, GPU1 as a
> peer expert-cache tier over NVLink, eight SSD reader workers, and no pinned
> host cache.**

It is the validated best GLM configuration. DeepSeek supports the same shared
streaming path, but its passive-peer configuration still needs the controlled
architecture sweep in the implementation plan before it is called the best
DeepSeek deployment.

For long-prefill-only jobs, the existing two-process layer split remains a
useful separate profile because it can pipeline prompt chunks. It should not be
the default interactive-decode architecture until it beats the single-process
peer-cache profile in a controlled A/B.

## 2. Machine constraints

### 2.1 Compute and interconnect

| Resource | Observed configuration | Consequence |
| --- | --- | --- |
| GPU0 | Tesla V100-SXM2, 32 GiB, `sm_70` | FP16 tensor cores; no TF32 |
| GPU1 | Tesla V100-SXM2, 32 GiB, `sm_70` | Equal compute and memory capacity |
| GPU link | `NV2`; measured around 48 GB/s in this fork | Cheap activation/partial-output exchange |
| CPU | i9-10850K, 10C/20T | Suitable for I/O orchestration, not a competitive layer backend |
| RAM | 46 GiB visible | Too small for a large pinned L2 while leaving safe OS headroom |
| CPU RAM peak | about 46.9 GB/s at stock DDR4-2933 dual channel | Faster than PCIe host-to-GPU, but not NVLink/HBM |
| NVMe | about 2.45–2.5 GB/s best cold sequential; 1.9–2.0 GB/s real expert pattern | Decode misses are device-bound |

CUDA 12.9 is the final toolkit supporting Volta. CUDA 13.x on the host must not
be used to build `sm_70`; builds stay in the pinned CUDA 12.9 container.

### 2.2 Current host health is itself a performance problem

At analysis time:

- `/mnt/full-models` was 98% full, with only about 6.8 GiB free.
- Swap usage was about 22 GiB out of 23 GiB.
- Only about 21 GiB RAM was reported available.

These conditions make benchmark variance and allocation failures more likely.
They also make an expert-repacked sidecar impossible to create on the current
filesystem. Before serious optimization work, free at least 15–20% of the
NVMe, eliminate active memory pressure, and verify that swap-in/swap-out is
zero during benchmarks. Used swap alone is not proof of current pressure, but
it must be measured rather than ignored.

## 3. The actual granularity

### 3.1 Compute dependency

Decode is sequential at the transformer-layer level:

```text
token N, layer L
  attention
  residual + FFN norm
  router
  shared expert + selected routed experts
  expert reduction + residual
  -> layer L+1
```

The next layer's exact expert IDs cannot generally be known before the current
layer produces its output. The next token cannot begin ordinary decode before
the current token reaches logits and sampling.

### 3.2 Current distributed boundary

The current external distributed API dispatches a contiguous range of complete
layers. Its minimum unit is one complete layer for one token. DeepSeek Flash
moves a 4096-element F32 hidden vector, or 16 KiB/token, between ranges.
Prefill dispatches chunks of tokens and can pipeline ranges.

This boundary is good for model capacity and long-prefill pipelining. It is not
the best boundary for two equal GPUs connected by NVLink when routed MoE
weights dominate the model.

### 3.3 Current SSD boundary

SSD streaming keeps dense/shared/control weights available and stages only the
selected routed experts. A logical expert consists of gate, up, and down
weights. DeepSeek Flash selects six of 256 experts per layer; GLM 5.2 selects
eight.

For the measured DeepSeek quant, one complete routed expert is about 6.75 MiB.
Loading every routed expert in a layer would be roughly 1.69 GiB; loading six
selected experts is roughly 40.5 MiB before cache hits. Selected-expert
streaming is therefore the correct fundamental storage granularity.

GLM's routed experts are about 11.81 MiB each. Across 75 routed layers and eight
experts, its uncached working set is about 7.1 GiB per token. This is why cache
admission and expert bytes dominate GLM decode.

## 4. Evidence from the current tree

**Post-analysis measurement update (2026-07-10):** the controlled DeepSeek
context-256/96-token sweep selected passive peer caching. Single GPU reached
2.37 median steady t/s; the corrected 8 GiB local + 26 GiB peer profile reached
4.25 (4.10–4.26), +79%. The initial peer result of 3.77 was superseded after an
expert trace found that a grow check compared the local target against combined
capacity and left only about 3 local slots beside 3,944 peer slots. Restoring
701+3,944 slots improved steady decode another 12.7%. Whole-layer distribution
screened at 0.14 t/s and is a decode no-go. A 181,632-row policy trace exactly
replayed 45,472 hits, 16,466 misses and 116,544,503,808 bytes; none of segmented
LRU, TinyLFU, equal layer quotas, decode protection, owner balancing, or top-K
replication reduced bytes, so runtime cache-policy work fails its 20% gate on
this workload. See `speed-bench/v100_architecture.csv` and
`V100_IMPLEMENTATION_PLAN.md` for qualifications. A subsequent all-peer
DeepSeek diagnostic computed six peer-owned slots directly on GPU1 and returned
unreduced rows with exact GPU0 equality. Across three runs its isolated
activation+compute+return+estimated-join path was 26–36% faster than measured
peer-weight-copy plus GPU0 MoE, passing the prototype's 15% microbenchmark gate.
The subsequent end-to-end replacement failed: passive peer delivered 4.04
median steady t/s versus 3.99 for owner replacement (-1.2%), despite
byte-identical logprobs. Synchronous control/device switching consumed the
isolated gain, so replacement was reverted and exclusive ownership is a no-go
on this implementation/host.

Recorded measurements include:

- DeepSeek Flash, one V100, 8 GiB streaming budget: 2.28–2.37 t/s reference
  after parallel reads.
- A quick old/new regression A/B on this host: 2.44 vs 2.53 steady t/s over
  seven steady decode tokens; this is a regression check, not a canonical
  benchmark.
- GLM before performance work: about 0.16 t/s.
- GLM after cache keep-alive, layer-major prefill, and parallel reads: about
  0.35 t/s.
- GLM with a 26 GiB peer expert cache: about 0.45 t/s.
- Actual expert-pattern disk reads: about 1.9–2.0 GB/s against a 2.45 GB/s
  synthetic ceiling.
- GLM decode attribution: roughly 2.0 seconds of expert reads plus roughly
  0.8 seconds of GPU compute/pipeline drain per token.

The implication is important: replacing the reader pool with a different I/O
API cannot yield a multi-fold gain. The high-value variables are bytes missed,
cache ownership, unnecessary weight movement, and GPU compute.

## 5. Candidate architectures

### A. One V100 plus SSD

**Description:** GPU0 computes everything; selected experts are cached locally
and misses come from NVMe.

**Strengths:** smallest implementation, no cross-GPU coordination, validated.

**Weaknesses:** leaves 32 GiB HBM and a full V100 idle; miss rate remains high.

**Verdict:** safe baseline, not the best use of this machine.

### B. GPU0 compute plus GPU1 passive peer cache (strongest available candidate)

**Description:** GPU1 stores extra expert slots. On a hit, the expert weights
are copied over NVLink into GPU0's compact buffers and GPU0 computes them.

**Strengths:** one graph and one KV cache; large combined HBM tier; already
implemented and soak-tested for GLM; low protocol complexity.

**Weaknesses:** GPU1 compute is idle. Every peer hit moves megabytes of expert
weights over NVLink, even though the input and result vectors are only tens of
KiB. GPU0 still performs all routed-MoE compute.

**Verdict:** deployment choice now, transition architecture long term.

### C. Two-process whole-layer split

**Description:** each V100 owns a contiguous layer range and its own cache;
activations cross via CUDA IPC/NVLink.

**Strengths:** uses both compute engines; doubles independently useful cache
capacity; excellent for pipelined long prefill; already works for DeepSeek.

**Weaknesses:** decode remains sequential across ranges; process duplication
costs memory; cache ownership is constrained by layer rather than actual
expert popularity; session/recovery/protocol complexity is higher. GLM
whole-layer distribution is not yet enabled.

**Verdict:** maintain as capacity/prefill profile. Benchmark against B; do not
assume it wins decode.

### D. Large pinned-host expert cache

**Description:** cache misses in non-reclaimable system RAM and copy over PCIe.

**Strengths:** faster than the NVMe; implemented.

**Weaknesses:** this host lacks safe RAM headroom. PCIe practical throughput is
about 12 GB/s, far below NVLink/HBM. Pinned pages can worsen system pressure and
swap behavior. A peer cache made the measured host-L2 residual hit rate
negligible.

**Verdict:** off by default on this machine. Reconsider only after a major RAM
upgrade and only when GPU1 cannot be used.

### E. CPU complete-layer workers

**Description:** spare CPU PCs own complete layer ranges.

**Strengths:** capacity aggregation; uses existing distributed boundary.

**Weaknesses:** every decode token waits for CPU attention and MoE compute;
network latency is paid in the sequential path; every host needs model access.

**Verdict:** useful for correctness/capacity experiments, not best throughput.

### F. Remote CPU expert workers

**Description:** GPU routes experts; CPU hosts compute selected expert
contributions and return output vectors.

**Strengths:** expert jobs are independent; network payloads are much smaller
than weights; hosts can own disjoint expert shards.

**Weaknesses:** up to one synchronization point per routed layer, CPU quantized
matvec performance, network tail latency, and complex failure semantics. Even
fast Ethernet latency accumulates over roughly 43/79 sequential layers.

**Verdict:** interesting optional cold tier for prefill/offline work, not the
primary design for two local V100s.

### G. Unified two-GPU expert ownership with owner-side compute (chosen)

**Description:** one process owns the graph. Expert slots are exclusive across
GPU0 and GPU1. After routing, selected experts are partitioned by owner. Each
GPU computes the experts whose weights it owns. GPU1 receives the normalized
activation and returns one output vector per peer-owned selected slot over
NVLink. GPU0 reduces every selected slot in canonical router order. SSD misses
are admitted to one owner, not mirrored.

**Strengths:** uses both HBM pools and both compute engines; removes peer-hit
weight copies; transfers KiB-scale activations/results instead of MiB-scale
weights; can overlap GPU0 shared/local experts with GPU1 experts; preserves a
single session/KV owner and avoids the distributed network protocol.

**Weaknesses:** multi-device scheduling inside a codebase designed around one
CUDA device; output reduction order can alter floating-point results; load
balance changes every layer; SSD remains shared; failure handling must not
silently use partial expert results.

**Verdict:** best architectural fit and highest-upside implementation target.

## 6. Chosen architecture in detail

### 6.1 Control plane

A single process remains bound to GPU0 for the transformer graph, attention,
KV cache, shared expert, output head, sampling, and API state. It opens GPU1 as
an explicit expert device. GPU visibility remains global.

Do not generalize the entire tensor API to arbitrary devices initially. Add a
narrow peer-expert context with explicit device ownership, streams, events, and
allocation lifetime. This limits risk to routed-MoE staging and execution.

### 6.2 Exclusive cache ownership

Each cached `(model, layer, expert, geometry)` key exists in one GPU tier:

```text
GPU0 local expert slot
or
GPU1 owner-compute expert slot
or
not resident
```

Do not duplicate the same expert in both HBM pools unless profiling proves a
specific hot-expert replication policy is worthwhile. Exclusive ownership
maximizes effective capacity.

Admission should be cost-aware rather than plain global LRU:

- protect frequently reused experts;
- keep per-layer statistics because popularity is layer-specific;
- avoid letting a large prefill flush the decode working set;
- include owner load when choosing GPU0 vs GPU1;
- reserve a small amount of free capacity for bursts and allocation safety.

### 6.3 Owner-side MoE execution

For one decode layer:

1. GPU0 computes attention, FFN norm, router IDs, and routing weights.
2. Host-visible IDs are obtained without adding a new global D2H
   synchronization; use existing router output availability or an async event.
3. Selected experts are partitioned into GPU0 hits, GPU1 hits, and misses.
4. Misses are read once and admitted to the chosen owner.
5. GPU0 computes the shared expert and GPU0-owned routed experts.
6. GPU1 receives the normalized activation and compact IDs/weights, then
   computes its owned routed experts directly from its resident slots.
7. GPU1 returns one output vector for each peer-owned selected router slot.
8. GPU0 reduces all local and peer slot outputs in the baseline's canonical
   slot order, then applies the residual. Cache admission/eviction must never
   change reduction grouping.

DeepSeek Flash sends a 16 KiB normalized input and receives at most six 16 KiB
slot outputs (96 KiB) per routed layer. GLM sends 24 KiB and receives at most
eight 24 KiB slot outputs (192 KiB). Even the worst case is far cheaper than
copying several 6.75/11.81 MiB expert triples per peer hit. Returning a single
pre-summed peer partial would save only a small transfer but would make FP
reduction grouping depend on cache ownership, so it is explicitly rejected.

### 6.4 SSD path

The NVMe remains the backing store. Eight reader workers are a good current
default; thread count is no longer a primary optimization axis.

A miss should target its final owner directly. Avoid SSD -> host -> GPU1 -> GPU0
weight movement. Pinned worker buffers and `cudaMemcpyDefault`/peer-aware
copies remain useful, but completion must be tracked per owner stream.

### 6.5 On-disk expert layout

The existing GGUF layout causes three scattered logical reads per expert
(gate/up/down). The preferred future layout is an optional sidecar or repacked
GGUF section ordered by `(layer, expert)` with gate/up/down adjacent and
alignment suitable for O_DIRECT.

Use an opt-in sidecar first. Preserve the canonical GGUF and numerical tensor
bytes exactly; store a manifest containing model identity, tensor offsets,
checksums, quant types, geometry, and alignment. The loader must fall back to
normal GGUF offsets if the sidecar is absent or invalid.

This is operationally blocked until substantial disk space is freed or a larger
NVMe is installed.

### 6.6 Prefill

Keep layer-major prefill and deduplicate selected experts over each chunk. For
large batches, owner-side compute still applies, but transfer one activation
matrix and per-selected-slot output matrices per layer rather than per-token
messages; preserve canonical slot reduction on GPU0.

The existing distributed layer pipeline may still beat owner-side MoE for very
long prompts. Retain both profiles and choose from measured workload thresholds
rather than forcing one permanent semantic variant.

## 7. Cache policy: challenge to the naive design

A larger cache is not automatically a better cache. The current tiers fill from
the same miss stream, which explains why equal-sized host and GPU tiers add
little. A policy must answer four questions:

1. **Admission:** should this miss displace anything?
2. **Victim:** which resident expert has the lowest future value?
3. **Owner:** which GPU should store and compute it?
4. **Replication:** is this expert hot enough to justify a duplicate?

Start with an offline trace simulator before changing runtime behavior. Record
`(token, layer, expert IDs, hit tier, age, bytes, latency)` and replay candidate
policies:

- LRU baseline;
- segmented LRU (probation/protected);
- TinyLFU-style frequency admission plus LRU eviction;
- per-layer quotas;
- decode-protected vs prefill-ephemeral regions;
- owner-balanced admission;
- optional top-K replicated experts.

Optimize predicted bytes read per token first. Policy CPU cost must remain well
below the milliseconds saved.

## 8. Expected performance ceiling

No exact speedup should be promised before measurement. A useful bound follows
from the GLM attribution:

```text
current token ≈ 2.0 s reads + 0.8 s compute/drain ≈ 2.8 s
```

If policy/layout cuts read time by 25% and owner-side peer compute cuts the
compute slice by 25%, the illustrative result is:

```text
1.5 s reads + 0.6 s compute ≈ 2.1 s => about 0.48 t/s
```

That modest example is already meaningful. A larger gain requires a large
reduction in SSD bytes, not merely faster peer compute. The absolute upper
bound with all selected experts resident would be dominated by compute and
could be much higher, but the routed pool is larger than combined usable HBM.

For DeepSeek, establish the same attribution before forecasting. The quick
2.53 steady-t/s result is too short to infer cache-policy headroom.

## 9. Risks and invariants

### Numerical invariants

- Selected IDs and routing weights must be unchanged.
- Every selected expert contributes exactly once.
- Per-expert slot outputs must be reduced in canonical router-slot order.
- Cache ownership and hit/miss decisions must not change FP grouping or model
  semantics.
- A peer failure must fail the layer, never continue with a missing partial.

### Memory invariants

- GPU0 reserves room for dense weights, KV, scratch, and context growth.
- GPU1 reserves driver/stream/workspace margin.
- Cache allocation responds to actual free memory and can shrink safely.
- No pinned-host allocation is enabled by default on this 46 GiB host.

### Concurrency invariants

- Every thread binds the intended CUDA device before runtime calls.
- An expert slot cannot be evicted while an owner stream reads it.
- SSD staging completion and compute completion use explicit events.
- Teardown waits for both devices and reader workers.

### Operational invariants

- The canonical GGUF remains read-only.
- Sidecars are model-identity checked and disposable.
- CUDA 12.9 and `sm_70` stay explicit.
- All performance claims include prompt length, context, cache state, and
  generated-token count.

## 10. Review of this recommendation

The recommendation was challenged against the following alternatives:

- **Why not whole-layer streaming?** It reads 256 experts when only six/eight
  are selected; rejected.
- **Why not two-process distribution as the default?** It is excellent for
  prefill but decode remains sequential and it constrains cache ownership by
  layer; retain as a measured alternative.
- **Why not host RAM?** Current RAM and swap state make a large pinned cache
  unsafe, and peer HBM/NVLink is superior.
- **Why not more I/O threads or io_uring first?** The current real pattern is
  already near the device ceiling; at most an incremental gain.
- **Why not CPU helpers first?** Network and CPU latency repeat at every layer;
  the idle local V100 is strictly more attractive.
- **Why not keep GPU1 passive?** Moving an input plus per-slot outputs is orders
  of magnitude smaller than moving peer-resident expert weights, and GPU1
  compute is otherwise wasted. Returning per-slot outputs also avoids making
  reduction order depend on placement.
- **Why not promise a large speedup?** GLM is still read-dominated. Owner-side
  compute is necessary but cache policy and byte reduction determine the main
  ceiling.

The strongest unresolved question is whether dynamic expert ownership and
cross-device launch/event overhead are low enough on Volta to beat the simpler
peer-copy path at the current hit rate. The implementation plan therefore puts
a minimal owner-compute prototype and an explicit A/B gate before broad
refactoring.
