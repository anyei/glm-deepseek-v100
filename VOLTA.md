# DwarfStar on NVIDIA Volta (V100, sm_70)

> **glm5.2 branch note:** GLM 5.2 inference is Metal-only upstream (the
> `ds4_gpu_glm_*` kernels exist only in `ds4_metal.m`, and
> `ds4_engine_open()` rejects GLM models on CUDA). This tree also fixes
> the branch's CUDA and CPU builds — which upstream left broken — and
> adds a **working CUDA port of the GLM kernels**, gated behind
> `DS4_GLM_CUDA_EXPERIMENTAL=1` and numerically validated against the
> CPU reference (see "GLM 5.2 CUDA port" below). The distributed GPU
> IPC fast path still declines GLM sessions (host-pointer chunked eval)
> and falls back to TCP automatically.

This tree contains local changes to run and optimize ds4 on Tesla V100
GPUs (Volta, compute capability 7.0). Upstream targets DGX Spark and
modern CUDA GPUs; Volta needs two extra considerations:

1. **CUDA 13.x dropped Volta support.** The last toolkit that can compile
   for `sm_70` is CUDA 12.9. The provided `Dockerfile` pins
   `nvidia/cuda:12.9.1` inside the container so the host driver/toolkit
   can stay current.
2. **Volta has FP16 tensor cores but no TF32.** Upstream accelerates
   FP32 GEMMs with `CUBLAS_TF32_TENSOR_OP_MATH`, which is silently
   ignored on sm_70, leaving those GEMMs at the plain FP32 FMA rate
   (~15.7 TFLOPS on V100 vs ~125 TFLOPS tensor).

## Local changes

### FP16 tensor-core path for FP32 GEMMs (`ds4_cuda.cu`)

`ds4_gpu_init()` now records the device's compute capability major
version. On sm_70 the six FP32 cuBLAS GEMM call sites (dense f32/q8
matmuls and the batched attention score/value GEMMs) go through
`cuda_sgemm()` / `cuda_sgemm_strided_batched()`, which use
`cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_16F`: inputs are
down-converted to FP16, accumulation stays FP32, and the V100 tensor
cores are engaged. This is the same speed-versus-precision trade TF32
makes on Ampere (FP16 and TF32 both carry 10 mantissa bits; FP16 has
less exponent range, which is safe here since activations, weights and
softmax-normalized scores are far below the FP16 max).

Gates (all leave non-Volta GPUs untouched):

- `--quality` mode disables it, exactly like TF32.
- `DS4_CUDA_NO_TF32=1` disables it, exactly like TF32.
- `DS4_CUDA_NO_FP16_GEMM=1` disables only this path (A/B testing knob).

Everything else in the CUDA backend was audited and is already
sm_70-clean: the WMMA indexer kernels use 16x16x16 `__half` fragments
guarded by `#if __CUDA_ARCH__ >= 700` (Volta-native), the only
shared-memory opt-in (~64KB for the CUB top-k sort) checks
`cudaDevAttrMaxSharedMemoryPerBlockOptin` at runtime and falls back
cleanly (V100 allows 96KB), and no sm_80+ intrinsics (cp.async, bf16,
`__reduce_*_sync`) are used anywhere.

### GLM 5.2 CUDA port (`ds4_cuda_glm.inc`)

Upstream implements the 32 `ds4_gpu_glm_*` entry points only in Metal.
This tree ports them to CUDA in four phases (one commit each):

1. **Infrastructure + MoE + elementwise** — the include structure,
   router/MoE kernels, and elementwise/norm kernels.
2. **Attention/KV/indexer core** — the 11 scalar-path kernels: indexer-K
   store, compact KV stores, `k_b`/`v_b` Q8_0 projections, expanded KV
   build with on-the-fly rope, dense causal attention with online
   softmax, absorbed indexed-decode attention, and the
   lightning-indexer scorers.
3. **Bounded expert streaming** — routed experts stage on demand under
   the SSD-streaming budget instead of mapping full multi-GiB tensors;
   first end-to-end generation on CUDA.
4. **IQ2_XXS experts + validation** — direct-f32 IQ2_XXS device dot for
   the 16/16/16 routed layout, plus the numerical validation below.

The scalar correctness path is complete; the remaining stubs are the
optional fast-path variants (flash/staged/batched attention, split-
group8 decode, batched low-rank QK) that `ds4_gpu_glm_kernel_caps()`
routes around. `DS4_GLM_DEBUG=1` traces any host-side validation
failure in the GLM include.

**Numerical validation (2026-07-07):** single raw-token prompt through
GLM-5.2 IQ2_XXS (211 GB GGUF, 79 layers), CPU reference on one host vs
this CUDA port on a V100 reading the same file over an sshfs/rclone
mount: top-8 tokens identical in identical order (argmax
`<|endoftext|>`, CPU 6.4550 vs CUDA 6.4602), max logit delta 0.036,
logit rms 7.151 vs 7.158 — within f32 reduction-order noise. Q2 GLM
generation and DeepSeek regressions still pass.

Running it (gated while the performance work is in progress):

```sh
DS4_GLM_CUDA_EXPERIMENTAL=1 ./ds4 -m gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 8GB \
  --ctx 1024 -p "hello"
```

**Pending (the performance phase):**

- Throughput. The port is correctness-first: the phase-3 first working
  run measured 0.15 t/s prefill / 0.16 t/s generation (Q2, cold
  streaming, 8 GB expert cache). GLM 5.2 routes 75 layers x 8 experts
  = 600 experts per token, so a per-token working set far above the
  cache budget thrashes the NVMe path; expert-cache sizing, batched
  expert staging, and the fast-path attention kernels are the levers.
- The optional fast-path kernels (flash, staged KV, batched attention,
  split-group8 decode) are still stubs — the caps mask routes to scalar
  equivalents.
- Broader validation: one prompt on IQ2_XXS (plus Q2 generation spot
  checks) so far; the 100-case official fixture should gate before the
  `DS4_GLM_CUDA_EXPERIMENTAL` gate and the startup WARNING are removed.
- Distributed: GPU IPC / distributed sessions still decline GLM and
  fall back to TCP or single-GPU.

### Docker build from the local tree (`Dockerfile`)

The image builds ds4 from this working tree (not a fresh upstream
clone), targeting `sm_70` explicitly because `make cuda-generic` uses
`-arch=native`, which cannot work in a GPU-less `docker build`:

```sh
docker build -t ds4:volta .
```

Quick smoke test (model header + CUDA init):

```sh
docker run --rm --gpus all -v /path/to/models:/models:ro \
  --entrypoint /app/ds4 ds4:volta --inspect -m /models/ds4flash.gguf
```

## Running on V100

A 32GB V100 cannot hold the ~81GB DeepSeek V4 Flash IQ2 GGUF, so SSD
streaming is required. The server default in the image:

```sh
docker run --rm --gpus '"device=0"' -p 8080:8080 \
  -v /path/to/models:/models:ro \
  ds4:volta -m /models/ds4flash.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 8GB \
  --ctx 32768 --host 0.0.0.0 --port 8080
```

`--ssd-streaming-cache-experts` is the VRAM budget for hot routed
experts: start at 8GB and raise it while watching `nvidia-smi`; the
non-routed weights, KV cache and scratch buffers need the rest of the
32GB.

### Using both GPUs (NVLink activation transport)

One ds4 process drives exactly one GPU. To use two V100s, run ds4's
distributed mode: two processes split per-layer, each pinned to a GPU
with `DS4_CUDA_DEVICE` (a local addition — unlike `CUDA_VISIBLE_DEVICES`
it keeps both GPUs visible, which CUDA IPC requires).

This tree adds a **same-host GPU IPC fast path** to the distributed
transport: when coordinator and worker run on the same machine with the
CUDA backend, hidden-state activations move GPU-to-GPU over NVLink/P2P
(measured 48 GB/s, ~3 µs per decode hop on NVLink-connected V100s)
instead of GPU → host → TCP → host → GPU. TCP still carries all control
traffic, tokens, and results, and remains the automatic per-frame
fallback (different hosts, oversized payloads, non-CUDA backends, or
`DS4_DIST_NO_IPC=1`).

How it works: each activation receiver allocates a small device "inbox"
(4 slots x 1 MiB by default; `DS4_DIST_IPC_SLOTS`,
`DS4_DIST_IPC_SLOT_BYTES`) and advertises CUDA IPC handles for it over
TCP right after HELLO (workers) or on request (worker-to-worker forward
connections). A sender maps the inbox once, then has its engine write
each hidden state directly into a slot device-to-device and ships only
an 8-byte slot descriptor in the WORK frame. Interprocess CUDA events
per slot provide flow control. Look for
`ds4: distributed: GPU IPC activation path enabled` in the logs.

```sh
# Same container/host; each process needs its own lock file.
# GPU 0: coordinator, layers 0..21
DS4_LOCK_FILE=/tmp/ds4-c.lock DS4_CUDA_DEVICE=0 ./ds4 -m ds4flash.gguf \
  --ssd-streaming --role coordinator --layers 0:21 --listen 127.0.0.1 1234

# GPU 1: worker, layers 22..output
DS4_LOCK_FILE=/tmp/ds4-w.lock DS4_CUDA_DEVICE=1 ./ds4 -m ds4flash.gguf \
  --ssd-streaming --role worker --layers 22:output --coordinator 127.0.0.1 1234
```

In Docker, run both processes in one container (or give both containers
`--ipc=host --pid=host` so CUDA IPC can cross the namespace boundary),
with `--gpus all`.

Notes and current limits:
- Decode and the non-pipelined span path use IPC; the pipelined prefill
  sender intentionally stays on TCP (v1).
- The RESULT hop back to the coordinator (final hidden state or logits)
  stays on TCP.
- Both endpoints must run this tree's binaries; for mixed-version rings
  set `DS4_DIST_NO_IPC=1`.
- SSD streaming is unaffected — experts still stream from disk.

## Verifying the FP16 GEMM path

A/B without rebuilding:

```sh
# tensor-core path (default on Volta)
./ds4-bench -m ds4flash.gguf ...

# plain FP32 baseline
DS4_CUDA_NO_FP16_GEMM=1 ./ds4-bench -m ds4flash.gguf ...
```

The GEMM wrappers only fire for batch sizes > 1 token, so the gain
shows up in prefill / prompt processing throughput, not single-token
decode.
