# DwarfStar on NVIDIA Volta (V100, sm_70)

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

### Using both GPUs

One ds4 process drives exactly one GPU (`cudaSetDevice(0)`, no device
flag). To use two V100s, run ds4's distributed mode: two processes
split per-layer over TCP, each pinned to a GPU with
`CUDA_VISIBLE_DEVICES`. Activations hop over loopback TCP (NVLink is
not used by ds4).

```sh
# GPU 0: coordinator, layers 0..30
CUDA_VISIBLE_DEVICES=0 ./ds4 -m ds4flash.gguf --ssd-streaming \
  --role coordinator --layers 0:30 --listen 127.0.0.1 1234

# GPU 1: worker, layers 31..output
CUDA_VISIBLE_DEVICES=1 ./ds4 -m ds4flash.gguf --ssd-streaming \
  --role worker --layers 31:output --coordinator 127.0.0.1 1234
```

Distributed mode mainly speeds up prefill (pipelined); decode is
~20% slower than single-process due to per-token hops. For casual use
the single-GPU `--ssd-streaming` recipe is simpler and well-tested
upstream; the distributed+streaming combination is functional but not
benchmarked upstream.

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
