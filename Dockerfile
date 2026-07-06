# ============================================================================
# ds4 (antirez/DwarfStar 4) — CUDA build for NVIDIA Tesla V100 (Volta, sm_70)
#
# Why this exists: CUDA 13.x dropped Volta. We pin CUDA 12.9 (the last series
# that can still compile for sm_70) INSIDE the container, so your host can keep
# CUDA 13.3 untouched. The V100's small VRAM is handled by ds4's SSD streaming.
# ============================================================================

# ---------- Stage 1: build ----------
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS builder

# ds4 builds with plain `make` (NOT cmake). It only needs a C toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build from the LOCAL working tree (this repo checkout), so local patches —
# e.g. the Volta/sm_70 optimizations — are part of the image. A .dockerignore
# keeps .git, objects, models, and docs out of the build context (so doc edits
# don't invalidate this layer and force a full nvcc rebuild).
COPY . /app
WORKDIR /app

# Build-time only: let the linker resolve CUDA driver-API symbols (libcuda)
# against the toolkit's stub. The real libcuda.so.1 is provided by the host
# driver at runtime (injected by the container toolkit). We intentionally do
# NOT create a libcuda.so.1 symlink or add stubs to LD_LIBRARY_PATH — the build
# never executes CUDA, and a loadable stub would be a runtime footgun.
ENV LDFLAGS="-L/usr/local/cuda/lib64/stubs"

# --- THE CRITICAL DIFFERENCE vs the llama.cpp Dockerfile ---
# `make cuda-generic` hardcodes CUDA_ARCH=native, which needs a GPU visible at
# BUILD time. Docker builds have no GPU, and even if they did, native on a V100
# resolves to compute_70. So we target sm_70 EXPLICITLY and use the `cuda`
# target that respects a passed CUDA_ARCH.
ARG CUDA_ARCH=sm_70
RUN make cuda CUDA_ARCH=${CUDA_ARCH}

# ---------- Stage 2: runtime ----------
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# ds4 is self-contained C; it links the CUDA runtime (already in this image).
# libgomp for OpenMP, curl only for the in-container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash ds4user

# ds4 binaries live in the repo ROOT (/app), not build/bin/ like llama.cpp.
WORKDIR /app
COPY --from=builder --chown=ds4user:ds4user /app/ds4-server /app/ds4-server
COPY --from=builder --chown=ds4user:ds4user /app/ds4        /app/ds4
COPY --from=builder --chown=ds4user:ds4user /app/ds4-agent  /app/ds4-agent

# Mount points (created writable so the non-root user can use them):
#   /models  -> your downloaded GGUF (~80GB, read-only is fine)
#   /cache   -> on-disk KV cache / streaming scratch (must be WRITABLE)
RUN mkdir -p /models /cache && chown -R ds4user:ds4user /models /cache
USER ds4user

ENV MODEL_PATH="/models/ds4flash.gguf"

EXPOSE 8080

# ds4-server has NO /health route (returns 404). Probe /v1/models instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/v1/models || exit 1

# Default to the streaming server. Override any flag at `docker run` time.
# --ssd-streaming-cache-experts is a VRAM budget for hot experts: start small
# on a 16/32GB V100 and tune up while watching `nvidia-smi`.
ENTRYPOINT ["/app/ds4-server"]
CMD ["-m", "/models/ds4flash.gguf", \
     "--ssd-streaming", \
     "--ssd-streaming-cache-experts", "8GB", \
     "--ctx", "32768", \
     "--host", "0.0.0.0", \
     "--port", "8080"]
