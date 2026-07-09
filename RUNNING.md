# Running the ds4 Docker image (V100 / sm_70 build)

The image `ds4:sm70-ipc` is built from this tree (CUDA 12.9, targets sm_70)
and contains `ds4-server` (the entrypoint), `ds4`, and `ds4-agent` in `/app`.
Build or rebuild it with:

```sh
docker build -t ds4:sm70-ipc .
```

Model files live outside the image; mount the directory read-only. The GGUFs
used below live in a `./models/` directory next to the repo.

## Recommended: both GPUs via docker compose

The compose stack runs the model split across both V100s, with activations
crossing GPU-to-GPU over NVLink (see VOLTA.md for the internals):

```sh
cd glm-deepseek-v100/
docker compose up -d
docker compose logs -f     # wait for the model load, then check:
                           #   "GPU IPC inbox ready"               (worker)
                           #   "GPU IPC activation path enabled"   (coordinator)
curl http://127.0.0.1:8080/v1/models
```

The API is OpenAI-style on port 8080:

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 128
}'
```

Tunables (put them in `.env` next to docker-compose.yml, or export them):

| Variable           | Default        | Meaning                              |
|--------------------|----------------|--------------------------------------|
| `DS4_MODELS_DIR`   | `../ds4-models`| Host dir containing the GGUF         |
| `DS4_MODEL`        | Flash IQ2 file | GGUF filename inside that dir        |
| `DS4_CTX`          | `16384`        | Context tokens                       |
| `DS4_COORD_CACHE`  | `8GB`          | GPU 0 expert-cache VRAM budget       |
| `DS4_WORKER_CACHE` | `8GB`          | GPU 1 expert-cache VRAM budget       |

**Size the caches to VRAM that is actually free.** Other containers (e.g. a
llama.cpp server) may already hold many GiB — check `nvidia-smi` first. If a
process logs `CUDA tensor alloc failed: out of memory` at startup, lower its
cache budget.

Stop with `docker compose down`.

## GLM 5.2 (single process, GPU0 + GPU1 peer expert cache)

GLM's validated best-decode config is **not** a distributed layer split — it is
one `ds4-server` on GPU0 that borrows GPU1's VRAM as an expert-cache tier over
NVLink (`DS4_CUDA_PEER_EXPERT_CACHE_GB`, +12% decode vs a single-GPU cache). It
is an opt-in compose profile, so the default `docker compose up` still brings up
the DeepSeek pair. Run only the GLM service with:

```sh
docker compose up glm      # runs the glm service (+ deps), not coordinator/worker
docker compose logs -f glm
curl http://127.0.0.1:8080/v1/models
```

Do not run it alongside the DeepSeek pair — they share port 8080 and want the
same GPUs; `docker compose down` first. The peer tier needs GPU1's ~26 GB
actually free, so stop or shrink any llama.cpp server holding it (see VOLTA.md).
GLM-on-CUDA is on by default; set `DS4_GLM_CUDA_DISABLE=1` to force the old
Metal-only refusal.

| Variable                | Default                          | Meaning                        |
|-------------------------|----------------------------------|--------------------------------|
| `DS4_GLM_MODEL`         | `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` | GLM GGUF filename in models dir|
| `DS4_GLM_CACHE`         | `26GB`                           | GPU0 expert-cache VRAM budget   |
| `DS4_GLM_PEER_CACHE`    | `26`                             | GPU1 peer cache, GB integer     |
| `DS4_CTX`               | `16384`                          | Context tokens (shared)         |

## Single GPU (simpler, no distributed mode)

```sh
docker run --rm --gpus '"device=0"' -p 8080:8080 \
  -v ./models:/models:ro \
  ds4:sm70-ipc \
  -m /models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 8GB \
  --ctx 32768 --host 0.0.0.0 --port 8080
```

One-shot prompt with the CLI instead of the server:

```sh
docker run --rm --gpus '"device=0"' \
  -v ./models:/models:ro \
  --entrypoint /app/ds4 ds4:sm70-ipc \
  -m /models/<model>.gguf --ssd-streaming --ctx 8192 \
  --nothink -n 256 -p "your prompt"
```

Inspect a GGUF without loading it onto the GPU:

```sh
docker run --rm -v ./models:/models:ro \
  --entrypoint /app/ds4 ds4:sm70-ipc --inspect -m /models/<model>.gguf
```

## Environment variables (local additions in this tree)

| Variable                 | Effect                                                       |
|--------------------------|--------------------------------------------------------------|
| `DS4_CUDA_DEVICE=N`      | CUDA device for this process (default 0). Used by compose.   |
| `DS4_LOCK_FILE=path`     | Per-process instance lock; required to run two ds4 processes |
| `DS4_DIST_NO_IPC=1`      | Force TCP activations (disable the NVLink IPC path)          |
| `DS4_DIST_IPC_SLOTS`     | IPC inbox slots (default 4, max 8)                           |
| `DS4_DIST_IPC_SLOT_BYTES`| IPC slot size (default 1 MiB)                                |
| `DS4_CUDA_NO_FP16_GEMM=1`| Disable the Volta FP16 tensor-core GEMM path (A/B testing)   |
| `DS4_CUDA_NO_TF32=1`     | Disable all reduced-precision GEMM (TF32 and FP16 paths)     |

## Troubleshooting

- **"another ds4 process is already running"** — two processes share the
  default `/tmp/ds4.lock`; give each its own `DS4_LOCK_FILE` (compose does).
- **No "GPU IPC" log lines in distributed mode** — both containers need all
  GPUs visible plus `ipc: host` and `pid: host` (compose sets these); check
  neither side has `DS4_DIST_NO_IPC=1`. TCP fallback still works correctly,
  just slower per hop.
- **OOM at startup** — lower `--ssd-streaming-cache-experts`; remember the
  q8 fp16 cache also budgets up to 8 GiB per process on top of non-routed
  weights (~5 GiB each).
- **Slow first tokens** — expected with `--ssd-streaming-cold`; drop that
  flag to let the popularity preload warm the expert cache at startup.
