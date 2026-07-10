# Running the ds4 Docker image (V100 / sm_70 build)

The image `ds4:sm70-ipc` is built from this tree (CUDA 12.9, targets sm_70)
and contains `ds4-server` (the entrypoint), `ds4`, and `ds4-agent` in `/app`.
Build or rebuild it with:

```sh
docker build -t ds4:sm70-ipc .
```

Model files live outside the image; mount the directory read-only. The GGUFs
used below live in a `./models/` directory next to the repo.

## Recommended interactive profile: both GPUs, one process

The default compose service runs the graph on GPU0 and uses GPU1 as a 26 GiB
passive expert-cache tier over NVLink. This is the validated interactive winner
for DeepSeek Flash: 4.25 median steady decode t/s versus 2.37 on one V100.

```sh
cd glm-deepseek-v100/
docker compose up -d
docker compose logs -f interactive
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

| Variable                | Default         | Meaning                          |
|-------------------------|-----------------|----------------------------------|
| `DS4_MODELS_DIR`        | `../ds4-models` | Host dir containing the GGUF     |
| `DS4_MODEL`             | Flash IQ2 file  | GGUF filename inside that dir    |
| `DS4_CTX`               | `16384`         | Context tokens                   |
| `DS4_INTERACTIVE_CACHE` | `8GB`           | GPU0 expert-cache budget         |
| `DS4_PEER_CACHE`        | `26`            | GPU1 peer-cache budget in GiB    |

**Size the caches to VRAM that is actually free.** Other containers may already
hold many GiB; check `nvidia-smi` first. Stop with `docker compose down`.

## Long-prefill profile: distributed layers

For prefill-heavy or offline jobs, the existing layer split is 23–26% faster
at 16K–32K prompt ingestion. It is not an interactive profile: distributed
decode measured about 0.14 t/s. Stop the interactive service, then explicitly
start only the two profiled services:

```sh
docker compose down
docker compose --profile long-prefill up -d coordinator worker
docker compose logs -f coordinator worker
# Expect "GPU IPC inbox ready" and "GPU IPC activation path enabled".
```

The validated split is GPU0 layers `0:21`, GPU1 `22:output`, with 8 GiB expert
caches, 4096-token prefill chunks, and a three-chunk flow window. A 2048-token
chunk regressed 16K prefill from 68.3 to 37.6 t/s by rereading substantially
more expert data.

| Variable                   | Default | Meaning                           |
|----------------------------|---------|-----------------------------------|
| `DS4_COORD_CACHE`          | `8GB`   | GPU0 expert-cache budget          |
| `DS4_WORKER_CACHE`         | `8GB`   | GPU1 expert-cache budget          |
| `DS4_LONG_PREFILL_CHUNK`   | `4096`  | Session and pipeline chunk tokens |
| `DS4_LONG_PREFILL_WINDOW`  | `3`     | End-to-end chunks in flight       |

## GLM 5.2 (single process, GPU0 + GPU1 peer expert cache)

GLM's validated best-decode config is **not** a distributed layer split — it is
one `ds4-server` on GPU0 that borrows GPU1's VRAM as an expert-cache tier over
NVLink (`DS4_CUDA_PEER_EXPERT_CACHE_GB`, +12% decode vs a single-GPU cache). It
is an opt-in compose profile, so the default `docker compose up` still brings up
the DeepSeek interactive service. Run only the GLM service with:

```sh
docker compose up glm      # runs the glm service (+ deps), not coordinator/worker
docker compose logs -f glm
curl http://127.0.0.1:8080/v1/models
```

Do not run it alongside a DeepSeek profile — they share port 8080 and want the
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

The compatibility compose profile hides GPU1:

```sh
docker compose down
docker compose --profile single-gpu up -d single-gpu
```

The equivalent direct invocation is:

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
