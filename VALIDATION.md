# GLM/DeepSeek CUDA Streaming — Validation Playbook

How to validate this tree's GLM CUDA port and streaming changes, with the
exact commands and the reference numbers they were measured against.

Reference hardware: Tesla V100-SXM2-32GB, models on local NVMe (device
ceiling for the streaming access pattern: ~2.45 GB/s at QD4+, ~1.98 GB/s
at QD1 for 4 MiB direct reads). Canonical tree:
`~/server/git-projects/ds4`, branch `glm5.2`. The image built from it is
`ds4:sm70-glm-fix` below; substitute your tag.

Models used throughout:

| Purpose | File |
| --- | --- |
| GLM benchmarks | `/mnt/full-models/ds4/gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` |
| GLM numerical oracle | `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` (on the oracle box) |
| DeepSeek regression | `~/server/git-projects/ds4-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` |

## 0. Environment notes (read first)

- On hosts with a stale CDI spec, fresh `docker run --gpus` fails with
  `failed to fulfil mount request: .../libnvidia-egl-gbm.so...`. Either
  regenerate the spec (root: `nvidia-ctk cdi generate
  --output=/etc/cdi/nvidia.yaml`) or use the legacy runtime, which every
  command below does:

  ```sh
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
  ```

- Never pipe `docker build` into `tail`/`grep` — the pipe eats the exit
  code and the tag silently stays on the previous image. Build as in §1
  and verify the binary content.
- Benchmarks need the GPU and its VRAM to themselves: stop any production
  inference containers first, restart them after.
- GLM on CUDA is gated: every GLM run needs `DS4_GLM_CUDA_EXPERIMENTAL=1`.

## 1. Build gates

CPU build links clean (catches missing symbols the CUDA build hides):

```sh
cd ~/server/git-projects/ds4
make cpu -j12 2>&1 | grep -cE 'error|undefined'   # expect: 0
```

CUDA image, with explicit failure reporting and a content check — pick a
string literal unique to the change being validated and confirm it is in
the shipped binary:

```sh
docker build -t ds4:sm70-glm-fix . > /tmp/ds4-build.log 2>&1 \
  && echo BUILD_OK || { echo BUILD_FAILED; grep -m5 error /tmp/ds4-build.log; }
docker run --rm --entrypoint grep ds4:sm70-glm-fix -c '<unique new string>' /app/ds4
# expect: 1
```

## 2. GLM numerical validation (CPU oracle vs CUDA logits)

Single raw-token prompt, first-token logits, CPU reference vs the CUDA
graph. This is the correctness gate for kernel changes.

CPU oracle (any box that holds the model on local disk; a CPU-only build
lives in `~/ds4-oracle` on the oracle host):

```sh
ssh <oracle-host> 'cd ~/ds4-oracle && DS4_LOCK_FILE=/tmp/ds4-oracle.lock \
  ./ds4 --cpu --first-token-test --raw \
  -m ~/glm-models/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  -p "hello" 2>&1 | tail -14'
```

CUDA side, dumping the full logit vector:

```sh
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e DS4_GLM_CUDA_EXPERIMENTAL=1 \
  -v <model-dir>:/models:ro -v <outdir>:/out \
  --entrypoint /app/ds4 ds4:sm70-glm-fix \
  -m /models/<same gguf> --cuda \
  --ssd-streaming --ssd-streaming-cache-experts 8GB \
  --ctx 1024 --raw -p "hello" --dump-logits /out/cuda-logits.json
```

Compare: top-8 token ids from the oracle's printout against the top-8 of
`cuda-logits.json`.

PASS: identical top-8 ids in identical order; max abs logit delta within
f32 reduction-order noise (reference 2026-07-07: argmax `<|endoftext|>`
154820, CPU 6.4550 vs CUDA 6.4602, max delta 0.036, rms 7.151 vs 7.158).

## 3. GLM decode benchmark

```sh
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e DS4_GLM_CUDA_EXPERIMENTAL=1 \
  -e DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE=1 \
  -v /mnt/full-models/ds4/gguf:/models:ro \
  --entrypoint /app/ds4 ds4:sm70-glm-fix \
  -m /models/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf --cuda \
  --ssd-streaming --ssd-streaming-cache-experts 26GB \
  --ctx 1024 --raw -p "Once upon a time in a small village by the sea," -n 64
```

Read the final `ds4: prefill: X t/s, generation: Y t/s` line and the
generated text (raw mode must produce a coherent story continuation —
gibberish means a kernel or staging regression, whatever the speed says).

Reference numbers (Q2 from local NVMe):

| Config | Generation |
| --- | --- |
| as above | ≥ 0.35 t/s |
| + `-e DS4_CUDA_HOST_EXPERT_CACHE_GB=28` (host L2) | ≥ 0.40 t/s |
| + `-e DS4_CUDA_STREAM_READ_THREADS=0` (serial-read control) | ~0.23 t/s |

With the L2 enabled, startup must print
`ds4: CUDA pinned host expert cache: ~2422 experts / 27.99 GiB`.

## 4. GLM prefill benchmark (batched layer-major path)

Same command with a long raw prompt (~190 tokens; e.g. the first
paragraph of any story text) and `-n 8`.

PASS:
- prefill ≥ 1.5 t/s at ~188 prompt tokens (short 25-token prompts: ~0.33);
- the verbose log shows ONE staging wave per layer with
  `slots=<n_tokens*8> compact=<unique experts>` — per-token waves of
  `slots=8` during prefill mean the token-major path regressed
  (`DS4_GPU_GLM_CAP_TOKEN_MAJOR_PREFILL` must stay Metal-only).

## 5. Cache-stats analysis

Aggregate the verbose staging lines into per-token hit/miss/bytes stats:

```sh
python3 speed-bench/expert-cache-stats.py run.log
```

Columns: L1 `hits`, `miss`, `host` (L2 hits), `direct`, combined `hit%`,
`MiB-read` (actual storage bytes) per staging wave, plus a steady-state
summary over the last decode tokens.

Red flags:
- `global_budget=0 ... direct=8` on decode waves — the GPU expert cache
  died mid-run (regression of the keep-alive fix, commit `280c967`);
- steady-state L1 hit rate well under ~35% at a 26 GiB budget;
- combined L1+L2 under ~55% with a 28 GiB L2.

## 6. DeepSeek regression smoke (shared streaming path)

GLM streaming changes share `begin_compact_load`, the reader pool and the
expert caches with DeepSeek — always smoke it:

```sh
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v ~/server/git-projects/ds4-models:/models:ro \
  --entrypoint /app/ds4 ds4:sm70-glm-fix \
  -m /models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 8GB \
  --ctx 1024 -p "Reply with exactly the word: hello" -n 16
```

PASS: exit 0, coherent reasoning text, generation ≥ ~2.2 t/s
(reference 2.28–2.37 after the parallel-read pool; 1.41 before it).

## 7. Runtime probes (when a number looks wrong)

Reader pool engaged? Expect 3 base threads + 8 workers = 11:

```sh
pid=$(pgrep -f 'ds4 -m /models/GLM' | head -1)
awk '/Threads/{print $2}' /proc/$pid/status
```

NVMe demand during decode (10 s window; this measures demand, not the
device ceiling — compare against bytes/token × tokens/s, not ~2.45 GB/s):

```sh
a=$(awk '$3=="nvme0n1"{print $6}' /proc/diskstats); sleep 10
b=$(awk '$3=="nvme0n1"{print $6}' /proc/diskstats)
echo $(( (b-a)*512/10/1048576 )) MB/s
```

Per-stage decode timing (one layer) plus the streaming-staging split:

```sh
-e DS4_METAL_DECODE_STAGE_PROFILE=40 \
-e DS4_METAL_GLM_STREAMING_ASYNC_PROFILE=1
```

Interpretation caveat: `shared_gate_up_swiglu` reads as 8-30 ms/layer
under streaming, but that is NOT the shared-expert GEMV (which is
sub-ms) — the per-layer expert staging IO runs between the `router`
and `shared_gate_up_swiglu` stage boundaries, so the profiler
attributes the whole ~95 MiB expert load to this stage. Verified
2026-07-07: dense/shared weights are fd-cached in VRAM after first
touch (`DS4_CUDA_WEIGHT_CACHE_VERBOSE=1` shows `fd-cached` lines, zero
`CUDA direct` fallbacks). The number is therefore a per-layer
staging-cost probe: it should shrink as cache hit rates rise, and a
genuine dense-weight regression would instead show `CUDA direct`
fallback lines in the weight-cache-verbose output.

Direct I/O status (`align=4096` at startup; any later `disabled` line is
a regression):

```sh
-e DS4_CUDA_WEIGHT_CACHE_VERBOSE=1
```

## 8. Release gate (before removing the EXPERIMENTAL flag)

The full 100-case official continuation fixture
(`tests/`, see QA_BEFORE_RELEASES.md) must pass on the CUDA path before
`DS4_GLM_CUDA_EXPERIMENTAL` and the startup WARNING are removed. §2-§6
are per-change gates; this one is the ship gate.
