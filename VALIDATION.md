# GLM/DeepSeek CUDA Streaming — Validation Playbook

How to validate this tree's GLM CUDA port and streaming changes, with the
exact commands and the reference numbers they were measured against.

Reference hardware: Tesla V100-SXM2-32GB, models on local NVMe (device
ceiling for the streaming access pattern: ~2.45 GB/s at QD4+, ~1.98 GB/s
at QD1 for 4 MiB direct reads). Canonical tree: `glm-deepseek-v100/`. The
image built from it is `ds4:sm70-glm-fix` below; substitute your tag.

Models used throughout:

| Purpose | File |
| --- | --- |
| GLM benchmarks | `glm-deepseek-v100/gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` |
| GLM numerical oracle | `GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf` (on the oracle box) |
| DeepSeek regression | `./models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` |

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
- GLM on CUDA is on by default since the §8 gate passed;
  `DS4_GLM_CUDA_EXPERIMENTAL=1` in older commands is harmless. On trees
  from before the gate removal it is still required.

## 1. Build gates

CPU build links clean (catches missing symbols the CUDA build hides):

```sh
cd glm-deepseek-v100/
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
of this tree lives on the oracle host):

```sh
ssh <oracle-host> 'cd glm-deepseek-v100/ && DS4_LOCK_FILE=/tmp/ds4-oracle.lock \
  ./ds4 --cpu --first-token-test --raw \
  -m ./models/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
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
  -v ./gguf:/models:ro \
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
| + `-e DS4_CUDA_PEER_EXPERT_CACHE_GB=26` (2nd GPU, needs `NVIDIA_VISIBLE_DEVICES=0,1`; 96-token warm run) | ≥ 0.45 t/s |
| + `-e DS4_CUDA_STREAM_READ_THREADS=0` (serial-read control) | ~0.23 t/s |

The peer tier makes the host L2 redundant (0.2% residual hits) — on a
two-GPU box prefer `PEER_EXPERT_CACHE` and leave `HOST_EXPERT_CACHE`
off. Expect `ds4: CUDA peer expert cache: +2253 experts / 25.99 GiB on
device 1` at startup and a combined L1 hit rate around 62% warm.

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
  -v ./models:/models:ro \
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

Staging-phase attribution (aggregate, printed at exit; near-zero cost):

```sh
-e DS4_CUDA_STREAM_STAGE_TIMING=1
# ds4: CUDA staging timing calls=... total=... classify+hitD2D=... io=... missD2D=... upload=...
```

Reference at the best config (26 GB + 28 GB L2, 32-token decode): io
~2.0 s/token and device-bound; missD2D and upload ~10 ms/token
combined. Caveat: classify+hitD2D issues synchronous D2D copies that
wait for all queued GPU work, so it absorbs pipeline drain and swings
several seconds between identical runs — treat it as
compute-plus-drain, not a scan cost.

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

## 8. Release gate — PASSED 2026-07-08

The 100-case official continuation fixture
(`gguf-tools/quality-testing/`, see QA_BEFORE_RELEASES.md) passed on the
CUDA path, so `DS4_GLM_CUDA_EXPERIMENTAL` and the startup WARNING were
removed (GLM on CUDA is now on by default; `DS4_GLM_CUDA_DISABLE`
restores the refusal). §2-§6 remain the per-change gates.

QA log — Q2 routed GGUF, CUDA + SSD streaming (26 GB expert budget +
28 GB host L2), reference band: first-token ~92/100, top-1 ~0.890,
pair-order ~0.800:

```
summary cases=100 tokens=2299 avg_nll=0.364632982 first_match=91 avg_lcp=7.500
api_summary ref_tokens=2294 target_tokens=2223 target_mae=0.305695700 target_mean_delta=-0.268561479 top_items=44460 top_mapped=44357 top_coverage=0.997683311 top1_match=1965/2223 top1_rate=0.883940621 topn_hit=32806/44357 topn_recall=0.739590144 top_logprob_count=44357 top_mae=1.849234104 top_mean_delta=0.966769438 pair_agree=331692/414321 pair_rate=0.800567676
```

Scorer build on Linux: `make gguf-tools/quality-testing/score_official
CUDA_ARCH=sm_70` (inside the CUDA 12.9 devel image; the rule links with
nvcc). Run from the repo root so the manifest's relative paths resolve.

## 9. Change-validation strategy: soak + fixture + binary A/B

The reusable recipe for any nontrivial streaming/runtime change,
composed of three machine-time-only steps (first used 2026-07-08 to
gate the peer-cache fix and the main fast-forward):

1. **Soak** — run the full 100-case fixture with the new feature
   enabled. It is ~10x longer than any benchmark run and catches what
   short runs miss (the peer-tier hang appeared only ~8 minutes into
   sustained load). Arm a watchdog on the output TSV's mtime: no new
   case row for >12 minutes while the container is up = stall; capture
   thread stacks before killing (`docker run --pid=container:<name>
   --cap-add=SYS_PTRACE ... gdb -p <pid> -batch -ex 'thread apply all
   bt'`).
2. **Fixture scoring** — the same run doubles as a quality datapoint
   for the affected model family; compare against the recorded band
   (§8 for GLM Q2) or the baseline below for Flash.
3. **Binary A/B** — for changes that should not alter numerics
   (IO plumbing, caching, scheduling), score the SAME model + manifest
   with the old binary (build it from the pre-change ref in a separate
   worktree) and the new one, then:
   `python3 gguf-tools/quality-testing/compare_scores.py old.tsv new.tsv`
   — the delta must be ~0. This turns "no documented reference band"
   into a non-issue and is the merge gate for forwarding shared
   improvements to another branch (e.g. glm5.2 -> main).

   **Fast variant (host-side / non-numeric changes only).** When the
   change cannot alter kernel math (host-side dedup, cached getenvs,
   allocation plumbing) a full 100-case score is overkill — the two
   binaries share identical kernels, so identical logits are exactly
   provable. Build old + new in worktrees, run each once with
   `--dump-logprobs out.json --logprobs-top-k 20 -p "<fixed prompt>"
   -n 48` (single GPU, any modest `--ssd-streaming-cache-experts` budget
   — cache size changes hit/miss rates, not logits) and `cmp -s` the
   dumps. The dump is fully deterministic (no timestamps/seed), so
   byte-identical output = numerics unchanged. ~15 min on one V100 vs
   ~2.3 h for the full A/B; first used 2026-07-08 to gate commit
   `885edc2`. Caveat: a single-GPU run does not exercise distributed-only
   code — cover that separately or by review. Not a substitute for the
   soak/fixture when a change *does* touch numerics.

DeepSeek Flash IQ2XXS baseline (recorded 2026-07-08, glm5.2 tree,
26 GiB expert budget, no L2):

```
summary cases=100 tokens=2289 avg_nll=0.404643735 first_match=64 avg_lcp=6.750
api_summary ref_tokens=2289 target_tokens=2289 target_mae=0.404643735 target_mean_delta=-0.404643735 top_items=11445 top_mapped=9484 top_coverage=0.828658803 top1_match=1979/2289 top1_rate=0.864569681 topn_hit=3014/9484 topn_recall=0.317798397 top_logprob_count=9484 top_mae=7567.322662597 top_mean_delta=7567.127337943 pair_agree=7133/7207 pair_rate=0.989732205
```

Caveat for the flash fixture set: its stored top-logprobs carry
placeholder values (5 alternatives/token, 83% mapping coverage), so
`api_top_mae`/`api_top_mean_delta` are meaningless there — judge Flash
by `avg_nll`, `first_match`, `top1_rate`, and the ordering-based
`pair_rate`, or better, by the binary A/B above. main predates the
Linux scorer rule; when building the old binary from refs before
98fb7a7, append the §8 link rule to the build copy's Makefile.
