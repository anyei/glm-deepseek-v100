# Development workflow (glm-deepseek-v100 fork)

How to make a change on this fork and land it on `main` safely. The engine is
numerically-validated shipping code, so the discipline below exists for one
reason: a refactor or perf change must never *silently* alter model outputs.

Related docs:
- [AGENT.md](AGENT.md) — code-quality rules, file layout, scope-of-impact checklist.
- [VALIDATION.md](VALIDATION.md) — the validation playbook; §1 build gates, §9 the A/B recipe this doc drives.
- [VOLTA.md](VOLTA.md) — V100/sm_70 internals. [RUNNING.md](RUNNING.md) — Docker / compose.
- [ROADMAP.md](ROADMAP.md) — what to work on next.

## The loop

1. **Branch.** Never commit unvalidated engine code straight to `main`. Work on a
   branch (`git switch -c cleanup/...`); `main` receives only docs and
   gate-passed code.
2. **Build** — see below.
3. **Classify the change** and run the matching gate — see the table.
4. **Merge** — cherry-pick / fast-forward onto `main`, push.

## Build

CUDA 13.x dropped Volta, so the CUDA build runs inside the CUDA 12.9 container.
The host toolchain (CUDA 13.x) *cannot* build `sm_70` — always use the container.

```sh
# sm_70 CUDA build (V100), inside nvidia/cuda:12.9.1-devel-ubuntu22.04:
docker run --rm -v "$PWD":/src -w /src nvidia/cuda:12.9.1-devel-ubuntu22.04 \
  bash -lc 'apt-get update -qq && apt-get install -y -qq make gcc g++ >/dev/null; \
            make cuda CUDA_ARCH=sm_70 -j"$(nproc)"'

# Quick syntax/compile check of a CUDA change (one translation unit, no link):
#   nvcc -O3 --use_fast_math -arch=sm_70 -Wno-deprecated-gpu-targets -c -o /tmp/x.o ds4_cuda.cu

# CPU reference/debug build (host gcc, no CUDA):
make cpu -j"$(nproc)"

# Deployed image from the working tree (pins CUDA 12.9, targets sm_70):
docker build -t ds4:sm70-ipc .
```

`make` alone on Linux prints the available targets.

## Which gate for which change

| Change | Gate | Why |
|---|---|---|
| Docs / comments / non-code | compile-check the affected build | nothing to run |
| Host-side, cannot alter kernel math — dedup, cached getenv, alloc plumbing, IO/scheduling | **fast-equivalence A/B** (below) | identical kernels ⇒ identical logits, provable in ~15 min |
| Touches kernel math (reductions, rope, dots, quant) or the staging/compaction logic | **full fixture** (VALIDATION.md §9) | FP eval-order changes break byte-identity; needs the scored 100-case run |
| Release | [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md) | full multi-backend checklist |

When unsure, escalate a tier. A single-GPU run never exercises distributed-only
code — cover that by review or an explicit distributed run.

## Fast-equivalence A/B (host-side / non-numeric changes)

Two binaries share identical kernels, so identical `--dump-logprobs` output
proves no behavioral change. The dump has no timestamps/seed, so byte-identical
= numerically inert. ~15 min on one idle V100.

```sh
# 1. worktrees at the pre-change ref and your change
git worktree add /tmp/ab-old <ref-before-change>   # e.g. HEAD~1 or the branch base
git worktree add /tmp/ab-new HEAD

# 2. build ds4 in each (sm_70 container, as above) -> /tmp/ab-{old,new}/ds4

# 3. run each: same prompt, greedy, dump logits. Set any env var the change
#    touches (e.g. DS4_CUDA_HOST_EXPERT_CACHE_GB=16) so the changed path runs.
for tag in old new; do
  docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v /tmp/ab-$tag:/src -v <model-dir>:/models:ro -v /tmp:/out -w /src \
    nvidia/cuda:12.9.1-devel-ubuntu22.04 \
    ./ds4 -m /models/<model.gguf> --cuda --nothink --ssd-streaming \
      --ssd-streaming-cache-experts 8GB --ctx 2048 \
      --dump-logprobs /out/$tag.json --logprobs-top-k 20 \
      -p "In one sentence, explain what makes the ocean blue." -n 48
done

# 4. verdict
cmp -s /tmp/old.json /tmp/new.json && echo "IDENTICAL — no numeric change" || echo "DIFFER — investigate"

# 5. clean up
git worktree remove --force /tmp/ab-old; git worktree remove --force /tmp/ab-new
```

Not a substitute for the full fixture when the change *can* alter numerics —
there, byte-identity would fail even when the change is correct.

## Full fixture gate

For numeric changes run [VALIDATION.md §9](VALIDATION.md): soak (the full
100-case fixture with a stall watchdog), fixture scoring against the §8 band,
and the old-vs-new binary A/B via `score_official` + `compare_scores.py` (delta
~0 for changes that should not alter numerics; a scored comparison for changes
that legitimately do).

## Models

| Purpose | Path |
|---|---|
| GLM 5.2 Q2 — benchmarks, release gate | `./gguf/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` |
| DeepSeek V4 Flash IQ2 — regression | `/mnt/models/ollama37-k80/.ollama/custom-models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf` |
| Disk-throughput probe | `speed-bench/io_probe.c` (io_uring; see [speed-bench/README.md](speed-bench/README.md)) |

## Merge & commit

- Validate on the branch, *then* cherry-pick / fast-forward onto `main` and push.
  Hold unvalidated code on the branch until its gate passes.
- End commit messages with the `Co-Authored-By:` trailer for AI-assisted work.
- Done means: `git status` clean and `HEAD == origin/main`. Build artifacts
  (`ds4`, `ds4-*`, `*.o`) are git-ignored; `make clean` removes them.
