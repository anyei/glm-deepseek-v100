## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `m3_max.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/m3_max.csv --title "M3 Max t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/m3_max_ts.svg`.

## Reproducible V100 architecture A/B

`v100_bench.sh` runs the fork's three current deployment profiles with the same
model, prompt, context sweep, and token count. It captures binary/model
identity, host/GPU metadata, raw logs, GPU telemetry, and diskstats under the
ignored `speed-bench/local-runs/` directory.

```sh
# Canonical single-GPU baseline: one warm-up, then three measured processes.
DS4_MODEL=/absolute/path/to/DeepSeek-V4-Flash.gguf \
DS4_CACHE=8GB DS4_CTX_START=256 DS4_CTX_MAX=256 \
DS4_GEN_TOKENS=96 DS4_WARMUPS=1 DS4_RUNS=3 \
  speed-bench/v100_bench.sh single flash-main

# GPU0 computes; GPU1 supplies a 26 GiB passive expert-cache tier.
DS4_MODEL=/absolute/path/to/DeepSeek-V4-Flash.gguf \
DS4_CACHE=24GB DS4_PEER_CACHE_GB=26 \
  speed-bench/v100_bench.sh peer flash-peer

# Existing whole-layer architecture, with CUDA IPC/NVLink activations.
DS4_MODEL=/absolute/path/to/DeepSeek-V4-Flash.gguf \
DS4_COORD_LAYERS=0:21 DS4_WORKER_LAYERS=22:output \
DS4_COORD_CACHE=8GB DS4_WORKER_CACHE=8GB \
  speed-bench/v100_bench.sh distributed flash-dist
```

Use `DS4_BINARY_DIR=/path/to/worktree` to benchmark a binary built in another
worktree. Set `DS4_HASH_MODEL=1` for a canonical run, or pass a previously known
hash as `DS4_MODEL_SHA256`; hashing is opt-in because the tested GGUFs are
81–245 GiB. Sensitive environment values containing `KEY`, `TOKEN`, `SECRET`,
or `PASSWORD` are redacted from metadata. Each measured process emits normal
`ds4-bench` CSV rows. The harness summarizes their median and range
automatically; compare two completed run directories with:

```sh
python3 speed-bench/v100_compare.py \
  speed-bench/local-runs/<old-run> \
  speed-bench/local-runs/<new-run>
```

Run `speed-bench/v100_bench.sh` without arguments for every override. Before a
real run, the harness requires at least 80% CPU idle, no swap-in/out during its
one-second sample, and idle GPUs using at most 512 MiB each. Stop competing
builds and model servers rather than contaminating a canonical result;
`DS4_ALLOW_BUSY=1` is available only for explicitly non-canonical smoke tests.
Set `DS4_DRY_RUN=1` to validate and print container commands without occupying
the GPUs. Performance claims should use at least 96 generated tokens and three
measured runs; shorter runs are smoke tests only.

## Disk read throughput (`io_probe.c`)

Streaming inference is disk-bound, so the NVMe read rate is a first-class
number. **Measure it with `io_probe.c`, not `dd`.** A single `dd
iflag=direct` is queue-depth 1 — latency-bound — and understates a modern
NVMe badly (we once recorded 1.25 GB/s of a drive's real ~2.5 GB/s that way).
`io_probe` keeps N requests in flight over one sequential stream via io_uring,
which is what actually saturates the device.

```sh
gcc -O2 speed-bench/io_probe.c -luring -o /tmp/io_probe
# io_probe <file> <queue_depth> <block_bytes> <total_bytes> <start_offset_bytes>
/tmp/io_probe model.gguf 4 16777216 8589934592 $((120*1024*1024*1024))
```

Read a file **larger than RAM** at an **offset you have not read this run**
(O_DIRECT bypasses cache, but fresh offsets avoid any device-side reuse), and
sweep both knobs — they teach different things:

- **Block size matters more than queue depth.** On the drives here, 1 MiB
  blocks give ~1.7 GB/s, 4 MiB ~2.4, 16 MiB ~2.5. Use ≥ 4 MiB.
- **Queue depth saturates early** (~QD4). Past that it is flat or, on
  DRAM-less QLC drives, *degrades* — scattered concurrent reads thrash a
  cache-less FTL. If more QD makes it slower, that is the drive telling you it
  is DRAM-less.

Measured ceilings (cold, 16 MiB blocks, QD4): the V100 box and the CPU-oracle
box (Kingston OM8TAP4, DRAM-less QLC) both land around **~2.45–2.5 GB/s** — a
drive's advertised sequential spec (e.g. "5.5 GB/s") is a warm/SLC-cache burst,
not a sustained cold read of a big model file, so don't plan against it.

To mimic ds4's *actual* access pattern (experts scattered through the file)
rather than best-case sequential, read random offsets instead:

```sh
F=model.gguf; t0=$(date +%s.%N)
for i in $(seq 0 63); do
  dd if=$F of=/dev/null bs=4M count=1 skip=$((RANDOM * 977 % 60000)) iflag=direct 2>/dev/null
done
t1=$(date +%s.%N); echo "QD1 scattered: $(echo "256 / ($t1 - $t0)" | bc) MiB/s"
```
