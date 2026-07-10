#!/usr/bin/env bash
# Reproducible benchmark harness for the fork's V100 deployment profiles.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: speed-bench/v100_bench.sh PROFILE RUN_NAME

Profiles:
  single       one process on GPU0, local expert cache + SSD
  peer         one process on GPU0, GPU1 passive expert cache + SSD
  distributed coordinator GPU0 layers + worker GPU1 layers, CUDA IPC + SSD

Required environment:
  DS4_MODEL=/absolute/path/to/model.gguf

Common overrides:
  DS4_BINARY_DIR=$PWD          tree containing ds4 and ds4-bench
  DS4_RUN_ROOT=speed-bench/local-runs
  DS4_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu22.04
  DS4_PROMPT_REL=speed-bench/promessi_sposi.txt
  DS4_CTX_START=256 DS4_CTX_MAX=256 DS4_STEP_INCR=256
  DS4_GEN_TOKENS=96 DS4_CACHE=8GB DS4_RUNS=3 DS4_WARMUPS=1
  DS4_WARM_WEIGHTS=1 DS4_TIMEOUT_SEC=1800
  DS4_READ_THREADS=8 DS4_PREFILL_CHUNK=0
  DS4_DIST_PREFILL_CHUNK=0 DS4_DIST_PREFILL_WINDOW=0
  DS4_MIN_CPU_IDLE_PCT=80       refuse a loaded benchmark host
  DS4_MAX_IDLE_GPU_MIB=512      refuse GPUs already holding a workload
  DS4_MAX_SWAP_IO_MIB=256       maximum combined swap I/O per process
  DS4_MAX_SWAP_IO_MIB_PER_SEC=16 maximum during each one-second preflight
  DS4_ALLOW_BUSY=1              override the idle-host preflight
  DS4_HASH_MODEL=1              compute the large GGUF SHA-256 once
  DS4_MODEL_SHA256=<known-hash> record a known hash without rereading it

Peer profile:
  DS4_PEER_CACHE_GB=26 DS4_PEER_DEVICE=1

Distributed profile:
  DS4_COORD_LAYERS=0:21 DS4_WORKER_LAYERS=22:output
  DS4_COORD_CACHE=8GB DS4_WORKER_CACHE=8GB
  DS4_DIST_HOST=127.0.0.1 DS4_DIST_PORT=19801 DS4_WORKER_PORT=19802

Set DS4_DRY_RUN=1 to print commands and create metadata without running them.
The output directory is printed at startup. Compare runs with:
  python3 speed-bench/v100_compare.py <old-run-dir> <new-run-dir>
EOF
}

die() { printf 'v100_bench: %s\n' "$*" >&2; exit 2; }
quote_cmd() { printf '%q ' "$@"; printf '\n'; }

[[ $# -eq 2 ]] || { usage >&2; exit 2; }
profile=$1
run_name=$2
case "$profile" in single|peer|distributed) ;; *) die "unknown profile: $profile";; esac
[[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_NAME may contain only letters, digits, dot, underscore, and dash"

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
binary_dir=${DS4_BINARY_DIR:-$repo}
run_root=${DS4_RUN_ROOT:-$repo/speed-bench/local-runs}
image=${DS4_IMAGE:-nvidia/cuda:12.9.1-devel-ubuntu22.04}
model=${DS4_MODEL:-}
[[ -n "$model" ]] || die "DS4_MODEL is required"
[[ "$model" = /* ]] || die "DS4_MODEL must be an absolute path"
[[ -f "$model" ]] || die "model not found: $model"
[[ -x "$binary_dir/ds4" ]] || die "missing executable: $binary_dir/ds4"
[[ -x "$binary_dir/ds4-bench" ]] || die "missing executable: $binary_dir/ds4-bench"

prompt_rel=${DS4_PROMPT_REL:-speed-bench/promessi_sposi.txt}
[[ -f "$binary_dir/$prompt_rel" ]] || die "prompt not found under DS4_BINARY_DIR: $prompt_rel"
model_dir=$(dirname "$model")
model_name=$(basename "$model")
ctx_start=${DS4_CTX_START:-256}
ctx_max=${DS4_CTX_MAX:-256}
step_incr=${DS4_STEP_INCR:-256}
gen_tokens=${DS4_GEN_TOKENS:-96}
cache=${DS4_CACHE:-8GB}
runs=${DS4_RUNS:-3}
warmups=${DS4_WARMUPS:-1}
timeout_sec=${DS4_TIMEOUT_SEC:-1800}
read_threads=${DS4_READ_THREADS:-8}
prefill_chunk=${DS4_PREFILL_CHUNK:-0}
dist_prefill_chunk=${DS4_DIST_PREFILL_CHUNK:-0}
dist_prefill_window=${DS4_DIST_PREFILL_WINDOW:-0}
warm_weights=${DS4_WARM_WEIGHTS:-1}
dry_run=${DS4_DRY_RUN:-0}
hash_model=${DS4_HASH_MODEL:-0}
min_cpu_idle=${DS4_MIN_CPU_IDLE_PCT:-80}
max_idle_gpu_mib=${DS4_MAX_IDLE_GPU_MIB:-512}
max_swap_io_mib=${DS4_MAX_SWAP_IO_MIB:-256}
max_swap_io_mib_per_sec=${DS4_MAX_SWAP_IO_MIB_PER_SEC:-16}
page_size=$(getconf PAGESIZE)
allow_busy=${DS4_ALLOW_BUSY:-0}

for pair in "DS4_CTX_START:$ctx_start" "DS4_CTX_MAX:$ctx_max" \
            "DS4_STEP_INCR:$step_incr" "DS4_GEN_TOKENS:$gen_tokens" \
            "DS4_RUNS:$runs" "DS4_WARMUPS:$warmups" \
            "DS4_TIMEOUT_SEC:$timeout_sec" "DS4_READ_THREADS:$read_threads" \
            "DS4_PREFILL_CHUNK:$prefill_chunk" \
            "DS4_DIST_PREFILL_CHUNK:$dist_prefill_chunk" \
            "DS4_DIST_PREFILL_WINDOW:$dist_prefill_window" \
            "DS4_MIN_CPU_IDLE_PCT:$min_cpu_idle" \
            "DS4_MAX_IDLE_GPU_MIB:$max_idle_gpu_mib" \
            "DS4_MAX_SWAP_IO_MIB:$max_swap_io_mib" \
            "DS4_MAX_SWAP_IO_MIB_PER_SEC:$max_swap_io_mib_per_sec"; do
    key=${pair%%:*}; value=${pair#*:}
    [[ "$value" =~ ^[0-9]+$ ]] || die "$key must be a non-negative integer"
done
(( runs > 0 )) || die "DS4_RUNS must be positive"
(( ctx_start > 0 && ctx_max >= ctx_start && step_incr > 0 )) || die "invalid context range"
(( min_cpu_idle <= 100 )) || die "DS4_MIN_CPU_IDLE_PCT must be <= 100"

preflight_idle_host() {
    [[ "$dry_run" != 0 || "$allow_busy" != 0 ]] && return 0
    local _ u1 n1 s1 i1 w1 q1 sq1 st1 u2 n2 s2 i2 w2 q2 sq2 st2
    local pin1 pout1 pin2 pout2 total1 total2 idle_delta total_delta idle_pct
    local swap_pages swap_bytes max_swap_bytes
    read -r _ u1 n1 s1 i1 w1 q1 sq1 st1 _ < /proc/stat
    pin1=$(awk '$1=="pswpin"{print $2}' /proc/vmstat)
    pout1=$(awk '$1=="pswpout"{print $2}' /proc/vmstat)
    sleep 1
    read -r _ u2 n2 s2 i2 w2 q2 sq2 st2 _ < /proc/stat
    pin2=$(awk '$1=="pswpin"{print $2}' /proc/vmstat)
    pout2=$(awk '$1=="pswpout"{print $2}' /proc/vmstat)
    total1=$((u1+n1+s1+i1+w1+q1+sq1+st1))
    total2=$((u2+n2+s2+i2+w2+q2+sq2+st2))
    idle_delta=$((i2-i1))
    total_delta=$((total2-total1))
    idle_pct=$((total_delta > 0 ? 100 * idle_delta / total_delta : 0))
    swap_pages=$((pin2-pin1 + pout2-pout1))
    swap_bytes=$((swap_pages * page_size))
    max_swap_bytes=$((max_swap_io_mib_per_sec * 1048576))

    local gpu_busy=0 gpu_state
    gpu_state=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null) || die "nvidia-smi preflight failed"
    while IFS=',' read -r gpu mem util; do
        gpu=${gpu// /}; mem=${mem// /}; util=${util// /}
        if (( mem > max_idle_gpu_mib || util > 5 )); then gpu_busy=1; fi
    done <<<"$gpu_state"

    if (( idle_pct < min_cpu_idle || swap_bytes > max_swap_bytes || gpu_busy )); then
        printf 'v100_bench: host is not idle enough for a canonical run:\n' >&2
        printf '  cpu_idle=%s%% (required >= %s%%)\n' "$idle_pct" "$min_cpu_idle" >&2
        printf '  swap_delta_in=%s swap_delta_out=%s pages (limit %s MiB combined)\n' \
            "$((pin2-pin1))" "$((pout2-pout1))" "$max_swap_io_mib_per_sec" >&2
        while IFS= read -r line; do printf '  gpu %s\n' "$line" >&2; done <<<"$gpu_state"
        die "stop competing work or set DS4_ALLOW_BUSY=1 for a non-canonical smoke test"
    fi
}
preflight_idle_host

if [[ -n ${DS4_MODEL_SHA256:-} ]]; then
    model_sha256=$DS4_MODEL_SHA256
elif [[ "$hash_model" != 0 ]]; then
    printf 'v100_bench: hashing model (set DS4_MODEL_SHA256 to reuse a known hash)\n'
    model_sha256=$(sha256sum "$model" | awk '{print $1}')
else
    model_sha256=not-computed
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="$run_root/${stamp}-${run_name}-${profile}"
[[ ! -e "$out" ]] || die "output already exists: $out"
mkdir -p "$out"
printf 'v100_bench: output: %s\n' "$out"

common_bench=(
    -m "/models/$model_name" --cuda --ssd-streaming
    --ssd-streaming-cache-experts "$cache"
    --prompt-file "/src/$prompt_rel"
    --ctx-start "$ctx_start" --ctx-max "$ctx_max" --step-incr "$step_incr"
    --gen-tokens "$gen_tokens"
)
(( prefill_chunk == 0 )) || common_bench+=(--prefill-chunk "$prefill_chunk")
[[ "$warm_weights" == 0 ]] || common_bench+=(--warm-weights)

container_common=(
    --runtime=nvidia
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e "DS4_CUDA_STREAM_READ_THREADS=$read_threads"
    -v "$binary_dir:/src:ro"
    -v "$model_dir:/models:ro"
    -w /src
)

write_metadata() {
    local meta=$out/metadata.txt
    {
        printf 'created_utc=%s\n' "$(date -u +%FT%TZ)"
        printf 'profile=%s\nrun_name=%s\n' "$profile" "$run_name"
        printf 'binary_dir=%s\nmodel=%s\nprompt_rel=%s\n' "$binary_dir" "$model" "$prompt_rel"
        printf 'model_bytes=%s\n' "$(stat -c %s "$model")"
        printf 'model_mtime=%s\n' "$(stat -c %y "$model")"
        printf 'model_device=%s\n' "$(findmnt -no SOURCE --target "$model" 2>/dev/null || echo unknown)"
        printf 'model_device_maj_min=%s\n' "$(findmnt -no MAJ:MIN --target "$model" 2>/dev/null || echo unknown)"
        printf 'model_sha256=%s\n' "$model_sha256"
        printf 'image=%s\nctx_start=%s\nctx_max=%s\nstep_incr=%s\n' "$image" "$ctx_start" "$ctx_max" "$step_incr"
        printf 'gen_tokens=%s\ncache=%s\nruns=%s\nwarmups=%s\n' "$gen_tokens" "$cache" "$runs" "$warmups"
        printf 'read_threads=%s\nprefill_chunk=%s\ndist_prefill_chunk=%s\n' \
            "$read_threads" "$prefill_chunk" "$dist_prefill_chunk"
        printf 'dist_prefill_window=%s\nwarm_weights=%s\ntimeout_sec=%s\n' \
            "$dist_prefill_window" "$warm_weights" "$timeout_sec"
        printf 'min_cpu_idle_pct=%s\nmax_idle_gpu_mib=%s\nmax_swap_io_mib=%s\n' \
            "$min_cpu_idle" "$max_idle_gpu_mib" "$max_swap_io_mib"
        printf 'max_swap_io_mib_per_sec=%s\nallow_busy=%s\n' \
            "$max_swap_io_mib_per_sec" "$allow_busy"
        printf 'git_sha=%s\n' "$(git -C "$binary_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
        printf 'git_status=%q\n' "$(git -C "$binary_dir" status --short 2>/dev/null || true)"
        printf 'ds4_sha256=%s\n' "$(sha256sum "$binary_dir/ds4" | awk '{print $1}')"
        printf 'ds4_bench_sha256=%s\n' "$(sha256sum "$binary_dir/ds4-bench" | awk '{print $1}')"
        printf 'kernel=%s\n' "$(uname -srmo)"
        printf '\n[nvidia-smi]\n'; nvidia-smi 2>&1 || true
        printf '\n[topology]\n'; nvidia-smi topo -m 2>&1 || true
        printf '\n[cpu]\n'; lscpu 2>&1 || true
        printf '\n[memory]\n'; free -h 2>&1 || true
        printf '\n[filesystem]\n'; df -hT "$model" 2>&1 || true
        printf '\n[environment]\n'
        env | LC_ALL=C sort | grep -E '^(DS4_|NVIDIA_)' | \
            sed -E 's/^([^=]*(KEY|TOKEN|SECRET|PASSWORD)[^=]*)=.*/\1=<redacted>/' || true
    } >"$meta"
}
write_metadata

monitor_pid=
start_monitor() {
    local dir=$1
    cp /proc/diskstats "$dir/diskstats.before"
    cp /proc/vmstat "$dir/vmstat.before"
    (
        printf 'timestamp,index,memory_used_MiB,util_gpu_pct,util_mem_pct,pstate,sm_clock_MHz,mem_clock_MHz,power_W,temp_C\n'
        while :; do
            nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu \
                --format=csv,noheader,nounits 2>/dev/null || true
            sleep 1
        done
    ) >"$dir/gpu.csv" &
    monitor_pid=$!
}
stop_monitor() {
    local dir=$1
    if [[ -n ${monitor_pid:-} ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=
    fi
    cp /proc/diskstats "$dir/diskstats.after"
    cp /proc/vmstat "$dir/vmstat.after"
    free -h >"$dir/memory.after"

    local pin_before pin_after pout_before pout_after
    pin_before=$(awk '$1=="pswpin"{print $2}' "$dir/vmstat.before")
    pin_after=$(awk '$1=="pswpin"{print $2}' "$dir/vmstat.after")
    pout_before=$(awk '$1=="pswpout"{print $2}' "$dir/vmstat.before")
    pout_after=$(awk '$1=="pswpout"{print $2}' "$dir/vmstat.after")
    {
        printf 'pswpin_delta=%s\n' "$((pin_after-pin_before))"
        printf 'pswpout_delta=%s\n' "$((pout_after-pout_before))"
    } >"$dir/swap_delta.txt"
    local swap_pages=$((pin_after-pin_before + pout_after-pout_before))
    local swap_bytes=$((swap_pages * page_size))
    if [[ "$allow_busy" == 0 ]] &&
       (( swap_bytes > max_swap_io_mib * 1048576 )); then
        printf 'v100_bench: swap I/O exceeded %s MiB during benchmark; run is non-canonical (%s)\n' \
            "$max_swap_io_mib" "$(tr '\n' ' ' <"$dir/swap_delta.txt")" >&2
        return 1
    fi
}
active_containers=()
cleanup() {
    [[ -z ${monitor_pid:-} ]] || kill "$monitor_pid" 2>/dev/null || true
    if (( ${#active_containers[@]} )); then
        docker rm -f "${active_containers[@]}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

run_single_process() {
    local dir=$1 index=$2
    local name="ds4-v100-$profile-$$-$index"
    local -a cmd=(docker run --rm --name "$name" "${container_common[@]}")
    if [[ "$profile" == single ]]; then
        cmd+=( -e NVIDIA_VISIBLE_DEVICES=0 -e DS4_LOCK_FILE=/tmp/ds4-v100-single.lock )
    else
        cmd+=( -e NVIDIA_VISIBLE_DEVICES=0,1
               -e DS4_LOCK_FILE=/tmp/ds4-v100-peer.lock
               -e DS4_CUDA_DEVICE=0
               -e "DS4_CUDA_PEER_DEVICE=${DS4_PEER_DEVICE:-1}"
               -e "DS4_CUDA_PEER_EXPERT_CACHE_GB=${DS4_PEER_CACHE_GB:-26}" )
    fi
    cmd+=( --entrypoint /src/ds4-bench "$image" "${common_bench[@]}" )
    quote_cmd "${cmd[@]}" >"$dir/command.sh"
    if [[ "$dry_run" != 0 ]]; then cat "$dir/command.sh"; return 0; fi
    active_containers=("$name")
    start_monitor "$dir"
    set +e
    timeout --signal=TERM --kill-after=30 "$timeout_sec" "${cmd[@]}" >"$dir/bench.log" 2>&1
    local rc=$?
    set -e
    docker rm -f "$name" >/dev/null 2>&1 || true
    active_containers=()
    local monitor_rc=0
    stop_monitor "$dir" || monitor_rc=$?
    if (( rc == 0 && monitor_rc != 0 )); then rc=$monitor_rc; fi
    printf '%s\n' "$rc" >"$dir/exit_code"
    return "$rc"
}

run_distributed() {
    local dir=$1 index=$2
    local host=${DS4_DIST_HOST:-127.0.0.1}
    local base_port=${DS4_DIST_PORT:-19801}
    [[ "$base_port" =~ ^[0-9]+$ ]] || die "DS4_DIST_PORT must be an integer"
    local coord_port=$((base_port + (index - 1) * 2))
    local worker_port=${DS4_WORKER_PORT:-$((coord_port + 1))}
    [[ "$worker_port" =~ ^[0-9]+$ ]] || die "DS4_WORKER_PORT must be an integer"
    (( coord_port > 0 && coord_port <= 65535 &&
       worker_port > 0 && worker_port <= 65535 )) ||
        die "distributed ports must be in 1..65535"
    local coord_layers=${DS4_COORD_LAYERS:-0:21}
    local worker_layers=${DS4_WORKER_LAYERS:-22:output}
    local coord_cache=${DS4_COORD_CACHE:-$cache}
    local worker_cache=${DS4_WORKER_CACHE:-$cache}
    local suffix="$$-$index"
    local coord="ds4-v100-coord-$suffix" worker="ds4-v100-worker-$suffix"
    local -a coord_cmd=(docker run -d --name "$coord" "${container_common[@]}"
        --ipc=host --pid=host --network=host
        -e NVIDIA_VISIBLE_DEVICES=0,1 -e DS4_CUDA_DEVICE=0
        -e DS4_LOCK_FILE=/tmp/ds4-v100-coordinator.lock
        --entrypoint /src/ds4-bench "$image"
        -m "/models/$model_name" --cuda --ssd-streaming
        --ssd-streaming-cache-experts "$coord_cache"
        --prompt-file "/src/$prompt_rel"
        --ctx-start "$ctx_start" --ctx-max "$ctx_max" --step-incr "$step_incr"
        --gen-tokens "$gen_tokens"
        --role coordinator --layers "$coord_layers" --listen "$host" "$coord_port")
    (( prefill_chunk == 0 )) || coord_cmd+=(--prefill-chunk "$prefill_chunk")
    (( dist_prefill_chunk == 0 )) || coord_cmd+=(--dist-prefill-chunk "$dist_prefill_chunk")
    (( dist_prefill_window == 0 )) || coord_cmd+=(--dist-prefill-window "$dist_prefill_window")
    [[ "$warm_weights" == 0 ]] || coord_cmd+=(--warm-weights)
    local -a worker_cmd=(docker run -d --name "$worker" "${container_common[@]}"
        --ipc=host --pid=host --network=host
        -e NVIDIA_VISIBLE_DEVICES=0,1 -e DS4_CUDA_DEVICE=1
        -e DS4_LOCK_FILE=/tmp/ds4-v100-worker.lock
        --entrypoint /src/ds4 "$image"
        -m "/models/$model_name" --cuda --ssd-streaming
        --ssd-streaming-cache-experts "$worker_cache"
        --ctx "$((ctx_max + gen_tokens + 1))"
        --role worker --layers "$worker_layers"
        --coordinator "$host" "$coord_port" --listen "$host" "$worker_port")
    {
        quote_cmd "${coord_cmd[@]}"
        quote_cmd "${worker_cmd[@]}"
    } >"$dir/command.sh"
    if [[ "$dry_run" != 0 ]]; then cat "$dir/command.sh"; return 0; fi

    docker rm -f "$coord" "$worker" >/dev/null 2>&1 || true
    active_containers=("$worker" "$coord")
    start_monitor "$dir"
    local rc=0
    set +e
    "${coord_cmd[@]}" >"$dir/coordinator.id"
    rc=$?
    set -e
    if (( rc == 0 )); then
        sleep 2
        set +e
        "${worker_cmd[@]}" >"$dir/worker.id"
        rc=$?
        set -e
    fi
    if (( rc == 0 )); then
        set +e
        timeout --signal=TERM --kill-after=30 "$timeout_sec" docker wait "$coord" >"$dir/coordinator.wait" 2>&1
        rc=$?
        set -e
        if (( rc == 0 )); then rc=$(cat "$dir/coordinator.wait" 2>/dev/null || echo 1); fi
    fi
    docker logs "$coord" >"$dir/bench.log" 2>&1 || true
    docker logs "$worker" >"$dir/worker.log" 2>&1 || true
    docker rm -f "$worker" "$coord" >/dev/null 2>&1 || true
    active_containers=()
    local monitor_rc=0
    stop_monitor "$dir" || monitor_rc=$?
    if (( rc == 0 && monitor_rc != 0 )); then rc=$monitor_rc; fi
    printf '%s\n' "$rc" >"$dir/exit_code"
    return "$rc"
}

run_one() {
    local kind=$1 index=$2
    local dir="$out/$kind-$index"
    # A sweep can run for hours; another workload may start after the initial
    # preflight. Recheck before every process rather than trusting startup state.
    preflight_idle_host
    mkdir -p "$dir"
    printf 'v100_bench: %s %s/%s\n' "$kind" "$index" "$([[ "$kind" == warmup ]] && echo "$warmups" || echo "$runs")"
    if [[ "$profile" == distributed ]]; then
        run_distributed "$dir" "$index"
    else
        run_single_process "$dir" "$index"
    fi
}

for ((i=1; i<=warmups; i++)); do run_one warmup "$i" || die "warmup $i failed; see $out/warmup-$i"; done
for ((i=1; i<=runs; i++)); do run_one run "$i" || die "run $i failed; see $out/run-$i"; done

if [[ "$dry_run" == 0 ]]; then
    python3 "$repo/speed-bench/v100_compare.py" "$out" | tee "$out/summary.txt"
fi
printf 'v100_bench: complete: %s\n' "$out"
