#!/usr/bin/env python3
"""Summarize one v100_bench run directory or compare two A/B directories."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROW_RE = re.compile(
    r"^(\d+),(\d+),([0-9.eE+-]+),(\d+),([0-9.eE+-]+),"
    r"([0-9.eE+-]+),(\d+),([0-9.eE+-]+),(\d+)$"
)
CACHE_RE = re.compile(
    r"dynamic cache \((\d+) experts, ([0-9.]+) MiB each\)"
)
CACHE_CAP_RE = re.compile(
    r"streaming expert cache capped from \d+ to (\d+) experts"
)
CACHE_ALLOC_RE = re.compile(
    r"expert cache allocated: local=(\d+) peer=(\d+) total=(\d+) slots"
)
PEER_CACHE_RE = re.compile(
    r"peer expert cache: \+(\d+) experts / ([0-9.]+) GiB on device (\d+)"
)


@dataclass(frozen=True)
class Row:
    run: str
    ctx_tokens: int
    prefill_tokens: int
    prefill_tps: float
    gen_tokens: int
    gen_tps: float
    gen_first_ms: float
    gen_steady_tokens: int
    gen_steady_tps: float
    kvcache_bytes: int


@dataclass
class RunResources:
    local_slots: int | None
    peer_slots: int | None
    peer_gib: float | None
    gpu_peak_mib: dict[int, int]
    disk_read_bytes: int | None


@dataclass
class Result:
    path: Path
    label: str
    profile: str
    sha: str
    rows: list[Row]
    resources: list[RunResources]


def parse_int(value: str, context: str) -> int:
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"invalid integer for {context}: {value!r}") from error


def parse_float(value: str, context: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"invalid number for {context}: {value!r}") from error


def run_index(path: Path) -> int:
    return parse_int(path.name.split("-", 1)[1], f"run directory {path.name}")


def metadata(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    meta = path / "metadata.txt"
    if not meta.exists():
        return out
    for line in meta.read_text(errors="replace").splitlines():
        if "=" not in line or line.startswith("["):
            continue
        key, value = line.split("=", 1)
        out.setdefault(key, value)
    return out


def disk_sectors(path: Path, maj_min: str) -> int | None:
    if not path.exists() or ":" not in maj_min:
        return None
    try:
        major_text, minor_text = maj_min.split(":", 1)
        major = parse_int(major_text, "model device major")
        minor = parse_int(minor_text, "model device minor")
    except ValueError:
        return None
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            line_major = parse_int(fields[0], "diskstats major")
            line_minor = parse_int(fields[1], "diskstats minor")
            sectors = parse_int(fields[5], "diskstats sectors")
        except ValueError:
            continue
        if line_major == major and line_minor == minor:
            return sectors
    return None


def run_resources(run_dir: Path, log_text: str, meta: dict[str, str]) -> RunResources:
    configured = CACHE_RE.search(log_text)
    capped = CACHE_CAP_RE.findall(log_text)
    allocated = CACHE_ALLOC_RE.findall(log_text)
    local_slots = parse_int(allocated[-1][0], "allocated local slots") if allocated else (
        parse_int(capped[-1], "capped local slots") if capped else (
            parse_int(configured.group(1), "configured local slots") if configured else None
        )
    )
    peer = PEER_CACHE_RE.search(log_text)
    allocated_peer = (
        parse_int(allocated[-1][1], "allocated peer slots") if allocated else None
    )

    peaks: dict[int, int] = {}
    gpu_csv = run_dir / "gpu.csv"
    if gpu_csv.exists():
        import csv
        with gpu_csv.open(newline="", errors="replace") as file:
            for row in csv.DictReader(file):
                try:
                    gpu = int(row["index"].strip())
                    used = int(row["memory_used_MiB"].strip())
                except (KeyError, TypeError, ValueError):
                    continue
                peaks[gpu] = max(peaks.get(gpu, 0), used)

    maj_min = meta.get("model_device_maj_min", "")
    before = disk_sectors(run_dir / "diskstats.before", maj_min)
    after = disk_sectors(run_dir / "diskstats.after", maj_min)
    disk_bytes = None
    if before is not None and after is not None and after >= before:
        # Linux /proc/diskstats sectors are always 512 bytes.
        disk_bytes = (after - before) * 512

    return RunResources(
        local_slots=local_slots,
        peer_slots=allocated_peer if allocated_peer is not None else (
            parse_int(peer.group(1), "peer slots") if peer else None
        ),
        peer_gib=parse_float(peer.group(2), "peer cache GiB") if peer else None,
        gpu_peak_mib=peaks,
        disk_read_bytes=disk_bytes,
    )


def load(path: Path) -> Result:
    if not path.is_dir():
        raise ValueError(f"not a run directory: {path}")
    meta = metadata(path)
    rows: list[Row] = []
    resources: list[RunResources] = []
    run_dirs = sorted(
        (p for p in path.glob("run-*") if p.is_dir()),
        key=run_index,
    )
    if not run_dirs:
        raise ValueError(f"no run-* directories under {path}")
    for run_dir in run_dirs:
        exit_file = run_dir / "exit_code"
        if exit_file.exists() and exit_file.read_text().strip() != "0":
            raise ValueError(f"nonzero benchmark exit in {run_dir}")
        log = run_dir / "bench.log"
        if not log.exists():
            raise ValueError(f"missing {log}")
        found = 0
        log_text = log.read_text(errors="replace")
        for line in log_text.splitlines():
            match = ROW_RE.match(line.strip())
            if not match:
                continue
            v = match.groups()
            rows.append(
                Row(
                    run=run_dir.name,
                    ctx_tokens=parse_int(v[0], f"{run_dir.name} context tokens"),
                    prefill_tokens=parse_int(v[1], f"{run_dir.name} prefill tokens"),
                    prefill_tps=parse_float(v[2], f"{run_dir.name} prefill throughput"),
                    gen_tokens=parse_int(v[3], f"{run_dir.name} generated tokens"),
                    gen_tps=parse_float(v[4], f"{run_dir.name} generation throughput"),
                    gen_first_ms=parse_float(v[5], f"{run_dir.name} first-token latency"),
                    gen_steady_tokens=parse_int(v[6], f"{run_dir.name} steady tokens"),
                    gen_steady_tps=parse_float(v[7], f"{run_dir.name} steady throughput"),
                    kvcache_bytes=parse_int(v[8], f"{run_dir.name} KV-cache bytes"),
                )
            )
            found += 1
        if found == 0:
            raise ValueError(f"no benchmark CSV rows in {log}")
        resources.append(run_resources(run_dir, log_text, meta))
    return Result(
        path=path,
        label=meta.get("run_name", path.name),
        profile=meta.get("profile", "unknown"),
        sha=meta.get("git_sha", "unknown")[:12],
        rows=rows,
        resources=resources,
    )


def values(result: Result, ctx: int, field: str) -> list[float]:
    output: list[float] = []
    for row in result.rows:
        if row.ctx_tokens != ctx:
            continue
        value = getattr(row, field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{result.label}: {field} is not numeric")
        output.append(value * 1.0)
    return output


def med(result: Result, ctx: int, field: str) -> float:
    vals = values(result, ctx, field)
    if not vals:
        raise ValueError(f"{result.label}: no {field} values at ctx={ctx}")
    return statistics.median(vals)


def fmt_range(vals: Sequence[int | float], decimals: int = 2) -> str:
    median = statistics.median(vals)
    if len(vals) == 1:
        return f"{median:.{decimals}f}"
    return f"{median:.{decimals}f} [{min(vals):.{decimals}f}..{max(vals):.{decimals}f}]"


def fmt_optional_range(values: Sequence[int | float], suffix: str = "") -> str:
    if not values:
        return "n/a"
    return f"{fmt_range(values, 0)}{suffix}"


def describe(result: Result) -> None:
    print(f"run={result.label} profile={result.profile} sha={result.sha} path={result.path}")
    print("ctx  samples  prefill_tps median[range]  gen_tps median[range]  "
          "steady_tps median[range]  first_ms median[range]")
    for ctx in sorted({row.ctx_tokens for row in result.rows}):
        p = values(result, ctx, "prefill_tps")
        g = values(result, ctx, "gen_tps")
        s = values(result, ctx, "gen_steady_tps")
        f = values(result, ctx, "gen_first_ms")
        print(
            f"{ctx:<5} {len(p):<7} {fmt_range(p):<27} {fmt_range(g):<24} "
            f"{fmt_range(s):<27} {fmt_range(f, 1)}"
        )
    local = [r.local_slots for r in result.resources if r.local_slots is not None]
    peer = [r.peer_slots for r in result.resources if r.peer_slots is not None]
    disk = [r.disk_read_bytes for r in result.resources if r.disk_read_bytes is not None]
    gpu_ids = sorted({gpu for r in result.resources for gpu in r.gpu_peak_mib})
    gpu_parts = []
    for gpu in gpu_ids:
        peak_values = [
            r.gpu_peak_mib[gpu] for r in result.resources if gpu in r.gpu_peak_mib
        ]
        gpu_parts.append(
            f"gpu{gpu}_peak={fmt_optional_range(peak_values, ' MiB')}"
        )
    gpu_text = " ".join(gpu_parts)
    disk_text = fmt_range([value / (1024 ** 3) for value in disk], 2) + " GiB" if disk else "n/a"
    print(
        f"resources local_slots={fmt_optional_range(local)} "
        f"peer_slots={fmt_optional_range(peer)} disk_read={disk_text} {gpu_text}".rstrip()
    )


def delta(old: float, new: float) -> str:
    if old == 0:
        return "n/a"
    return f"{(new / old - 1.0) * 100.0:+.2f}%"


def compare(old: Result, new: Result) -> None:
    common = sorted(
        {row.ctx_tokens for row in old.rows} & {row.ctx_tokens for row in new.rows}
    )
    if not common:
        raise ValueError("the two runs have no common context rows")
    print(f"comparison old={old.label}@{old.sha} new={new.label}@{new.sha}")
    print("ctx  prefill_old  prefill_new  delta    gen_old  gen_new  delta    "
          "steady_old  steady_new  delta    first_ms_old  first_ms_new  delta")
    for ctx in common:
        po, prefill_new = med(old, ctx, "prefill_tps"), med(new, ctx, "prefill_tps")
        go, gn = med(old, ctx, "gen_tps"), med(new, ctx, "gen_tps")
        so, sn = med(old, ctx, "gen_steady_tps"), med(new, ctx, "gen_steady_tps")
        first_old, fn = med(old, ctx, "gen_first_ms"), med(new, ctx, "gen_first_ms")
        print(
            f"{ctx:<5} {po:>11.2f} {prefill_new:>11.2f} {delta(po, prefill_new):>8} "
            f"{go:>8.2f} {gn:>8.2f} {delta(go, gn):>8} "
            f"{so:>11.2f} {sn:>11.2f} {delta(so, sn):>8} "
            f"{first_old:>12.1f} {fn:>12.1f} {delta(first_old, fn):>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="+", type=Path, help="one run to summarize or old new to compare")
    args = parser.parse_args()
    if len(args.run_dir) not in (1, 2):
        parser.error("provide one run directory or two directories (old new)")
    try:
        results = [load(path.resolve()) for path in args.run_dir]
        for i, result in enumerate(results):
            if i:
                print()
            describe(result)
        if len(results) == 2:
            print()
            compare(results[0], results[1])
    except (OSError, ValueError) as exc:
        print(f"v100_compare: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
