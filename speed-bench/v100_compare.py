#!/usr/bin/env python3
"""Summarize one v100_bench run directory or compare two A/B directories."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROW_RE = re.compile(
    r"^(\d+),(\d+),([0-9.eE+-]+),(\d+),([0-9.eE+-]+),"
    r"([0-9.eE+-]+),(\d+),([0-9.eE+-]+),(\d+)$"
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
class Result:
    path: Path
    label: str
    profile: str
    sha: str
    rows: list[Row]


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


def load(path: Path) -> Result:
    if not path.is_dir():
        raise ValueError(f"not a run directory: {path}")
    meta = metadata(path)
    rows: list[Row] = []
    run_dirs = sorted(
        (p for p in path.glob("run-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-", 1)[1]),
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
        for line in log.read_text(errors="replace").splitlines():
            match = ROW_RE.match(line.strip())
            if not match:
                continue
            v = match.groups()
            rows.append(
                Row(
                    run=run_dir.name,
                    ctx_tokens=int(v[0]),
                    prefill_tokens=int(v[1]),
                    prefill_tps=float(v[2]),
                    gen_tokens=int(v[3]),
                    gen_tps=float(v[4]),
                    gen_first_ms=float(v[5]),
                    gen_steady_tokens=int(v[6]),
                    gen_steady_tps=float(v[7]),
                    kvcache_bytes=int(v[8]),
                )
            )
            found += 1
        if found == 0:
            raise ValueError(f"no benchmark CSV rows in {log}")
    return Result(
        path=path,
        label=meta.get("run_name", path.name),
        profile=meta.get("profile", "unknown"),
        sha=meta.get("git_sha", "unknown")[:12],
        rows=rows,
    )


def values(result: Result, ctx: int, field: str) -> list[float]:
    return [float(getattr(row, field)) for row in result.rows if row.ctx_tokens == ctx]


def med(result: Result, ctx: int, field: str) -> float:
    vals = values(result, ctx, field)
    if not vals:
        raise ValueError(f"{result.label}: no {field} values at ctx={ctx}")
    return statistics.median(vals)


def fmt_range(vals: list[float], decimals: int = 2) -> str:
    median = statistics.median(vals)
    if len(vals) == 1:
        return f"{median:.{decimals}f}"
    return f"{median:.{decimals}f} [{min(vals):.{decimals}f}..{max(vals):.{decimals}f}]"


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
        po, pn = med(old, ctx, "prefill_tps"), med(new, ctx, "prefill_tps")
        go, gn = med(old, ctx, "gen_tps"), med(new, ctx, "gen_tps")
        so, sn = med(old, ctx, "gen_steady_tps"), med(new, ctx, "gen_steady_tps")
        fo, fn = med(old, ctx, "gen_first_ms"), med(new, ctx, "gen_first_ms")
        print(
            f"{ctx:<5} {po:>11.2f} {pn:>11.2f} {delta(po, pn):>8} "
            f"{go:>8.2f} {gn:>8.2f} {delta(go, gn):>8} "
            f"{so:>11.2f} {sn:>11.2f} {delta(so, sn):>8} "
            f"{fo:>12.1f} {fn:>12.1f} {delta(fo, fn):>8}"
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
