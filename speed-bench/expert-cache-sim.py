#!/usr/bin/env python3
"""Replay DS4 CUDA expert traces and simulate cache admission policies.

`exact-lru` models the current combined local+peer directory. Alternatives are
not trusted unless it reproduces observed unique-expert hits, misses, and bytes.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

Key = tuple[int, int]  # layer, expert
POLICIES = {"exact-lru", "segmented", "tinylfu", "per-layer-quota",
            "decode-protected", "owner-balanced", "topk-replication"}


@dataclass
class Event:
    call: int
    token: int
    layer: int
    experts: list[int]
    capacity: int
    local_capacity: int
    expert_bytes: int
    observed: dict[int, tuple[int, int]]


def load_trace(path: Path) -> tuple[list[Event], dict[str, str]]:
    header: dict[str, str] = {}; lines = []
    with path.open(newline="", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                for field in line[1:].split():
                    if "=" in field:
                        key, value = field.split("=", 1); header[key] = value
            else: lines.append(line)
    reader = csv.DictReader(lines)
    required = {"call", "token", "layer", "expert", "hit", "bytes_read",
                "cache_slots", "local_slots"}
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise ValueError(f"trace lacks columns: {sorted(required - set(reader.fieldnames or []))}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list); expert_bytes = 0
    for row in reader:
        grouped[int(row["call"])].append(row)
        expert_bytes = max(expert_bytes, int(row["bytes_read"]))
    if not grouped or expert_bytes == 0: raise ValueError("empty trace or no expert byte geometry")
    events = []
    for call in sorted(grouped):
        group = grouped[call]; seen: dict[int, tuple[int, int]] = {}
        for row in group:
            expert = int(row["expert"]); old = seen.get(expert, (int(row["hit"]), 0))
            seen[expert] = (old[0], old[1] + int(row["bytes_read"]))
        events.append(Event(call, int(group[0]["token"]), int(group[0]["layer"]),
                            sorted(seen), int(group[0]["cache_slots"]),
                            int(group[0]["local_slots"]), expert_bytes, seen))
    return events, header


class Simulator:
    def __init__(self, policy: str, capacity: int | None, decode_start: int | None,
                 layers: set[int], ephemeral_fraction: float, hot: set[Key],
                 initial_residents: list[Key]):
        self.policy = policy; self.capacity_override = capacity
        self.decode_start = decode_start; self.layers = sorted(layers)
        self.ephemeral_fraction = ephemeral_fraction; self.hot = hot
        self.initial_residents = initial_residents; self.initialized = False
        self.cache: OrderedDict[Key, None] = OrderedDict(); self.freq: Counter[Key] = Counter()
        self.owner: dict[Key, str] = {}; self.owner_admit = Counter(); self.owner_resident = Counter()
        self.hits = self.misses = self.bytes = self.evictions = 0
        self.decode_hits = self.decode_misses = self.decode_bytes = 0
        self.layer_hits = Counter(); self.layer_access = Counter(); self.layer_bytes = Counter()

    def is_decode(self, event: Event) -> bool:
        return self.decode_start is not None and event.token >= self.decode_start

    def capacity(self, event: Event) -> int:
        cap = self.capacity_override if self.capacity_override is not None else event.capacity
        if self.policy == "decode-protected" and not self.is_decode(event):
            cap = int(cap * self.ephemeral_fraction)
        if self.policy == "topk-replication":
            cap -= sum(key in self.hot for key in self.cache)
        return max(cap, 0)

    def evict(self, victim: Key) -> None:
        del self.cache[victim]; old_owner = self.owner.pop(victim, None)
        if old_owner: self.owner_resident[old_owner] -= 1
        self.evictions += 1

    def initialize(self, event: Event) -> None:
        if self.initialized or self.capacity(event) <= 0: return
        self.initialized = True; cap = self.capacity(event)
        for key in self.initial_residents[-cap:]:
            self.cache[key] = None
            local = min(event.local_capacity, cap)
            owner = "local" if self.owner_resident["local"] < local else "peer"
            self.owner[key] = owner; self.owner_resident[owner] += 1

    def access(self, event: Event, expert: int) -> None:
        self.initialize(event)
        key = (event.layer, expert); cap = self.capacity(event); decode = self.is_decode(event)
        self.layer_access[event.layer] += 1; self.freq[key] += 1
        while len(self.cache) > cap:
            self.evict(next(iter(self.cache)))
        if key in self.cache:
            self.hits += 1; self.layer_hits[event.layer] += 1
            if decode: self.decode_hits += 1
            self.cache.move_to_end(key); return
        self.misses += 1; self.bytes += event.expert_bytes; self.layer_bytes[event.layer] += event.expert_bytes
        if decode: self.decode_misses += 1; self.decode_bytes += event.expert_bytes
        if cap <= 0: return
        if self.policy == "tinylfu" and len(self.cache) >= cap:
            victim = next(iter(self.cache))
            if self.freq[key] < self.freq[victim]: return
        if self.policy == "per-layer-quota":
            quota = max(1, cap // max(len(self.layers), 1))
            residents = [resident for resident in self.cache if resident[0] == event.layer]
            if len(residents) >= quota: self.evict(residents[0])
        if len(self.cache) >= cap:
            if self.policy == "segmented":
                victim = next((k for k in self.cache if self.freq[k] <= 1), next(iter(self.cache)))
            elif self.policy == "decode-protected" and decode:
                # Older prefill entries naturally lead the LRU; prefer one if present.
                victim = next(iter(self.cache))
            else: victim = next(iter(self.cache))
            self.evict(victim)
        self.cache[key] = None
        local = min(event.local_capacity, cap)
        owner = "local" if cap <= local or self.owner_resident["local"] < local else "peer"
        if self.policy == "owner-balanced" and cap > local:
            ll = self.owner_resident["local"] / max(local, 1)
            pl = self.owner_resident["peer"] / max(cap - local, 1)
            owner = "local" if ll <= pl else "peer"
        self.owner[key] = owner; self.owner_admit[owner] += 1; self.owner_resident[owner] += 1

    def run(self, events: list[Event]) -> None:
        for event in events:
            for expert in event.experts: self.access(event, expert)


def observed(events: list[Event]) -> tuple[int, int, int]:
    hits = misses = byte_count = 0
    for event in events:
        for hit, nbytes in event.observed.values():
            hits += bool(hit); misses += not bool(hit); byte_count += nbytes
    return hits, misses, byte_count


def reuse_distances(events: list[Event]) -> tuple[int, list[int]]:
    # Exact LRU stack distance; traces are small enough for the simple ordered stack.
    stack: OrderedDict[Key, None] = OrderedDict(); cold = 0; distances = []
    for event in events:
        for expert in event.experts:
            key = (event.layer, expert)
            if key not in stack: cold += 1
            else:
                keys = list(stack); distances.append(len(keys) - 1 - keys.index(key)); del stack[key]
            stack[key] = None
    return cold, distances


def percentile(values: list[int], q: float) -> int:
    if not values: return 0
    values = sorted(values); return values[min(int((len(values) - 1) * q), len(values) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path); ap.add_argument("--capacity", type=int)
    ap.add_argument("--decode-token-start", type=int,
                    help="first generated-token epoch; enables phase reporting/policies")
    ap.add_argument("--ephemeral-fraction", type=float, default=.20)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--layer-report", action="store_true")
    ap.add_argument("--policies", default=",".join(sorted(POLICIES)))
    args = ap.parse_args()
    if not 0 <= args.ephemeral_fraction <= 1: ap.error("--ephemeral-fraction must be in 0..1")
    events, header = load_trace(args.trace); oh, om, ob = observed(events)
    frequency = Counter((event.layer, expert) for event in events for expert in event.experts)
    hot = {key for key, _ in frequency.most_common(args.top_k)}
    first_seen: set[Key] = set(); initial_residents: list[Key] = []
    for event in events:
        if event.capacity <= 0: continue
        for expert in event.experts:
            key = (event.layer, expert)
            if key in first_seen: continue
            first_seen.add(key)
            if event.observed[expert][0]: initial_residents.append(key)
    cold, distances = reuse_distances(events)
    print(f"trace={args.trace} model_id={header.get('model_id','unknown')} calls={len(events)}")
    print(f"observed unique_hits={oh} unique_misses={om} read_bytes={ob} "
          f"inferred_initial_residents={len(initial_residents)}")
    print(f"reuse cold={cold} samples={len(distances)} p50={percentile(distances,.50)} "
          f"p90={percentile(distances,.90)} p99={percentile(distances,.99)}")
    results = []
    for name in (item.strip() for item in args.policies.split(",")):
        if name not in POLICIES: raise SystemExit(f"unknown policy: {name}")
        if name == "decode-protected" and args.decode_token_start is None:
            print("policy=decode-protected skipped=requires_--decode-token-start"); continue
        sim = Simulator(name, args.capacity, args.decode_token_start,
                        {event.layer for event in events}, args.ephemeral_fraction,
                        hot, initial_residents)
        sim.run(events); results.append((name, sim)); total = sim.hits + sim.misses
        delta = (sim.bytes / ob - 1) * 100 if ob else 0
        print(f"policy={name} hits={sim.hits} misses={sim.misses} hit_rate={sim.hits/max(total,1):.6f} "
              f"read_bytes={sim.bytes} byte_delta={delta:+.2f}% evictions={sim.evictions} "
              f"decode_hits={sim.decode_hits} decode_misses={sim.decode_misses} decode_bytes={sim.decode_bytes} "
              f"owner_admissions={dict(sim.owner_admit)} owner_resident={dict(sim.owner_resident)}")
        if args.layer_report:
            for layer in sorted(sim.layer_access):
                print(f"layer policy={name} layer={layer} hits={sim.layer_hits[layer]} "
                      f"accesses={sim.layer_access[layer]} read_bytes={sim.layer_bytes[layer]}")
    baseline = next((sim for name, sim in results if name == "exact-lru"), None)
    if baseline is not None and args.capacity is None:
        if (baseline.hits, baseline.misses, baseline.bytes) != (oh, om, ob):
            print("ERROR: exact-lru does not reproduce observed trace; alternatives are untrusted")
            return 2
        print("baseline_reproduced=yes")
        alternatives = [(name, sim) for name, sim in results if name != "exact-lru"]
        if alternatives:
            best_name, best = min(alternatives, key=lambda item: item[1].bytes)
            reduction = (1 - best.bytes / baseline.bytes) * 100 if baseline.bytes else 0
            print(f"best_alternative={best_name} byte_reduction={reduction:.2f}% "
                  f"runtime_policy_gate={'yes' if reduction >= 20 else 'no'}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
