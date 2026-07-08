#!/usr/bin/env python3
"""Aggregate DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE lines into per-token stats.

Line format:
ds4: CUDA streaming selected layer=U slots=U compact=U global_budget=U before=U after=U hits=U misses=U direct=U gate/up X MiB down Y MiB
Decode tokens are delimited by wrap-around of layer index (layer < previous layer).
"""
import re, sys

pat = re.compile(
    r'streaming selected layer=(\d+) slots=(\d+) compact=(\d+) global_budget=(\d+) '
    r'before=(\d+) after=(\d+) hits=(\d+) misses=(\d+) (?:host=(\d+) )?direct=(\d+) '
    r'gate/up ([0-9.]+) MiB down ([0-9.]+) MiB')

tokens = []   # list of dicts per token (layer-wrap delimited)
cur = None
prev_layer = -1
for line in open(sys.argv[1]):
    m = pat.search(line)
    if not m:
        continue
    layer, slots, compact, budget, before, after, hits, misses = map(int, m.groups()[:8])
    host = int(m.group(9)) if m.group(9) is not None else 0
    direct = int(m.group(10))
    gate_mib, down_mib = float(m.group(11)), float(m.group(12))
    if layer <= prev_layer or cur is None:
        cur = dict(calls=0, hits=0, misses=0, host=0, direct=0, mib=0.0, slots=0, compact=0, budget=budget, after=after)
        tokens.append(cur)
    prev_layer = layer
    cur['calls'] += 1
    cur['hits'] += hits
    cur['misses'] += misses
    cur['host'] += host
    cur['direct'] += direct
    cur['slots'] += slots
    cur['compact'] += compact
    cur['after'] = after
    # bytes actually read from storage = misses that were not served by the host cache
    frac = (misses - host + direct) / max(1, compact)
    cur['mib'] += (gate_mib * 2 + down_mib) * frac

print(f"{len(tokens)} staging waves (prefill chunks + decode tokens)")
print(f"{'wave':>5} {'calls':>6} {'compact':>8} {'hits':>6} {'miss':>6} {'host':>6} {'direct':>7} {'hit%':>6} {'MiB-read':>10} {'cache-fill':>10}")
for i, t in enumerate(tokens):
    ref = t['hits'] + t['misses'] + t['direct']
    hr = 100.0 * (t['hits'] + t['host']) / ref if ref else 0.0
    print(f"{i:>5} {t['calls']:>6} {t['compact']:>8} {t['hits']:>6} {t['misses']:>6} {t['host']:>6} {t['direct']:>7} {hr:>5.1f}% {t['mib']:>10.0f} {t['after']:>10}")

# steady-state = last quarter of decode waves (calls==75-ish, slots small)
decode = [t for t in tokens if t['calls'] >= 70 and t['slots'] <= t['calls'] * 8]
if decode:
    tail = decode[len(decode)//2:]
    ref = sum(t['hits'] + t['misses'] + t['direct'] for t in tail)
    hits = sum(t['hits'] for t in tail)
    host = sum(t['host'] for t in tail)
    mib = sum(t['mib'] for t in tail) / len(tail)
    print(f"\nsteady-state (last {len(tail)} decode tokens): L1 {100.0*hits/max(1,ref):.1f}% + L2 {100.0*host/max(1,ref):.1f}% hits, avg {mib:.0f} MiB read/token")
