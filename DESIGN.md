# Experiment contract

This file defines behavior that must remain stable across implementations.

## Processing order

- Batch size is one.
- A window contains `S+1` source tokens.
- The model consumes the first `S` and scores the following `S`.
- There are exactly `S` routing events per MoE layer.
- Every layer owns an independent expert cache and logit-range estimator.
- A full-sequence router call processes its token rows in position order.
- Incremental decoding may call a router repeatedly; its cumulative token
  count must still equal `S`.

## State lifecycle

- Routed-expert caches start empty and reset before each window.
- Range estimators start empty at the beginning of each dataset/policy run.
- Range estimators persist between windows in the same run.
- Nothing persists across datasets or routing policies.
- Shared experts and dense parameters are treated as permanently resident.

## Hit accounting

For one layer/token event:

1. Snapshot cache membership.
2. Select all top-k experts.
3. Compute hits against the snapshot.
4. Update the cache.

All top-k IDs are unique. The proof configuration requires cache capacity to
be at least top-k.

Selected experts are updated from lowest original probability to highest.
Thus, the highest-probability selected expert is most recent after the event.

## Original routing

The wrapped original route must return the installed Transformers router's
expert IDs and scores unchanged.

## Cache-Prior

For each token and layer:

```text
p = softmax(z)
delta = max(z) - min(z)
delta_avg = inclusive cumulative mean(delta)
protected = in_cache OR original_top_j
z_prime = z + lambda * delta_avg * protected
selected = top_k(z_prime)
weights = gather(p, selected)
```

If the original model renormalizes selected weights, the gathered weights are
renormalized identically.

The current token's `delta` is included in `delta_avg` before reranking. This
defines cold-start behavior without future information.

## LRU

LRU is simulated live for original and Cache-Prior runs. Original-routing live
LRU counts must equal replay over the saved trace.

## Belady

Belady sees only the complete original expert-ID trace for one window. It
resets per window and layer. On eviction, it removes the resident expert whose
next use is farthest in the future; no future use is infinity.

Belady does not alter model output or perplexity. It is not run on Cache-Prior
traces in the four-way comparison.

## Quality

```text
PPL = exp(sum(window NLL) / sum(window scored tokens))
```

Per-window perplexities are diagnostics and are never averaged into the final
metric.

## Transfer estimates

For every layer:

```text
parameters_fetched =
    cache_misses * parameters_per_routed_expert
estimated_bytes =
    parameters_fetched * configured_storage_bits / 8
```

These are modelled quantities, not observed I/O or latency.

## Claim boundary

The harness may establish a quality/cache-locality trade-off. It may not claim
runtime, memory, energy, or device-speed improvements without an inference
engine that actually moves expert weights.
