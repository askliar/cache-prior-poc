# Cache-Prior proof-of-concept

This project tests whether cache-aware reranking can reduce Mixture-of-Experts
weight-cache misses without materially damaging language-model quality.

It deliberately simulates the expert cache while keeping all model weights
resident. It proves routing and quality behavior; it does **not** measure
offloading latency or claim a runtime speedup.

The implementation follows *Mixture of Cache-Conditional Experts for Efficient
Mobile Device Inference*:

```text
z' = z + lambda * running_mean(max(z) - min(z)) * protected_mask
```

`z'` selects top-k expert IDs. Expert mixing weights are gathered from
`softmax(z)`, not `softmax(z')`.

## Implemented comparisons

1. Original routing, no cache: quality baseline.
2. Original routing + LRU: live simulation during the same baseline forward.
3. Original routing + Belady: offline replay of the original route trace.
4. Cache-Prior + LRU: a new model forward because routing changes hidden states.

## Scope

- Hugging Face Transformers model execution.
- Initial model adapter: `allenai/OLMoE-1B-7B-0924`.
- Generic Hugging Face text datasets through YAML.
- Batch size one, teacher-forced causal-LM perplexity.
- One independent routed-expert cache per MoE layer.
- Compact token/layer route traces.
- Aggregate and per-layer hit/miss metrics.
- Estimated parameter and byte transfers.

Not implemented:

- actual expert offloading,
- inference-engine integration,
- latency or throughput measurement,
- training,
- checkpoint conversion,
- multiple requests in one batch,
- lm-evaluation-harness tasks.

## Installation

Python 3.10–3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plot]'
```

For a GPU that cannot hold the BF16 model, install the quantization extra and
change the model config to `quantization: 8bit`:

```bash
python -m pip install -e '.[dev,plot,quantization]'
```

Use identical quantization for original and Cache-Prior runs.

## Validate without downloading a model

```bash
cacheprior validate-config --config configs/experiments/smoke.yaml
pytest
```

The integration tests instantiate a tiny random OLMoE locally; they do not
download the 7B checkpoint.

## Small smoke run

First run original routing:

```bash
cacheprior run \
  --config configs/experiments/smoke.yaml \
  --routing original
```

That single run writes baseline quality, live LRU statistics, a compact route
trace, and offline Belady statistics.

Then run Cache-Prior:

```bash
cacheprior run \
  --config configs/experiments/smoke.yaml \
  --routing cache_prior \
  --lambda 0.5
```

Each command prints its immutable run directory.

## Two-dataset decision matrix

```bash
cacheprior matrix \
  --base-config configs/experiments/proof.yaml \
  --datasets \
    configs/datasets/wikitext2.yaml \
    configs/datasets/c4-en.yaml \
  --lambdas 0.5 \
  --output-root runs \
  --summary-dir runs/summary
```

This loads the model once and performs:

- one original forward per dataset,
- one Cache-Prior forward per dataset,
- LRU and Belady accounting without extra original-model forwards.

If `lambda=0.5` gives an ambiguous trade-off:

```bash
cacheprior matrix \
  --base-config configs/experiments/proof.yaml \
  --datasets \
    configs/datasets/wikitext2.yaml \
    configs/datasets/c4-en.yaml \
  --lambdas 0.25 0.5 0.75 \
  --output-root runs \
  --summary-dir runs/summary-three-lambdas
```

## Replay a trace

```bash
cacheprior replay \
  --trace runs/<original-run> \
  --route original \
  --policy belady \
  --capacity 32
```

## Summarize existing runs

```bash
cacheprior summarize \
  --runs runs/<original-run> runs/<cache-prior-run> \
  --output-dir runs/comparison
```

The report expands the original run into three rows—no-cache quality, LRU, and
Belady—and adds one Cache-Prior+LRU row.

## Add another dataset

Create a YAML file:

```yaml
source: some-org/some-dataset
subset: null
split: validation
revision: null
text_field: text
mode: document
separator: "\n\n"
prediction_length: 512
max_windows: 32
streaming: true
```

Use `mode: concatenate` to join records into one token stream, or
`mode: document` to keep windowing inside each document. No model or routing
code should change.

For durable experiment provenance, replace `revision: null` with immutable Hub
revisions after the first successful download.

## Output

Every run contains:

```text
resolved_config.yaml
environment.json
dataset_manifest.json
model_manifest.json
metrics.json
per_window.jsonl
traces/*.npz
```

Compact traces contain original and selected expert IDs, selected original
probabilities, cache-hit masks, per-token logit ranges, and running range
means. Raw full router logits are intentionally not stored.

## Correctness properties enforced by tests

- Original routing wrapper is transparent.
- Cache-Prior with lambda zero is transparent.
- Original top-J experts are retained.
- Mixing weights come from original probabilities.
- Hits are measured against the pre-token cache state.
- LRU update order is deterministic.
- Belady never has more misses than LRU on randomized fixed traces.
- Full-sequence Cache-Prior execution matches token-by-token execution on a
  tiny causal OLMoE.
- Cache state resets per window while range means persist within a run.

See [DESIGN.md](DESIGN.md) for the exact lifecycle and metric contract.

## Interpreting results

A useful initial result is:

- fewer Cache-Prior+LRU misses on both WikiText-2 and C4,
- no more than 5% relative perplexity increase on either,
- 100% top-J retention.

A strong result is 20–25% fewer misses with no more than 3% relative
perplexity increase on both datasets.

Belady is optimal only for the fixed original-routing trace. Cache-Prior can
legitimately beat that miss count by changing the trace.

