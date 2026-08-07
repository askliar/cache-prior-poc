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

## Layer-wise teacher-forced execution

The router receives all `T` token logits for a layer at once. For any fixed
route tensor `[T, K]`, the harness constructs every pre-token LRU state as one
boolean tensor `[T, E]` using timestamp scatter, prefix `cummax`, and `topk`.
The Cache-Prior promotion is then the single batched operation

```text
reranked_logits[T, E] =
    logits[T, E] + scale[T, 1] * protected_cache_mask[T, E]
```

This removes the Python loop from baseline LRU accounting and from any replay
whose expert choices are already known.

Within each token, selected experts are inserted into LRU state from largest
to smallest original router weight. Consequently, the largest-weight expert
is the least recent of that token's selections and is evicted first.

Exact Cache-Prior routing still has a causal recurrence: the selected experts
at token `t` update the LRU state used to rerank token `t+1`. A mask generated
only from the original choices can be reranked in one shot, but it is a
frozen-trace approximation because changed choices are not fed back into later
cache states. The default experiment path keeps that feedback exact; the
vectorized fixed-trace implementation is tested against the sequential LRU
reference.

## Implemented comparisons

1. Original routing, no cache: quality baseline.
2. Original routing + LRU: live simulation during the same baseline forward.
3. Original routing + Belady: offline replay of the original route trace.
4. Cache-Prior + LRU: a new model forward because routing changes hidden states.

## Scope

- Hugging Face Transformers model execution.
- Model adapters for OLMoE, Qwen1.5/Qwen2-MoE, DeepSeek-V2, and Nemotron-H.
- PPL-only evaluation for dense causal language models.
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

For a denser view of the important part of the trade-off curve, replace the
explicit `--lambdas` argument with:

```text
--lambda-grid log --lambda-points 50 --lambda-min 0.001
```

This produces a fixed grid containing 0 and 1 with the intermediate points
log-spaced toward zero. It does not run unless explicitly selected.

## Paper WikiText reproduction

The paper-compatible configurations use the full WikiText-2-raw-v1 validation
split, tokenize the exact `"\n\n"`-joined text blob, use 1,024-token windows,
keep the original top-2 experts, and cache half of each model's routed experts:

```text
configs/experiments/paper-qwen-wikitext.yaml
configs/experiments/paper-deepseek-wikitext.yaml
```

Run each original-routing baseline once:

```bash
cacheprior run \
  --config configs/experiments/paper-qwen-wikitext.yaml \
  --routing original

cacheprior run \
  --config configs/experiments/paper-deepseek-wikitext.yaml \
  --routing original
```

On a multi-GPU node, independent lambda points can run concurrently while each
logical experiment remains batch size one:

```bash
CACHEPRIOR_GPUS=0,1,2,3 \
CACHEPRIOR_EXECUTABLE=cacheprior \
bash scripts/run_parallel_lambdas.sh \
  configs/experiments/paper-qwen-wikitext.yaml \
  runs/paper-qwen \
  runs/paper-qwen-logs \
  0.1 0.2 0.3 0.4
```

DeepSeek-V2-Lite's Hub checkpoint contains custom modeling code written for
Transformers 4.x. Use Transformers 4.48.3 for that model; the loader supports
both the 4.x `torch_dtype` and 5.x `dtype` loading contracts.

## Nemotron 3 Nano NVFP4

The stack includes the immutable release configuration:

```text
configs/experiments/nemotron3-nano-nvfp4.yaml
```

Install the custom model's dependencies in a separate Transformers 4.x
environment:

```bash
python -m pip install -e '.[dev,plot,nemotron]'
```

The adapter follows Nemotron-H's released router contract: 128 routed experts,
top-6 selection, sigmoid routing scores, learned score-correction bias,
normalized weights, and a 2.5 routed-weight scale. The configuration uses a
half-size cache of 64 experts and accounts for routed expert storage at four
bits per logical parameter.

The checkpoint is a native ModelOpt NVFP4 export. `quantization: native` tells
the loader to preserve the checkpoint representation and use its published
`from_pretrained(..., device_map="auto")` path rather than wrapping it with
BitsAndBytes or casting its packed weights. Use an NVIDIA environment that
supports the checkpoint's `hf_quant_config.json`.

Run it against any configured datasets with:

```bash
cacheprior matrix \
  --base-config configs/experiments/nemotron3-nano-nvfp4.yaml \
  --datasets \
    configs/datasets/wikitext2.yaml \
    configs/datasets/c4-en.yaml \
  --lambdas 0.1 0.2 0.4 \
  --output-root runs/nemotron3-nano \
  --summary-dir runs/nemotron3-nano/summary
```

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

## FP8 evaluation on H100

The FP8 configurations use checkpoints with published Transformers loading
paths and pin their Hub revisions:

```text
configs/experiments/qwen3-8b-fp8-wikitext.yaml
configs/experiments/nemotron3-nano-fp8.yaml
```

`Qwen/Qwen3-8B-FP8` is dense, so its run reports teacher-forced PPL only:

```bash
cacheprior run \
  --config configs/experiments/qwen3-8b-fp8-wikitext.yaml
```

The Nemotron FP8 checkpoint uses the same Nemotron-H router adapter as the
NVFP4 release. One original-routing forward reports baseline PPL, live LRU,
and offline Belady metrics. For a four-window validation run:

```bash
cacheprior run \
  --config configs/experiments/nemotron3-nano-fp8.yaml \
  --dataset configs/datasets/wikitext2-quick.yaml
```

Qwen uses `quantization: native`, following its published Transformers loading
path. Nemotron uses `quantization: modelopt_fp8`: the loader consumes the
unified checkpoint's static per-tensor input and weight scales and evaluates
each FP8 linear with `torch._scaled_mm`. This keeps the standard Transformers
model and its router visible to the cache policies without depending on an
inference engine. The logical expert-cache transfer accounting uses eight bits
per parameter.

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
- LRU update order is deterministic and processes larger router weights first.
- Cache-Prior reranking uses only finite logits.
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
