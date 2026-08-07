#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 CONFIG OUTPUT_ROOT LOG_DIR LAMBDA [LAMBDA ...]" >&2
    exit 2
fi

config=$1
output_root=$2
log_dir=$3
shift 3
lambdas=("$@")

IFS=, read -r -a gpus <<< "${CACHEPRIOR_GPUS:-0}"
executable=${CACHEPRIOR_EXECUTABLE:-cacheprior}
if (( ${#lambdas[@]} > ${#gpus[@]} )); then
    echo "need at least one GPU per lambda" >&2
    exit 2
fi

mkdir -p "$output_root" "$log_dir"
pids=()
labels=()
for index in "${!lambdas[@]}"; do
    lambda=${lambdas[$index]}
    gpu=${gpus[$index]}
    log="$log_dir/lambda-${lambda}.log"
    echo "launch lambda=$lambda gpu=$gpu log=$log"
    CUDA_VISIBLE_DEVICES=$gpu \
        "$executable" run \
        --config "$config" \
        --routing cache_prior \
        --lambda "$lambda" \
        --output-root "$output_root" \
        >"$log" 2>&1 &
    pids+=("$!")
    labels+=("lambda=$lambda gpu=$gpu")
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "completed ${labels[$index]}"
    else
        code=$?
        echo "failed ${labels[$index]} exit=$code" >&2
        status=$code
    fi
done
exit "$status"
