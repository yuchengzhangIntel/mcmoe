#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-/path/to/qwen-moe}"
ATTN_BITS="${ATTN_BITS:-4}"
MOE_BITS="${MOE_BITS:-2}"
DATASET="${DATASET:-wikitext2}"
EVAL_DATASETS="${EVAL_DATASETS:-}"
NSAMPLES="${NSAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GROUPSIZE="${GROUPSIZE:-128}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
TASKS="${TASKS:-}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
GEN_KWARGS="${GEN_KWARGS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ "$MODEL_PATH" = "/path/to/qwen-moe" ]; then
  echo "Please set MODEL_PATH before running this script."
  exit 1
fi

CMD=(
  python qwen_gptq.py "$MODEL_PATH"
  --attn_bits "$ATTN_BITS"
  --moe_bits "$MOE_BITS"
  --dataset "$DATASET"
  --nsamples "$NSAMPLES"
  --batch_size "$BATCH_SIZE"
  --groupsize "$GROUPSIZE"
  --seqlen "$SEQLEN"
  --seed "$SEED"
  --attn_implementation "$ATTN_IMPL"
  --eval_ppl
)

if [ -n "$EVAL_DATASETS" ]; then
  CMD+=(--eval_datasets "$EVAL_DATASETS")
fi

if [ -n "$TASKS" ]; then
  CMD+=(--tasks "$TASKS" --lm_eval_batch_size "$LM_EVAL_BATCH_SIZE")
fi

if [ -n "$GEN_KWARGS" ]; then
  CMD+=(--gen_kwargs "$GEN_KWARGS")
fi

if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Running: ${CMD[*]}"
"${CMD[@]}"