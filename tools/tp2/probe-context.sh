#!/usr/bin/env bash
# Find how far --max-context can go at --tp 2 on two 16 GB cards.
#
# Each case starts the engine with a 1-token generation and reports whether
# startup succeeded plus the per-GPU KV pool / reservation actually allocated
# (the KV pool is preallocated at startup, so startup is the whole test).
#
# Usage: ./probe-context.sh "ctx:kv-dtype:rope" ...
#   ./probe-context.sh 262144:bf16:native 262144:int8:native 524288:int8:yarn
# Default case list is used when no argument is given.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${NINFER_BIN:-$ROOT/build/apps/ninfer}"
ARTIFACT="${NINFER_ARTIFACT:-$HOME/projects/ninfer-5060ti/out/qwen3_8_27b_nvfp4.ninfer}"
TDEV="${NINFER_TDEV:-0,5}"
PROMPT="${PROMPT:-hi}"

CASES=("$@")
if [[ ${#CASES[@]} -eq 0 ]]; then
  CASES=(262144:int8:yarn:auto 294912:int8:yarn:explicit 327680:int8:yarn:explicit 393216:int8:yarn:explicit)
fi

# Extra flags for every case, e.g. NINFER_EXTRA="--spec mtp --draft-tokens 3"
EXTRA="${NINFER_EXTRA:-}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
OUTDIR="$ROOT/bench/ctx-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"

printf '%-9s %-6s %-7s %-8s %-12s %-12s %s\n' "ctx" "kv" "rope" "result" "kv pool/gpu" "reserved/gpu" "free/gpu"
for case in "${CASES[@]}"; do
  IFS=: read -r ctx kv rope cap <<<"$case"
  cap="${cap:-auto}"
  log="$OUTDIR/ctx${ctx}-${kv}-${rope}-${cap}.log"
  args=("$ARTIFACT" --prompt "$PROMPT" --tp 2 --devices "$TDEV"
        --max-context "$ctx" --kv-dtype "$kv"
        --no-thinking --greedy --max-new 1)
  if [[ "$cap" == "explicit" ]]; then
    args+=(--kv-capacity "$ctx")
  else
    args+=(--kv-capacity auto)
  fi
  if [[ "$rope" == "yarn" ]]; then
    args+=(--rope yarn --yarn-factor 4.0)
  fi
  if [[ -n "$EXTRA" ]]; then
    # shellcheck disable=SC2206
    args+=($EXTRA)
  fi

  if timeout 600 "$BIN" "${args[@]}" > "$OUTDIR/ctx${ctx}-${kv}-${rope}-${cap}.out" 2> "$log"; then
    result="OK"
  else
    result="FAIL"
  fi

  pool=$(grep -oP 'gpu0 kv pool\s+\K.*'  "$log" | tail -1 | sed 's/ *$//')
  res=$(grep -oP  'gpu0 reserved\s+\K.*' "$log" | tail -1 | sed 's/ *$//')
  free=$(grep -oP 'gpu0 free/total\s+\K.*' "$log" | tail -1 | sed 's/ *$//')
  printf '%-9s %-6s %-7s %-9s %-8s %-12s %-12s %s\n' \
    "$ctx" "$kv" "$rope" "$cap" "$result" "${pool:-none}" "${res:-none}" "${free:-none}"

  if [[ "$result" == "FAIL" ]]; then
    grep -iE "^error|out of memory|exceed" "$log" | head -2 | sed 's/^/    /'
  fi
done

echo
echo "logs: $OUTDIR"
