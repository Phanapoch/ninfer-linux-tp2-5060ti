#!/usr/bin/env bash
# TP2 decode throughput benchmark on two RTX 5060 Ti (no P2P).
#
# Mirrors the methodology in the Windows-TP2 fork README: single request, greedy,
# thinking off, --ignore-eos, short prompt, engine-committed decode tok/s.
#
# Usage:
#   ./bench-tp2.sh [--mtp N]... [--max-new N] [--prompt "..."]
# Defaults: MTP 0 3 4, max-new 512.
#
# Env:
#   NINFER_ARTIFACT  path to the .ninfer file (default: the Ostfralla nvfp4-split copy)
#   NINFER_TDEV      two CUDA device ids (default 0,5; CUDA_DEVICE_ORDER=PCI_BUS_ID is set)

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${NINFER_BIN:-$ROOT/build/apps/ninfer}"
ARTIFACT="${NINFER_ARTIFACT:-$HOME/projects/ninfer-5060ti/out/qwen3_8_27b_nvfp4.ninfer}"
TDEV="${NINFER_TDEV:-0,5}"
PROMPT="${PROMPT:-Explain how photosynthesis works in detail.}"
MAX_NEW=512
MTP_LIST=(0 3 4)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mtp) MTP_LIST=($2); shift 2 ;;
    --max-new) MAX_NEW="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

[[ -x $BIN ]] || { echo "no binary at $BIN"; exit 1; }
[[ -f $ARTIFACT ]] || { echo "no artifact at $ARTIFACT"; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
OUTDIR="$ROOT/bench/tp2-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"

echo "artifact: $ARTIFACT"
echo "devices:  $TDEV   max-new: $MAX_NEW"
echo "logs:     $OUTDIR"
echo

printf '%-10s %-14s %-14s %s\n' "config" "decode tok/s" "prefill tok/s" "notes"
for m in "${MTP_LIST[@]}"; do
  log="$OUTDIR/mtp$m.log"
  args=("$ARTIFACT" --prompt "$PROMPT" --tp 2 --devices "$TDEV"
        --no-thinking --greedy --ignore-eos --max-new "$MAX_NEW")
  if [[ "$m" != "0" ]]; then
    args+=(--spec mtp --draft-tokens "$m")
  fi
  if ! "$BIN" "${args[@]}" > "$OUTDIR/mtp$m.out" 2> "$log"; then
    printf '%-10s %-14s %-14s %s\n' "MTP$m" "FAILED" "-" "see $log"
    continue
  fi
  dec=$(grep -oP 'decode speed\s+\K[0-9.]+'  "$log" | tail -1)
  pre=$(grep -oP 'prefill speed\s+\K[0-9.]+' "$log" | tail -1)
  acc=$(grep -iP 'acceptance|accepted' "$log" | tail -1 | sed 's/^ *//')
  printf '%-10s %-14s %-14s %s\n' "MTP$m" "${dec:-?}" "${pre:-?}" "${acc:0:60}"
done

echo
echo "full logs in $OUTDIR"
