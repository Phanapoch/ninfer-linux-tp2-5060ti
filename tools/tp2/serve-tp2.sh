#!/usr/bin/env bash
# Run ninfer-serve with Qwen3.8-27B NVFP4 tensor-parallel on two RTX 5060 Ti 16GB.
#
# Validated on this box (2x RTX 5060 Ti, sm_120a, CUDA 13.1, no CUDA P2P):
#   327680 ctx, int8 KV, YaRN x4, MTP3 -> 5.48 GiB KV/GPU, 651 MiB free/GPU  (recommended)
#   360448 ctx same flags             -> 6.03 GiB KV/GPU,  91 MiB free/GPU  (the max; tight)
# Context beyond 262144 needs --rope yarn (the artifact's registered native ceiling).
# An explicit --kv-capacity is used because "auto" reserves 1 GiB of headroom.
#
# Usage:
#   ./serve-tp2.sh                 # foreground (Ctrl-C to stop)
#   ./serve-tp2.sh start --daemon  # background, logs + pidfile, waits until healthy
#   ./serve-tp2.sh status | logs | stop | restart
#
# Env overrides (all optional):
#   ARTIFACT  path to the .ninfer file      DEVICES   CUDA device ids, default 0,5
#   HOST      127.0.0.1                     PORT      default 8090
#   CTX       327680                        KV_DTYPE  int8 | bf16
#   KV_CAP    explicit | auto               MTP       draft tokens; 0 disables MTP
#   ROPE      yarn | native                 CONCURRENCY  default 1
#   API_KEY   unset = no auth               EXTRA     extra ninfer-serve flags
#   FORCE     1 = start even if the GPUs are busy
#
# Endpoints: /health  /v1/models  /v1/chat/completions  /v1/responses  /v1/messages

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${NINFER_BIN:-$ROOT/build/apps/ninfer-serve}"
ARTIFACT="${ARTIFACT:-$HOME/projects/ninfer-5060ti/out/qwen3_8_27b_nvfp4.ninfer}"
DEVICES="${DEVICES:-0,5}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8095}"
CTX="${CTX:-327680}"
KV_DTYPE="${KV_DTYPE:-int8}"
KV_CAP="${KV_CAP:-explicit}"
MTP="${MTP:-3}"
ROPE="${ROPE:-yarn}"
CONCURRENCY="${CONCURRENCY:-1}"
API_KEY="${API_KEY:-}"
EXTRA="${EXTRA:-}"

STATE_DIR="${STATE_DIR:-$HOME/.local/state/ninfer-tp2}"
PIDFILE="$STATE_DIR/serve.pid"
LOGFILE="$STATE_DIR/serve.log"

# Device ids must be interpreted the same way nvidia-smi numbers them. Without this,
# CUDA's default order is by speed and the ids point at other cards.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

die() { echo "ERROR: $*" >&2; exit 1; }

build_args() {
  ARGS=("$ARTIFACT"
        --host "$HOST" --port "$PORT"
        --tp 2 --devices "$DEVICES"
        --max-context "$CTX"
        --kv-dtype "$KV_DTYPE"
        --max-concurrency "$CONCURRENCY")
  if [[ "$KV_CAP" == "auto" ]]; then ARGS+=(--kv-capacity auto); else ARGS+=(--kv-capacity "$CTX"); fi
  if [[ "$ROPE" == "yarn" ]]; then ARGS+=(--rope yarn --yarn-factor 4.0); fi
  if [[ "$MTP" != "0" ]]; then ARGS+=(--spec mtp --draft-tokens "$MTP"); fi
  [[ -n "$API_KEY" ]] && ARGS+=(--api-key "$API_KEY")
  if [[ -n "$EXTRA" ]]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA)
  fi
}

preflight() {
  [[ -x "$BIN" ]] || die "no ninfer-serve at $BIN (build it: cmake --build build -j)"
  [[ -f "$ARTIFACT" ]] || die "no artifact at $ARTIFACT"
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found"

  local busy=0
  IFS=',' read -ra devs <<<"$DEVICES"
  for d in "${devs[@]}"; do
    local pids
    pids=$(nvidia-smi -i "$d" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | paste -sd, -)
    local used
    used=$(nvidia-smi -i "$d" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [[ -n "$pids" ]]; then
      echo "note: gpu$d already has compute processes [$pids] using ${used} MiB"
      busy=1
    fi
  done
  if [[ "$busy" == "1" && "${FORCE:-0}" != "1" ]]; then
    echo
    echo "The target GPUs are busy. TP2 needs both cards almost empty."
    echo "Stop the other process, or re-run with FORCE=1 to try anyway." >&2
    exit 1
  fi

  # Fail fast on a taken port instead of after a 17 GiB model load.
  local host_for_test="$HOST"
  [[ "$host_for_test" == "0.0.0.0" || "$host_for_test" == "::" ]] && host_for_test="127.0.0.1"
  if (exec 3<>"/dev/tcp/$host_for_test/$PORT") 2>/dev/null; then
    exec 3>&- 2>/dev/null
    die "port $PORT on $HOST is already in use; pick another with PORT=<n>"
  fi
  if [[ "$PORT" -ge 1 && "$PORT" -le 65535 ]]; then :; else
    die "--port must be in [1,65535] (got $PORT)"
  fi
}

wait_healthy() {
  local deadline=$((SECONDS + 300))
  while ((SECONDS < deadline)); do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -f "$PIDFILE" ]] && ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      return 1
    fi
    sleep 2
  done
  return 1
}

banner() {
  echo
  echo "ninfer-serve up:  http://$HOST:$PORT"
  echo "  ctx=$CTX  kv=$KV_DTYPE  rope=$ROPE  mtp=$MTP  concurrency=$CONCURRENCY  devices=$DEVICES"
  echo "  health:  curl http://$HOST:$PORT/health"
  echo "  models:  curl http://$HOST:$PORT/v1/models"
  local auth=()
  [[ -n "$API_KEY" ]] && auth=(-H "Authorization: Bearer $API_KEY")
  echo "  chat:"
  echo "    curl http://$HOST:$PORT/v1/chat/completions ${auth[*]:-} \\"
  echo "      -H 'Content-Type: application/json' \\"
  echo "      -d '{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"สวัสดี\"}],\"max_tokens\":256}'"
  echo
  echo "  logs:  tail -f $LOGFILE      stop:  $0 stop"
  echo
  echo "note: at --tp 2 with MTP, prefix reuse is disabled by design (answers are unaffected)."
}

start_foreground() {
  banner
  echo "running in the foreground (Ctrl-C to stop)"
  build_args
  exec "$BIN" "${ARGS[@]}"
}

start_daemon() {
  mkdir -p "$STATE_DIR"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    die "already running (pid $(cat "$PIDFILE")); use '$0 stop' first"
  fi
  build_args
  echo "starting ninfer-serve ... (loading ~17 GiB of weights, this takes ~20 s)"
  nohup "$BIN" "${ARGS[@]}" >>"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  if wait_healthy; then
    banner
  else
    echo "server did not become healthy within 300 s; last log lines:" >&2
    tail -15 "$LOGFILE" >&2
    exit 1
  fi
}

stop_server() {
  [[ -f "$PIDFILE" ]] || { echo "no pidfile ($PIDFILE)"; return 0; }
  local pid
  pid=$(cat "$PIDFILE")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "not running (stale pidfile)"; rm -f "$PIDFILE"; return 0
  fi
  echo "stopping pid $pid ..."
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || { echo "stopped"; rm -f "$PIDFILE"; return 0; }
    sleep 1
  done
  echo "still alive; SIGKILL"
  kill -KILL "$pid" 2>/dev/null
  rm -f "$PIDFILE"
}

status_server() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "pid $(cat "$PIDFILE") alive"
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo "healthy: http://$HOST:$PORT/health"
    else
      echo "not answering on http://$HOST:$PORT/health yet"
    fi
  else
    echo "not running"
  fi
}

CMD="${1:-start}"
DAEMON=0
for a in "$@"; do [[ "$a" == "--daemon" ]] && DAEMON=1; done

case "$CMD" in
  start)   preflight; if [[ "$DAEMON" == "1" ]]; then start_daemon; else start_foreground; fi ;;
  stop)    stop_server ;;
  restart) stop_server; preflight; start_daemon ;;
  status)  status_server ;;
  logs)    tail -f "$LOGFILE" ;;
  -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//' ;;
  *) echo "unknown command: $1 (start|stop|restart|status|logs)"; exit 2 ;;
esac
