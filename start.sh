#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/scripts/runtime.sh"
[[ -x "$VITS_PYTHON" ]] || { echo 'Run ./install.sh first.'; exit 1; }
exec 9>"$VITS_ROOT/runtime/run/start.lock"
flock -n 9 || { echo 'Another start operation is running.'; exit 1; }
if vits_is_running; then echo "VITS is already running: PID $(cat "$VITS_PID_FILE")"; exit 0; fi
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  if [[ "${VITS_GPU:-auto}" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=''
  elif [[ "${VITS_GPU:-auto}" != auto ]]; then
    export CUDA_VISIBLE_DEVICES="$VITS_GPU"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    gpu_list="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2,2nr | cut -d, -f1 | xargs | tr ' ' ',')" || gpu_list=''
    export CUDA_VISIBLE_DEVICES="$gpu_list"
  fi
fi
cd "$VITS_ROOT"
if [[ "${1:-}" == --foreground ]]; then
  exec "$VITS_PYTHON" -u "$VITS_ROOT/app.py" --host "$VITS_HOST" --port "$VITS_PORT" 9>&-
fi
nohup "$VITS_PYTHON" -u "$VITS_ROOT/app.py" --host "$VITS_HOST" --port "$VITS_PORT" \
  >> "$VITS_ROOT/data/logs/web.log" 2>&1 < /dev/null 9>&- &
pid=$!
printf '%s\n' "$pid" > "$VITS_PID_FILE"
for attempt in $(seq 1 60); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'Startup failed. See data/logs/web.log.'; tail -n 20 "$VITS_ROOT/data/logs/web.log"; exit 1
  fi
  if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$VITS_PORT/" >/dev/null 2>&1; then
    echo "VITS started. PID $pid, port $VITS_PORT, GPU ${CUDA_VISIBLE_DEVICES:-CPU}."
    echo "Log: $VITS_ROOT/data/logs/web.log"
    exit 0
  fi
  sleep 1
done
echo "Process is running but not ready yet. Inspect $VITS_ROOT/data/logs/web.log"; exit 1
