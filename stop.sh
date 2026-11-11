#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/scripts/runtime.sh"
if ! vits_is_running; then echo 'VITS is not running.'; exit 0; fi
pid="$(cat "$VITS_PID_FILE")"
kill -TERM "$pid"
for attempt in $(seq 1 30); do
  if ! vits_is_running; then rm -f -- "$VITS_PID_FILE"; echo 'VITS stopped.'; exit 0; fi
  sleep 1
done
echo "PID $pid has not stopped yet. Inspect data/logs/web.log; no unrelated process was killed."
exit 1
