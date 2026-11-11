#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/scripts/runtime.sh"
if vits_is_running; then echo "Running, PID $(cat "$VITS_PID_FILE"), port $VITS_PORT"; else echo 'Stopped'; fi
echo "Python: $VITS_PYTHON"
echo "Runtime: $VITS_ROOT/runtime"
echo "Log: $VITS_ROOT/data/logs/web.log"
