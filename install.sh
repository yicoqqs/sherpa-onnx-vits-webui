#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/scripts/runtime.sh"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'This installer supports Linux x86_64.'; exit 1; }
exec 9>"$VITS_ROOT/runtime/run/install.lock"
flock -n 9 || { echo 'Another installation is running.'; exit 1; }
if vits_is_running; then echo 'Stop VITS with ./stop.sh before installing dependencies.'; exit 1; fi
if [[ ! -x "$VITS_ROOT/runtime/bin/micromamba" ]]; then
  curl -fLsS --retry 2 --connect-timeout 20 --max-time 180 \
    "${MICROMAMBA_URL:-https://micro.mamba.pm/api/micromamba/linux-64/latest}" \
    -o "$VITS_ROOT/runtime/downloads/micromamba.tar.bz2"
  tar -xjf "$VITS_ROOT/runtime/downloads/micromamba.tar.bz2" -C "$VITS_ROOT/runtime/bin" --strip-components=1 bin/micromamba
fi
if [[ ! -x "$VITS_PYTHON" ]]; then
  if [[ -f "$VITS_ROOT/conda-linux-64.lock.txt" ]]; then
    "$VITS_ROOT/runtime/bin/micromamba" --no-rc create -y -p "$VITS_ROOT/runtime/py312" \
      --file "$VITS_ROOT/conda-linux-64.lock.txt"
  else
    "$VITS_ROOT/runtime/bin/micromamba" --no-rc create -y -p "$VITS_ROOT/runtime/py312" \
      -c "${VITS_CONDA_CHANNEL:-https://conda.anaconda.org/conda-forge}" \
      python=3.12 pip uv ffmpeg libsndfile
  fi
fi
if [[ ! -x "$VITS_ROOT/runtime/py312/bin/uv" ]]; then
  "$VITS_ROOT/runtime/bin/micromamba" --no-rc install -y -p "$VITS_ROOT/runtime/py312" \
    -c "${VITS_CONDA_CHANNEL:-https://conda.anaconda.org/conda-forge}" uv
fi
export PIP_INDEX_URL="${VITS_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
if [[ -f "$VITS_ROOT/requirements-linux.lock.txt" ]]; then
  "$VITS_ROOT/runtime/py312/bin/uv" pip install --python "$VITS_PYTHON" \
    --index-url "$PIP_INDEX_URL" -r "$VITS_ROOT/requirements-linux.lock.txt"
else
  "$VITS_ROOT/runtime/py312/bin/uv" pip install --python "$VITS_PYTHON" \
    --index-url "$PIP_INDEX_URL" 'torch==2.8.0' -r "$VITS_ROOT/requirements.txt"
fi
if [[ ! -f "$VITS_ROOT/deploy.env" ]]; then
  addresses="$(hostname -I | xargs | tr ' ' ',')"
  cat > "$VITS_ROOT/deploy.env" <<EOF
VITS_HOST=0.0.0.0
VITS_PORT=7860
VITS_ALLOWED_HOSTS=localhost,127.0.0.1${addresses:+,$addresses}
# Choose auto, cpu, or a physical GPU index (0, 1, ...).
VITS_GPU=auto
VITS_PIP_INDEX_URL=$PIP_INDEX_URL
EOF
fi
"$VITS_PYTHON" "$VITS_ROOT/scripts/freeze_runtime.py" > "$VITS_ROOT/requirements-linux.lock.txt"
"$VITS_ROOT/runtime/bin/micromamba" --no-rc list -p "$VITS_ROOT/runtime/py312" --explicit > "$VITS_ROOT/runtime/conda-explicit.txt"
cp "$VITS_ROOT/runtime/conda-explicit.txt" "$VITS_ROOT/conda-linux-64.lock.txt"
"$VITS_PYTHON" "$VITS_ROOT/scripts/check_runtime.py"
echo "Installed entirely under $VITS_ROOT. Start with: $VITS_ROOT/start.sh"
