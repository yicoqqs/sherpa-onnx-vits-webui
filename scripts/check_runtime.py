"""Installation diagnostics; no external model download or user data changes."""
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent.parent
for name in ('torch', 'fastapi', 'uvicorn', 'soundfile', 'scipy', 'librosa', 'numba',
             'sherpa_onnx', 'onnx', 'onnxruntime', 'faster_whisper', 'yt_dlp', 'pypinyin', 'cn2an', 'opencc'):
    importlib.import_module(name)
for name in ('ffmpeg', 'ffprobe'):
    executable = shutil.which(name)
    if not executable or not Path(executable).resolve().is_relative_to(root / 'runtime'):
        raise RuntimeError(f'{name} must be installed inside {root / "runtime"}: {executable}')
import torch
print(json.dumps(dict(python=sys.version, interpreter=sys.executable, torch=torch.__version__,
                     cuda=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                     devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]), indent=2))
subprocess.run([shutil.which('ffmpeg'), '-version'], check=True, stdout=subprocess.DEVNULL)
if torch.cuda.is_available():
    value = torch.ones(8, 8, device='cuda')
    assert (value @ value)[0, 0].item() == 8
print('Runtime checks passed.')
