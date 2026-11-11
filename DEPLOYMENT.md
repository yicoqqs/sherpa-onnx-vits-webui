# Linux deployment (scripts only)

The project is deployed at `/usr/local/share/vits`. No systemd, cron, supervisor,
or other system service is registered.

```bash
cd /usr/local/share/vits
./install.sh            # First installation / missing dependencies, Python 3.12
./start.sh              # Background process; survives closing the SSH terminal
./status.sh
./stop.sh
./start.sh --foreground # Optional foreground mode, Ctrl+C to stop
```

Run these as the owner of the deployment directory, not with sudo. Creating the
directory may require an administrator once. The installer writes inside the
deployment directory and does not modify system Python, shell startup files, or
system services. Linux itself and the installed NVIDIA driver remain host prerequisites.

All application runtime files are below this directory:

- `runtime/py312/`: Python, FFmpeg, ffprobe, native libraries, pip packages and CUDA userspace libraries.
- `runtime/bin/`: micromamba bootstrap tool.
- `runtime/packages/`, `runtime/mamba/`: environment package cache and metadata.
- `runtime/run/`: PID and lock files.
- `cache/`: pip, uv, Hugging Face, Torch, Numba, Triton, Python bytecode and other caches.
- `data/`: projects, models, Whisper weights, jobs, trash, logs and temporary files.
- `deploy.env`: listen address, trusted hostnames, port and GPU selection.
- `requirements-linux.lock.txt`, `conda-linux-64.lock.txt`: actual installed versions, reused on subsequent installations.

On this server the browser address is `http://192.168.0.150:7860`.
The default `VITS_GPU=auto` exposes all GPUs (ordered by free memory). Training automatically uses distributed data parallel for multiple usable GPUs, a single GPU when only one is available, or CPU when CUDA is unavailable. It does not stop other GPU jobs.
Set `VITS_GPU=cpu`, `0`, `1`, or `0,1` in `deploy.env` to override, then restart. An existing `CUDA_VISIBLE_DEVICES` is respected. Batch size is per GPU; global batch size is multiplied by the number of GPUs. Checkpoints can resume across CPU, single GPU and multiple GPUs without changing the export workflow. CUDA out-of-memory during training is reported; it does not silently restart or discard progress.

To deploy on a new host, extract the source bundle to the desired directory,
run `install.sh`, then `start.sh`. An unpacked Conda environment is not assumed
to be relocatable: the installer recreates its prefix for the new location.
Build an installer-ready source archive with:

```bash
python scripts/package_release.py
python scripts/package_release.py --with-data
```

The first installation currently requires network access. A fully offline
installer additionally needs a curated package/wheel repository for the target
Linux/Python/CUDA combination; the source archive does not claim to include it.

For multi-GPU NCCL timeouts on this dual RTX 3090 server, `deploy.env` now includes `NCCL_P2P_DISABLE=1`. This uses a compatible transport while retaining both GPUs; it does not alter host drivers or other GPU jobs. Restart after changing this setting. Confirm the transport mode in the training log.

The default base download is now Chinese-only C, using `c_finetune_speaker.json` with `chinese_cleaners` and its matching symbol table. Explicit `{"variant":"CJE"}` remains supported by `/api/base/download`. Use a separate project when changing base families; do not resume CJE checkpoints with C.
