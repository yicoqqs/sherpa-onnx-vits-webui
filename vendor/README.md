# Bundled VITS engine

Source: https://github.com/ANONEKO777/sherpa-onnx-vits-fine-tuning-webui

Revision: `d6de394d1a84391f2b1da98e1a24ba3f67f04f59`.

The upstream MIT license is retained in `vits/LICENSE`. This source itself derives
from Plachtaa/VITS-fast-fine-tuning and the original VITS implementation; retain
upstream attribution when distributing derived work.

Local adaptations:

- Lazy optional language imports so Chinese works without Japanese/Korean/Thai toolchains.
- Numba maximum-path alignment replaces the platform-specific Cython extension.
- Keyword arguments for current librosa and complex STFT for current PyTorch.
- Relative embedding lookup uses gather and masking to preserve dynamic ONNX text lengths.
- `vits_engine.py` implements a single-process CPU/CUDA loop and atomic resume checkpoints.

Chinese FST and jieba resource files under `source-for-output` were copied from the
user's existing official-style `vits-zh-hf-bronya.tar.bz2` package. They are the same
kind of frontend resources required by the upstream exporter. See the bundled
`dict/README.md` for resource information. No pretrained neural weights are bundled.
