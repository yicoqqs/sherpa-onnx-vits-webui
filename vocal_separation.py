"""Separate vocals before downmixing; original uploads are never modified."""
from pathlib import Path
import subprocess
import os


def separate_vocals(source, destination, work, root):
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    torch.set_num_threads(min(4, torch.get_num_threads()))
    cache = Path(root)/'cache/torch'
    cache.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache))
    print('加载人声分离模型 htdemucs，首次使用需下载，后续复用本地缓存。', flush=True)
    model = get_model('htdemucs')
    model.eval()
    decoded = Path(work)/'separation-input.wav'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-vn',
                    '-ac',str(model.audio_channels),'-ar',str(model.samplerate),str(decoded)], check=True)
    audio, rate = sf.read(decoded, dtype='float32', always_2d=True)
    wave = torch.from_numpy(audio.T.copy())
    reference = wave.mean(0)
    mean, std = reference.mean(), reference.std()
    if not torch.isfinite(std) or std < 1e-8:
        raise ValueError('素材静音或音量过低，无法分离人声')
    normalized = (wave-mean)/std
    device = 'cpu'
    if torch.cuda.is_available() and os.environ.get('VITS_GPU') != 'cpu':
        for index in range(torch.cuda.device_count()):
            try:
                probe = torch.ones(1, device=f'cuda:{index}')
                torch.cuda.synchronize(index)
                del probe
                device = f'cuda:{index}'
                break
            except RuntimeError as error:
                print(f'人声分离 GPU {index} 不可用：{error}，尝试其他设备。', flush=True)
    print(f'人声分离设备：{device}；模型 htdemucs。', flush=True)
    with torch.no_grad():
        stems = apply_model(model, normalized[None], device=device, shifts=1,
                            split=True, overlap=.25, progress=True, num_workers=0)[0]
    vocals = (stems[model.sources.index('vocals')].cpu()*std+mean).numpy().T
    if not np.isfinite(vocals).all():
        raise ValueError('人声分离产生无效音频')
    sf.write(destination, vocals, rate, subtype='FLOAT')
    del stems, model
    if device == 'cuda':
        torch.cuda.empty_cache()
