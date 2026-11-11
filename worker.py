"""One isolated process per expensive task; stdout is the persistent task log."""
from __future__ import annotations
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
import hashlib
import urllib.request
from chinese_text import to_simplified

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def ffmpeg(source, destination, rate=16000):
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-vn', '-ac', '1', '-ar', str(rate), str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        if 'does not contain any stream' in result.stderr or 'matches no streams' in result.stderr:
            raise ValueError('这条媒体直链没有音轨，可能只包含画面。请提供带声音的完整视频，或抓取对应的音频链接。')
        raise ValueError('录音解码失败：' + result.stderr[-2000:])


def prepare_progress(stage, *, percent=None, **details):
    # Stored with the job log, so old running servers can expose progress without restart.
    print('@@VITS_PROGRESS@@' + json.dumps(dict(stage=stage, percent=percent, **details), ensure_ascii=False), flush=True)


def task_progress(stage, percent=None, **details):
    print('@@TASK_PROGRESS@@' + json.dumps(dict(stage=stage, percent=percent, **details), ensure_ascii=False), flush=True)


def media_progress(stage, **details):
    print('@@MEDIA_PROGRESS@@' + json.dumps(dict(stage=stage, **details), ensure_ascii=False), flush=True)


def import_video(job, folder):
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
    import soundfile as sf
    from video_source import normalize_video_url
    url = normalize_video_url(job['options']['url'])
    direct = (urlsplit(url).hostname or '').endswith('.douyinvod.com') or urlsplit(url).hostname == 'douyinvod.com'
    bilibili = urlsplit(url).hostname in ('www.bilibili.com', 'b23.tv')
    platform = '哔哩哔哩' if bilibili else '抖音'
    project = DATA / 'projects' / job['project_id']
    limit = 1024**3
    last_report = 0
    media_progress('resolving')
    print('正在读取抖音媒体直链…' if direct else f'正在解析{platform}单个视频链接…', flush=True)

    def hook(event):
        nonlocal last_report
        downloaded = event.get('downloaded_bytes', 0)
        total = event.get('total_bytes') or event.get('total_bytes_estimate')
        if downloaded > limit or (total and total > limit):
            raise ValueError('视频素材超过 1 GB，请选择较短的视频或上传裁剪后的本地素材')
        if time.monotonic() - last_report >= 1 or event['status'] == 'finished':
            media_progress('downloading', percent=min(100, int(downloaded/total*100)) if total else None,
                           downloaded_bytes=downloaded, total_bytes=total)
            last_report = time.monotonic()

    class Logger:
        def debug(self, message): pass
        def warning(self, message): print('视频下载提示：' + message, flush=True)
        def error(self, message): pass  # Report a concise actionable exception below.

    options = dict(format='bestaudio/best[acodec!=none]/best' if bilibili else 'best', outtmpl=str(folder/'source.%(ext)s'), noplaylist=True,
                   playlistend=1, max_filesize=limit, socket_timeout=30, retries=2,
                   extractor_retries=2, progress_hooks=[hook], logger=Logger(),
                   quiet=True, no_warnings=False, restrictfilenames=True, continuedl=True,
                   http_headers={'Referer':'https://www.bilibili.com/' if bilibili else 'https://www.douyin.com/', 'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36'})
    if direct or bilibili:
        options['proxy'] = ''  # Domestic media CDNs may reject an overseas system proxy.
    cookie_copy = folder / 'session-cookies.txt'
    cookie_path = job['options'].get('cookies_path')
    if cookie_path:
        shutil.copyfile(cookie_path, cookie_copy)
        options['cookiefile'] = str(cookie_copy)
    try:
        try:
            with YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
                if not info or info.get('_type') in ('playlist', 'multi_video'):
                    raise ValueError('请使用单个视频链接，不支持主页、直播或视频合集')
                source = Path(downloader.prepare_filename(info))
        except DownloadError as error:
            reason = str(error)
            if direct:
                raise ValueError('媒体直链下载失败，可能是网络连接问题、地址已过期或受到访问限制。请重新播放视频后复制最新完整链接，保留 ? 后的全部参数。') from None
            if any(word in reason.lower() for word in ('cookie', 'login', 'logged in', '403', '412', 'verify', 'captcha', '-352')):
                raise ValueError(f'{platform}未返回可下载的素材。请在访问验证设置中提供该平台近期导出的 Cookies 文件后重试；若仍失败，可在浏览器下载视频后直接上传。') from None
            raise ValueError('视频下载失败，请检查链接和网络，或下载视频后直接上传。原因：' + reason[-800:]) from None
        if not source.is_file():
            raise ValueError('未下载到视频文件，可能被平台限制，或文件超过 1 GB')
        title = ('抖音直链素材-' + time.strftime('%Y%m%d-%H%M%S')) if direct else str(info.get('title') or info.get('id') or platform+'视频')[:120]
        save(folder/'media.json', {'file':source.name, 'title':title})
        media_progress('extracting', title=title)
        print('正在提取单声道音频：' + title, flush=True)
        destination = folder/'extracted.wav'
        ffmpeg(source, destination)
        audio = sf.info(destination)
        if audio.duration <= 0 or destination.stat().st_size > limit:
            raise ValueError('提取的音频无效或超过 1 GB')
        filename = uuid.uuid4().hex[:12] + '.wav'
        target = project/'uploads'/filename
        shutil.move(str(destination), target)
        try:
            state = read(project/'project.json')
            state.setdefault('uploads', []).append(dict(file=filename, name=title+'.wav', bytes=target.stat().st_size,
                                                       source_url=url, duration=round(audio.duration, 2)))
            save(project/'project.json', state)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        media_progress('completed', percent=100, title=title, duration=round(audio.duration, 2))
        print(f'素材导入完成：{audio.duration:.1f} 秒。可以继续自动切分并转写。', flush=True)
    finally:
        cookie_copy.unlink(missing_ok=True)


def whisper_model_path(size):
    """Use the Hub's resumable cache, while exposing actual per-file byte progress."""
    from huggingface_hub import snapshot_download, hf_hub_download
    from tqdm.auto import tqdm
    repo = 'Systran/faster-whisper-' + size
    files = ['config.json', 'tokenizer.json', 'vocabulary.txt', 'model.bin']
    cache = str(DATA / 'whisper')
    try:
        return snapshot_download(repo, cache_dir=cache, allow_patterns=files, local_files_only=True)
    except (OSError, ValueError):
        pass
    result = None
    for index, filename in enumerate(files, 1):
        prepare_progress('downloading', file=filename, file_index=index, files_total=len(files), clips=0)
        print(f'准备转写模型文件 {index}/{len(files)}：{filename}（支持缓存和断点下载）', flush=True)

        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.update(disable=False, mininterval=1)
                super().__init__(*args, **kwargs)

            def display(self, *args, **kwargs):
                if not hasattr(self, 'n'):
                    return
                total = self.total or 0
                prepare_progress('downloading', percent=min(100, int(self.n / total * 100)) if total else None,
                                 file=filename, file_index=index, files_total=len(files), clips=0,
                                 downloaded_bytes=int(self.n), total_bytes=int(total))

        result = hf_hub_download(repo, filename, cache_dir=cache, tqdm_class=DownloadProgress)
        print(f'模型文件就绪：{filename}', flush=True)
    return str(Path(result).parent)


def prepare(job, folder):
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    import soundfile as sf
    from faster_whisper import WhisperModel
    project = DATA / 'projects' / job['project_id']
    state = read(project / 'project.json')
    staging = folder / 'clips'
    staging.mkdir()
    from preparation import pending_uploads
    incremental = job['options'].get('mode', 'all') == 'new'
    uploads = pending_uploads(state) if incremental else state['uploads']
    if not uploads:
        print('没有需要切分的新增素材。', flush=True)
        return
    file_count = len(uploads)
    prepare_progress('loading', files_total=file_count, clips=0)
    print('加载 Whisper 模型。首次使用需要联网下载，可通过 HF_ENDPOINT 配置镜像。', flush=True)
    model_path = whisper_model_path(job['options']['whisper'])
    prepare_progress('loading', files_total=file_count, clips=0)
    print('模型文件已就绪，正在加载到内存…', flush=True)
    model = WhisperModel(model_path, device='cpu', compute_type='int8')
    clips, recordings = [], []
    for index, upload in enumerate(uploads, 1):
        prepare_progress('decoding', file=upload['name'], file_index=index, files_total=file_count, clips=0)
        converted = folder / f'converted-{index}.wav'
        source = project / 'uploads' / upload['file']
        if job['options'].get('separate_vocals', False):
            from vocal_separation import separate_vocals
            prepare_progress('separating', file=upload['name'], file_index=index, files_total=file_count, clips=0)
            print('先分离人声：' + upload['name'], flush=True)
            separated = folder / f'vocals-{index}.wav'
            separate_vocals(source, separated, folder, ROOT)
            source = separated
        ffmpeg(source, converted)
        recordings.append((upload, converted, sf.info(converted).duration))
    total_seconds = sum(item[2] for item in recordings)
    processed_seconds = 0
    for index, (upload, converted, duration) in enumerate(recordings, 1):
        print(f"正在处理：{upload['name']}", flush=True)
        def report(position):
            done = min(total_seconds, processed_seconds + max(0, min(duration, position)))
            prepare_progress('transcribing', percent=min(99, int(done / max(total_seconds, .001) * 100)),
                             file=upload['name'], file_index=index, files_total=file_count, clips=len(clips),
                             processed_seconds=round(done, 1), total_seconds=round(total_seconds, 1))
        report(0)
        samples, rate = sf.read(converted, dtype='float32')
        # FFmpeg + soundfile already decoded to mono 16 kHz; passing the array also
        # avoids PyAV API differences between faster-whisper and newer PyAV releases.
        segments, _ = model.transcribe(samples, language='zh', vad_filter=True,
                                     word_timestamps=True, beam_size=5, condition_on_previous_text=False)
        for segment in segments:
            # Whisper word boundaries keep text aligned even when a segment exceeds 12 s.
            words = segment.words or []
            groups, group = [], []
            for word in words:
                if group and word.end - group[0].start > 12:
                    groups.append(group)
                    group = []
                group.append(word)
            if group:
                groups.append(group)
            spans = [(g[0].start, g[-1].end, ''.join(w.word for w in g)) for g in groups]
            if not spans:
                spans = [(segment.start, segment.end, segment.text)]
            for start, end, text in spans:
                start, end = max(0, start - .08), min(len(samples) / rate, end + .08)
                original_text = text.strip().replace('|', '，').replace('\n', '')
                text = to_simplified(original_text)
                if not text or end - start < .7 or end - start > 16:
                    continue
                clip_id = uuid.uuid4().hex[:12]
                sf.write(staging / f'{clip_id}.wav', samples[int(start*rate):int(end*rate)], rate, subtype='PCM_16')
                clips.append(dict(id=clip_id, text=text, duration=round(end-start, 3), enabled=True,
                                  source=upload['name'], source_file=upload['file'], original_text=original_text,
                                  vocals_separated=job['options'].get('separate_vocals', False)))
                print(f'片段 {len(clips):03d} · {end-start:.1f}s · {text}', flush=True)
            report(segment.end)
        processed_seconds += duration
        report(0)
    if not clips:
        raise ValueError('未识别到可用语音，请检查录音是否有清晰的单人讲话')
    target = project / 'clips'
    prepare_progress('saving', percent=99, files_total=file_count, clips=len(clips),
                     processed_seconds=round(total_seconds, 1), total_seconds=round(total_seconds, 1))
    # Keep old files until the new manifest is committed, so failed preparation is recoverable.
    target.mkdir(exist_ok=True)
    for clip in clips:
        shutil.move(str(staging / f"{clip['id']}.wav"), target / f"{clip['id']}.wav")
    state['clips'] = state.get('clips', []) + clips if incremental else clips
    previous = {u['file'] for u in state['uploads']} - {u['file'] for u in pending_uploads(state)}
    state['processed_uploads'] = sorted(previous | {u['file'] for u in uploads})
    save(project / 'project.json', state)
    prepare_progress('completed', percent=100, files_total=file_count, clips=len(clips),
                     processed_seconds=round(total_seconds, 1), total_seconds=round(total_seconds, 1))
    print(f'数据准备完成：{len(clips)} 段，共 {sum(c["duration"] for c in clips)/60:.1f} 分钟。请校对文字。', flush=True)


def inspect_model(folder):
    import onnxruntime as ort
    tokens = list(folder.rglob('tokens.txt'))
    for token in tokens:
        for model in sorted(token.parent.glob('*.onnx'), key=lambda p: ('int8' in p.name, 'fp16' in p.name, p.name)):
            session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
            metadata = session.get_modelmeta().custom_metadata_map
            if metadata.get('model_type', '').lower() == 'vits':
                return model, token, metadata
    raise ValueError('没有找到兼容的 VITS 模型：需要同目录的 tokens.txt 和带 VITS 元数据的 ONNX 文件')


def tts_for(folder, model, token):
    import sherpa_onnx
    resources = model.parent
    rules = ','.join(str(resources / f) for f in ('date.fst', 'number.fst', 'phone.fst', 'new_heteronym.fst') if (resources / f).exists())
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=str(model), tokens=str(token),
        lexicon=str(resources / 'lexicon.txt') if (resources / 'lexicon.txt').exists() else '',
        data_dir=str(resources / 'espeak-ng-data') if (resources / 'espeak-ng-data').is_dir() else '',
        dict_dir=str(resources / 'dict') if (resources / 'dict').is_dir() else '')
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=max(1, min(4, os.cpu_count() or 1)), provider='cpu'),
        rule_fsts=rules)
    if not config.validate():
        raise ValueError('sherpa-onnx 模型配置验证失败，请确认词典资源齐全')
    return sherpa_onnx.OfflineTts(config)


def register_model(source, name, origin='import', progress=None):
    if progress: progress('validating_package', 85)
    model, tokens, metadata = inspect_model(source)
    print('验证 sherpa-onnx 能否加载模型…', flush=True)
    engine = tts_for(source, model, tokens)
    speaker_count = max(1, engine.num_speakers)
    del engine
    from speaker_catalog import normalize_names
    speaker_names = {}
    for field in ('speaker_names', 'speakers', 'spk2id', 'speaker_id_map'):
        try:
            speaker_names.update(normalize_names(json.loads(metadata.get(field, '{}'))))
        except (ValueError, TypeError):
            pass
    key = uuid.uuid4().hex[:12]
    folder = DATA / 'models' / key
    resources = folder / 'vits-voice'
    if progress: progress('copying', 90)
    resources.parent.mkdir(parents=True)
    shutil.copytree(model.parent, resources)
    try:
        archive = folder / 'voice.tar.bz2'
        print('压缩语音包…', flush=True)
        if progress: progress('packing', 95)
        with tarfile.open(archive, 'w:bz2') as tar:
            tar.add(resources, arcname='vits-voice')
        value = dict(id=key, name=name, type='VITS / Piper' if (resources / 'espeak-ng-data').exists() else 'VITS',
                     created=time.time(), origin=origin, model=str(Path('vits-voice') / model.name),
                     sample_rate=metadata.get('sample_rate', ''), n_speakers=speaker_count,
                     speaker_names=speaker_names)
        save(folder / 'model.json', value)
    except BaseException:
        shutil.rmtree(folder)
        raise
    print(f'模型已加入模型库：{name} · {key}', flush=True)
    return value


def import_model(job, folder):
    source = Path(job['options']['path'])
    print('读取模型：' + str(source), flush=True)
    if source.is_dir():
        register_model(source, source.name)
        return
    target = folder / 'unpacked'
    target.mkdir()
    total = 0
    with tarfile.open(source, 'r:*') as tar:
        count = 0
        for member in tar:
            count += 1
            if count > 50000:
                raise ValueError('语音包文件过多')
            dest = (target / member.name).resolve()
            if not dest.is_relative_to(target.resolve()) or member.issym() or member.islnk():
                raise ValueError('语音包包含不安全的文件路径或链接')
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                if total > 4 * 1024**3:
                    raise ValueError('语音包解压后不能超过 4 GB')
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, dest.open('wb') as out:
                    shutil.copyfileobj(src, out)
            else:
                raise ValueError('语音包包含不支持的特殊文件')
    register_model(target, source.name.removesuffix('.tar.bz2').removesuffix('.tar.gz'))


def download_base(job, folder):
    manifest = read(ROOT / 'vendor/vits/base-models.json')
    variant = job['options']['variant']
    destination = DATA / 'base' / variant.lower()
    destination.mkdir(parents=True, exist_ok=True)
    if variant not in ('C', 'CJE'):
        raise ValueError('不支持的底模类型')
    print(f'下载 VITS-fast-fine-tuning {variant} 底模，约 331 MB。首次使用需要等待下载完成。', flush=True)
    for filename, info in manifest['models'][variant].items():
        checksum = info['oid'].split(':')[1]
        target = destination / filename
        if target.exists() and target.stat().st_size == int(info['size']):
            with target.open('rb') as existing:
                actual = hashlib.file_digest(existing, 'sha256').hexdigest()
            if actual == checksum:
                print(filename + ' 已存在且校验通过。', flush=True)
                continue
        url = ('https://media.githubusercontent.com/media/ANONEKO777/sherpa-onnx-vits-fine-tuning-webui/'
               + manifest['revision'] + '/training_data/pretrained_models/' + variant + '/' + filename)
        print('下载 ' + filename, flush=True)
        part = target.with_suffix('.part')
        size, last = 0, 0
        digest = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                size += len(chunk); digest.update(chunk); out.write(chunk)
                if time.monotonic() - last > 3:
                    print(f'{filename} · {size / int(info["size"])*100:.1f}% · {size/1024**2:.1f} MB', flush=True)
                    last = time.monotonic()
        if size != int(info['size']) or digest.hexdigest() != checksum:
            part.unlink(missing_ok=True)
            raise ValueError(filename + ' 完整性校验失败，请重新下载')
        part.replace(target)
        print(filename + ' SHA256 校验通过。', flush=True)
    config = 'c_finetune_speaker.json' if variant == 'C' else 'cje_finetune_speaker.json'
    shutil.copyfile(ROOT / 'vendor/vits/training_data/configs' / config, destination / 'config.json')
    save(folder/'result.json', {'base':str(destination.resolve()),'variant':variant})
    print('底模就绪，请在训练页使用目录：' + str(destination.resolve()), flush=True)


def synthesize(job, folder):
    import soundfile as sf
    task_progress('loading', 0)
    options = job['options']
    source = DATA / 'models' / options['model_id']
    info = read(source / 'model.json')
    print('加载模型：' + info['name'], flush=True)
    model = source / Path(str(info['model']).replace('\\', '/'))
    engine = tts_for(source, model, model.parent / 'tokens.txt')
    speakers = max(1, engine.num_speakers)
    if options['speaker'] >= speakers:
        raise ValueError(f'说话人 ID 超出范围，可用范围 0 至 {speakers-1}')
    print('正在合成：' + options['text'], flush=True)
    task_progress('generating', 15)
    last_percent = 15
    def callback(samples, progress):
        nonlocal last_percent
        percent = max(last_percent, min(90, 15 + int(float(progress)*75)))
        if percent > last_percent:
            task_progress('generating', percent)
            last_percent = percent
        return 1
    audio = engine.generate(options['text'], sid=options['speaker'], speed=options['speed'], callback=callback)
    if len(audio.samples) == 0:
        raise ValueError('未生成音频，请检查文字是否属于模型支持的语言')
    task_progress('saving', 95)
    sf.write(folder / 'output.wav', audio.samples, audio.sample_rate, subtype='PCM_16')
    task_progress('completed', 100, duration=round(len(audio.samples)/audio.sample_rate,2))
    print(f'试听已生成：{len(audio.samples)/audio.sample_rate:.2f} 秒 · {audio.sample_rate} Hz', flush=True)


def main():
    folder = Path(sys.argv[1]).resolve()
    job = read(folder / 'job.json')
    print(f"任务开始：{job['kind']}\nPython：{sys.executable}", flush=True)
    if job['kind'] in ('train', 'export', 'export_base'):
        from vits_engine import train, export, export_base
        {'train': train, 'export': export, 'export_base': export_base}[job['kind']](job, folder)
    else:
        {'prepare': prepare, 'import': import_model, 'synthesize': synthesize, 'download_base': download_base, 'video': import_video}[job['kind']](job, folder)
    print('任务完成。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n任务失败：{e}', flush=True)
        traceback.print_exc()
        sys.exit(1)
