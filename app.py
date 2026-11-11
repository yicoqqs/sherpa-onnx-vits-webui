"""Local VITS Studio. Run with the conda py312 interpreter."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from typing import Literal
import urllib.request

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from video_source import normalize_video_url
from preparation import pending_uploads
from speaker_catalog import model_speakers
from training_modes import base_speaker_count, configure_training, validate_resume
from chinese_text import to_simplified, simplify_project

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
PROJECTS = DATA / 'projects'
MODELS = DATA / 'models'
JOBS = DATA / 'jobs'
for directory in (PROJECTS, MODELS, JOBS):
    directory.mkdir(parents=True, exist_ok=True)
@asynccontextmanager
async def lifespan(app):
    recover()
    yield
    for process in list(active.values()):
        if process.poll() is None:
            terminate_job(process)


app = FastAPI(title='VITS Studio', docs_url='/api/docs', lifespan=lifespan)
lock = threading.RLock()
active: dict[str, subprocess.Popen] = {}
MAX_UPLOAD = 1024 * 1024 * 1024


def terminate_job(process):
    """Stop the worker and all of its distributed training subprocesses."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def save(path: Path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def identified(base: Path, key: str):
    if not re.fullmatch(r'[a-f0-9]{12}', key):
        raise HTTPException(400, '无效的资源编号')
    target = base / key
    if not target.exists():
        raise HTTPException(404, '资源不存在')
    return target


def project(key):
    return identified(PROJECTS, key)


def start_job(kind: str, project_id: str | None, options: dict):
    with lock:
        if active:
            raise HTTPException(409, '已有任务运行中，请等待完成或先停止任务')
        key = uuid.uuid4().hex[:12]
        folder = JOBS / key
        folder.mkdir()
        state = dict(id=key, kind=kind, project_id=project_id, status='running',
                     created=time.time(), options=options)
        save(folder / 'job.json', state)
        log = (folder / 'output.log').open('w', encoding='utf-8')
        env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
        for scheme, proxy in urllib.request.getproxies().items():
            if scheme in ('http', 'https'):
                env.setdefault(scheme.upper() + '_PROXY', proxy)
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'worker.py'), str(folder)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=os.name != 'nt')
        active[key] = process

        def monitor():
            code = process.wait()
            log.close()
            (folder / 'session-cookies.txt').unlink(missing_ok=True)
            with lock:
                latest = read(folder / 'job.json')
                if latest['status'] != 'cancelled':
                    latest['status'] = 'completed' if code == 0 else 'failed'
                latest.update(exit_code=code, finished=time.time())
                save(folder / 'job.json', latest)
                active.pop(key, None)
        threading.Thread(target=monitor, daemon=True).start()
        return state


def recover():
    for file in JOBS.glob('*/job.json'):
        (file.parent / 'session-cookies.txt').unlink(missing_ok=True)
        item = read(file)
        if item['status'] == 'running':
            item.update(status='interrupted', finished=time.time())
            save(file, item)


@app.middleware('http')
async def local_only(request, call_next):
    # Prevent other sites from submitting requests to this local file-management service.
    origin = request.headers.get('origin')
    if origin and origin != f'{request.url.scheme}://{request.headers.get("host")}':
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': '仅允许本地页面发起请求'}, status_code=403)
    host = request.headers.get('host', '').split(':')[0]
    allowed_hosts = {'127.0.0.1', 'localhost', 'testserver'}
    allowed_hosts.update(item.strip().lower() for item in os.environ.get('VITS_ALLOWED_HOSTS', '').split(',') if item.strip())
    if host.lower() not in allowed_hosts:
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': '访问地址未在 VITS_ALLOWED_HOSTS 中配置'}, status_code=403)
    response = await call_next(request)
    if request.url.path in ('/', '/index.html', '/app.js', '/style.css'):
        response.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return response


@app.get('/api/environment')
def environment():
    import torch
    modules = ['sherpa_onnx', 'faster_whisper', 'onnx', 'librosa', 'pypinyin', 'cn2an', 'jieba', 'numba']
    return dict(python=sys.version.split()[0], executable=sys.executable,
                torch=torch.__version__, cuda=torch.cuda.is_available(),
                device=' · '.join(f'GPU {i}: {torch.cuda.get_device_name(i)}' for i in range(torch.cuda.device_count())) if torch.cuda.is_available() else 'CPU',
                ffmpeg=bool(shutil.which('ffmpeg')), dependencies={m: bool(importlib.util.find_spec(m)) for m in modules},
                engine=(ROOT / 'vendor/vits/vits_fast_fine_tuning/models.py').exists(),
                base_models=[str(p.parent) for p in (DATA / 'base').glob('*/config.json')],
                model_library=str(MODELS), data_directory=str(DATA))


class NewProject(BaseModel):
    name: str = Field(min_length=1, max_length=60)


@app.post('/api/projects')
def create_project(body: NewProject):
    key = uuid.uuid4().hex[:12]
    folder = PROJECTS / key
    (folder / 'uploads').mkdir(parents=True)
    value = dict(id=key, name=body.name.strip() or '我的声音', created=time.time(), clips=[])
    save(folder / 'project.json', value)
    return value


@app.get('/api/projects')
def projects():
    return sorted([read(p) for p in PROJECTS.glob('*/project.json')], key=lambda x: x['created'], reverse=True)


class BaseDownload(BaseModel):
    variant: Literal['C', 'CJE'] = 'C'


@app.post('/api/base/download')
def download_base(body: BaseDownload = BaseDownload()):
    return start_job('download_base', None, body.model_dump())


@app.get('/api/projects/{key}')
def get_project(key: str):
    folder = project(key)
    state = read(folder / 'project.json')
    config = folder / 'training/config.json'
    if config.exists():
        cfg = read(config)
        state['training_settings'] = cfg.get('finetune', dict(mode='single', reference_speaker=0, protect_existing=False))
    return state


def ensure_idle():
    if active:
        raise HTTPException(409, '任务运行期间不能修改数据，请先停止任务')


def trash_root():
    root = PROJECTS.parent / 'trash'
    root.mkdir(exist_ok=True)
    return root


def move_project_file(source, target, project_folder):
    # Validate every computed move against the intended project and trash directories.
    if not source.resolve().is_relative_to(project_folder.resolve()) or not target.resolve().is_relative_to(trash_root().resolve()):
        raise HTTPException(400, '无效的文件路径')
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        source.rename(target)


@app.delete('/api/projects/{key}')
def delete_project(key: str):
    with lock:
        ensure_idle()
        folder = project(key)
        token = uuid.uuid4().hex[:12]
        trash = trash_root() / token
        trash.mkdir()
        save(trash / 'deletion.json', dict(kind='project', project_id=key, created=time.time()))
        move_project_file(folder, trash / 'project', folder)
        return dict(deleted=True, undo=token)


def delete_project_items(key, *, filename=None, clip_id=None):
    ensure_idle()
    folder = project(key)
    state = read(folder / 'project.json')
    uploads = state.get('uploads', [])
    if filename:
        upload = next((item for item in uploads if item['file'] == filename), None)
        if not upload:
            raise HTTPException(404, '素材不存在')
        same_names = sum(item['name'] == upload['name'] for item in uploads)
        ambiguous = [c for c in state['clips'] if not c.get('source_file') and c.get('source') == upload['name'] and same_names > 1]
        if ambiguous:
            raise HTTPException(409, '旧片段存在重名素材，无法准确关联。请先删除相关片段，或重新切分后再删除素材。')
        clips = [c for c in state['clips'] if c.get('source_file') == filename or (not c.get('source_file') and c.get('source') == upload['name'])]
    else:
        clips = [c for c in state['clips'] if c['id'] == clip_id]
        if not clips:
            raise HTTPException(404, '片段不存在')
        upload = None
    token = uuid.uuid4().hex[:12]
    trash = trash_root() / token
    trash.mkdir()
    removed_ids = {c['id'] for c in clips}
    metadata = dict(kind='items', project_id=key, created=time.time(), clips=clips,
                    clip_order=[c['id'] for c in state['clips']], upload=upload,
                    upload_order=[u['file'] for u in uploads])
    save(trash / 'deletion.json', metadata)
    paths = ([Path('uploads') / filename] if filename else []) + [Path('clips') / (c['id'] + '.wav') for c in clips]
    moved = []
    try:
        for relative in paths:
            source = folder / relative
            if source.exists():
                move_project_file(source, trash / relative, folder)
                moved.append(relative)
        state['clips'] = [c for c in state['clips'] if c['id'] not in removed_ids]
        state['uploads'] = [u for u in uploads if u['file'] != filename]
        save(folder / 'project.json', state)
    except Exception:
        for relative in reversed(moved):
            (trash / relative).rename(folder / relative)
        raise
    return dict(project=state, undo=token, removed_clips=len(clips))


@app.delete('/api/projects/{key}/uploads/{filename}')
def delete_upload(key: str, filename: str):
    with lock:
        return delete_project_items(key, filename=filename)


@app.delete('/api/projects/{key}/clips/{clip_id}')
def delete_clip(key: str, clip_id: str):
    with lock:
        return delete_project_items(key, clip_id=clip_id)


@app.post('/api/trash/{token}/restore')
def restore_deletion(token: str):
    with lock:
        ensure_idle()
        trash = identified(trash_root(), token)
        metadata = read(trash / 'deletion.json')
        if metadata.get('restored'):
            raise HTTPException(409, '该删除记录已撤销')
        if metadata['kind'] == 'model':
            target = MODELS / metadata['model_id']
            source = trash / 'model'
            if target.exists():
                raise HTTPException(409, '同编号语音包已存在，无法覆盖')
            if not source.resolve().is_relative_to(trash.resolve()) or not target.resolve().is_relative_to(MODELS.resolve()):
                raise HTTPException(400, '无效的恢复路径')
            source.rename(target)
            metadata['restored'] = True
            save(trash / 'deletion.json', metadata)
            return read(target / 'model.json')
        key = metadata['project_id']
        target = PROJECTS / key
        if metadata['kind'] == 'project':
            if target.exists():
                raise HTTPException(409, '同编号音色已存在，无法覆盖')
            source = trash / 'project'
            if not source.resolve().is_relative_to(trash.resolve()) or not target.resolve().is_relative_to(PROJECTS.resolve()):
                raise HTTPException(400, '无效的恢复路径')
            source.rename(target)
        else:
            target = project(key)
            state = read(target / 'project.json')
            clips = metadata['clips']
            upload = metadata['upload']
            if any(c['id'] in {x['id'] for x in state['clips']} for c in clips) or (upload and upload['file'] in {u['file'] for u in state.get('uploads', [])}):
                raise HTTPException(409, '相关素材或片段已存在，无法覆盖')
            paths = ([Path('uploads') / upload['file']] if upload else []) + [Path('clips') / (c['id'] + '.wav') for c in clips]
            moved = []
            try:
                for relative in paths:
                    source, destination = trash / relative, target / relative
                    if not source.resolve().is_relative_to(trash.resolve()) or not destination.resolve().is_relative_to(target.resolve()):
                        raise HTTPException(400, '无效的恢复路径')
                    if source.exists():
                        if destination.exists():
                            raise HTTPException(409, '目标文件已存在，无法覆盖')
                        destination.parent.mkdir(exist_ok=True)
                        source.rename(destination)
                        moved.append(relative)
                state['clips'].extend(clips)
                if upload:
                    state.setdefault('uploads', []).append(upload)
                for field, order_field, identity in [('clips','clip_order','id'),('uploads','upload_order','file')]:
                    order = {v:i for i,v in enumerate(metadata[order_field])}
                    state.setdefault(field, []).sort(key=lambda item: order.get(item[identity], len(order)))
                save(target / 'project.json', state)
            except Exception:
                for relative in reversed(moved):
                    (target / relative).rename(trash / relative)
                raise
        metadata['restored'] = True
        save(trash / 'deletion.json', metadata)
        value = read(target / 'project.json')
        if simplify_project(value):
            save(target / 'project.json', value)
        return value


@app.post('/api/projects/{key}/upload')
def upload(key: str, files: list[UploadFile] = File(...)):
    folder = project(key)
    with lock:
        ensure_idle()
        state = read(folder / 'project.json')
        added = []
        try:
            for file in files:
                ext = Path(file.filename or '').suffix.lower()
                if ext not in ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm', '.mp4', '.mkv'):
                    raise HTTPException(400, '支持 WAV、MP3、FLAC、M4A、OGG、WebM、MP4、MKV')
                filename = uuid.uuid4().hex[:12] + ext
                target = folder / 'uploads' / filename
                added.append(target)
                size = 0
                with target.open('wb') as out:
                    while chunk := file.file.read(1024 * 1024):
                        size += len(chunk)
                        if size > MAX_UPLOAD:
                            raise HTTPException(413, '单个文件不能超过 1 GB')
                        out.write(chunk)
                state.setdefault('uploads', []).append(dict(file=filename, name=file.filename, bytes=size))
            save(folder / 'project.json', state)
        except Exception:
            for path in added:
                path.unlink(missing_ok=True)
            raise
    return state


class Prepare(BaseModel):
    separate_vocals: bool = False
    mode: Literal["all", "new"] = "all"
    whisper: str = 'small'
    language: str = 'zh'


class VideoImport(BaseModel):
    url: str = Field(min_length=1, max_length=4000)
    cookies_path: str = Field(default='', max_length=1000)


@app.post('/api/projects/{key}/video')
def video_import(key: str, body: VideoImport):
    project(key)
    try:
        url = normalize_video_url(body.url)
    except ValueError as error:
        raise HTTPException(400, str(error))
    cookies_path = body.cookies_path.strip()
    if cookies_path:
        cookies = Path(cookies_path).expanduser()
        if not cookies.is_file() or cookies.suffix.lower() != '.txt' or cookies.stat().st_size > 5 * 1024**2:
            raise HTTPException(400, 'Cookies 需为本地 Netscape 格式 .txt 文件，大小不超过 5 MB')
        cookies_path = str(cookies.resolve())
    return start_job('video', key, dict(url=url, cookies_path=cookies_path))


@app.post('/api/projects/{key}/prepare')
def prepare(key: str, body: Prepare):
    state = read(project(key) / 'project.json')
    if not state.get('uploads'):
        raise HTTPException(400, '请先上传录音')
    if body.whisper not in ('tiny', 'base', 'small', 'medium') or body.language != 'zh':
        raise HTTPException(400, '当前版本支持中文，可选择 tiny/base/small/medium 转写模型')
    if body.mode == 'new' and not pending_uploads(state):
        raise HTTPException(400, '没有需要切分的新增素材')
    return start_job('prepare', key, body.model_dump())


class Clips(BaseModel):
    clips: list[dict]


@app.put('/api/projects/{key}/clips')
def update_clips(key: str, body: Clips):
    with lock:
        ensure_idle()
        folder = project(key)
        state = read(folder / 'project.json')
        existing = {c['id']: c for c in state['clips']}
        if len(body.clips) != len(existing) or {c.get('id') for c in body.clips} != set(existing):
            raise HTTPException(400, '片段列表已变化，请刷新')
        for c in body.clips:
            original_text = str(c.get('text', '')).strip()
            text = to_simplified(original_text)
            if len(text) > 1000 or any(ch in text for ch in '|\n\r'):
                raise HTTPException(400, '转写文本不能包含换行或 |，且不能超过 1000 字')
            existing[c['id']].update(text=text, enabled=bool(c.get('enabled', True)))
            if original_text != text:
                existing[c['id']].setdefault('original_text', original_text)
        save(folder / 'project.json', state)
        return state


class Training(BaseModel):
    base: str
    epochs: int = Field(default=30, ge=1, le=10000)
    batch_size: int = Field(default=2, ge=1, le=32)
    learning_rate: float = Field(default=0.0001, ge=0.000001, le=0.001)
    resume: bool = True
    mode: Literal['single', 'add', 'modify'] = 'single'
    reference_speaker: int = Field(default=0, ge=0, le=10000)
    protect_existing: bool = True


@app.get('/api/base/speakers')
def base_speakers(base: str):
    folder = Path(base).expanduser()
    try:
        cfg = read(folder / 'config.json')
        count = base_speaker_count(folder, cfg)
        return dict(speakers=model_speakers(folder, {'model': 'G_0.pth', 'n_speakers': count}))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(400, f'无法读取底模音色：{error}') from error


@app.post('/api/projects/{key}/train')
def train(key: str, body: Training):
    state = read(project(key) / 'project.json')
    clips = [c for c in state['clips'] if c.get('enabled', True) and c['text'].strip()]
    if len(clips) < 2:
        raise HTTPException(400, '至少需要两个已校对且启用的录音片段')
    base = Path(body.base).expanduser()
    if not all((base / f).is_file() for f in ('config.json', 'G_0.pth', 'D_0.pth')):
        raise HTTPException(400, '底模目录需包含 config.json、G_0.pth、D_0.pth；ONNX 语音包不能用作训练底模')
    try:
        cfg = configure_training(read(base / 'config.json'), base, state['name'], body.model_dump())
        training = project(key) / 'training'
        if body.resume and (training / 'resume.pth').exists():
            validate_resume(read(training / 'config.json'), cfg)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    return start_job('train', key, body.model_dump())


class Export(BaseModel):
    quantize: bool = False
    name: str = ""


class BaseExport(Export):
    base: str


@app.post('/api/base/export')
def export_base(body: BaseExport):
    base = Path(body.base).expanduser()
    if not all((base/f).is_file() for f in ('config.json', 'G_0.pth')):
        raise HTTPException(400, '底模导出需要 config.json 和 G_0.pth，无需先训练')
    try:
        cfg = read(base/'config.json')
        base_speaker_count(base, cfg)
        if not cfg.get('symbols') or cfg['model'].get('gin_channels', 0) <= 0:
            raise ValueError('请选择带符号表和说话人嵌入的 VITS-fast-fine-tuning 底模')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    return start_job('export_base', None, body.model_dump())


@app.post('/api/projects/{key}/export')
def export(key: str, body: Export):
    folder = project(key)
    if not (folder / 'training/G_latest.pth').exists():
        raise HTTPException(400, '请先完成训练，生成 G_latest.pth')
    return start_job('export', key, body.model_dump())


@app.get('/api/jobs')
def jobs():
    return sorted([read(p) for p in JOBS.glob('*/job.json')], key=lambda x: x['created'], reverse=True)[:30]


@app.get('/api/jobs/{key}')
def job(key: str):
    folder = identified(JOBS, key)
    state = read(folder / 'job.json')
    if (folder / 'result.json').exists():
        state['result'] = read(folder / 'result.json')
    if state['kind'] == 'train' and state['status'] == 'running':
        control = folder / 'training-control.json'
        state['pause_requested'] = control.exists() and bool(read(control).get('pause'))
        state['training_paused'] = (folder / 'training-paused.json').exists()
    path = folder / 'output.log'
    if path.exists():
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 40000))
            state['log'] = f.read().decode('utf-8', errors='replace')
    return state


class TrainingPause(BaseModel):
    pause: bool


@app.post('/api/jobs/{key}/training-pause')
def training_pause(key: str, body: TrainingPause):
    with lock:
        folder = identified(JOBS, key)
        state = read(folder / 'job.json')
        process = active.get(key)
        if state['kind'] != 'train' or state['status'] != 'running' or process is None or process.poll() is not None:
            raise HTTPException(409, '当前没有可暂停或恢复的训练任务')
        save(folder / 'training-control.json', {'pause': body.pause})
    return {'pause_requested': body.pause}


@app.post('/api/jobs/{key}/cancel')
def cancel(key: str):
    folder = identified(JOBS, key)
    with lock:
        process = active.get(key)
        if not process:
            raise HTTPException(409, '任务已经结束')
        state = read(folder / 'job.json')
        state['status'] = 'cancelled'
        save(folder / 'job.json', state)
        terminate_job(process)
    return {'status': 'cancelled'}


class ImportModel(BaseModel):
    path: str


@app.post('/api/models/import')
def import_model(body: ImportModel):
    source = Path(body.path).expanduser()
    if not source.exists():
        raise HTTPException(400, '模型目录或压缩包不存在')
    return start_job('import', None, {'path': str(source.resolve())})


@app.get('/api/models')
def models():
    items = []
    for path in MODELS.glob('*/model.json'):
        item = read(path)
        item['speakers'] = model_speakers(path.parent, item)
        items.append(item)
    return sorted(items, key=lambda x: x['created'], reverse=True)


@app.delete('/api/models/{key}')
def delete_model(key: str):
    with lock:
        ensure_idle()
        folder = identified(MODELS, key)
        token = uuid.uuid4().hex[:12]
        trash = trash_root() / token
        target = trash / 'model'
        if not folder.resolve().is_relative_to(MODELS.resolve()) or not target.resolve().is_relative_to(trash_root().resolve()):
            raise HTTPException(400, '无效的语音包路径')
        trash.mkdir()
        save(trash / 'deletion.json', dict(kind='model', model_id=key, created=time.time()))
        folder.rename(target)
        return dict(deleted=True, undo=token)


class Synthesis(BaseModel):
    model_id: str
    text: str = Field(min_length=1, max_length=1000)
    speaker: int = Field(default=0, ge=0, le=10000)
    speed: float = Field(default=1, ge=0.5, le=2)


@app.post('/api/synthesize')
def synthesize(body: Synthesis):
    identified(MODELS, body.model_id)
    if not body.text.strip():
        raise HTTPException(400, '请输入试听文本')
    return start_job('synthesize', None, body.model_dump())


@app.get('/api/models/{key}/download')
def download(key: str):
    folder = identified(MODELS, key)
    path = folder / 'voice.tar.bz2'
    if not path.exists():
        raise HTTPException(404, '模型包尚未生成')
    return FileResponse(path, filename=f'vits-{key}.tar.bz2', media_type='application/octet-stream')


@app.get('/api/audio/{key}/{clip_id}')
def clip_audio(key: str, clip_id: str):
    folder = project(key)
    clips = read(folder / 'project.json')['clips']
    clip = next((c for c in clips if c['id'] == clip_id), None)
    if not clip:
        raise HTTPException(404, '片段不存在')
    return FileResponse(folder / 'clips' / (clip_id + '.wav'), media_type='audio/wav')


class HistoryDelete(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=500)


@app.get('/api/listen-history')
def listen_history():
    items = []
    for path in JOBS.glob('*/job.json'):
        item = read(path)
        if item.get('kind') != 'synthesize' or item.get('status') != 'completed' or item.get('history_deleted') or not (path.parent / 'output.wav').is_file():
            continue
        model = MODELS / item['options']['model_id'] / 'model.json'
        model_info = read(model) if model.is_file() else {}
        item['model_name'] = model_info.get('name') or item['options']['model_id']
        item['model_created'] = model_info.get('created')
        items.append(item)
    return sorted(items, key=lambda item: item['created'], reverse=True)


@app.post('/api/listen-history/delete')
def delete_listen_history(body: HistoryDelete):
    with lock:
        entries = []
        for key in set(body.ids):
            folder = identified(JOBS, key)
            item = read(folder / 'job.json')
            if item.get('kind') != 'synthesize' or item.get('status') != 'completed':
                raise HTTPException(400, '只能删除已完成的试听')
            entries.append((folder, item))
        for folder, item in entries:
            item['history_deleted'] = True
            save(folder / 'job.json', item)
            (folder / 'output.wav').unlink(missing_ok=True)
    return {'deleted': len(entries)}


@app.get('/api/jobs/{key}/audio')
def generated_audio(key: str):
    folder = identified(JOBS, key)
    if not (folder / 'output.wav').exists():
        raise HTTPException(404, '试听尚未生成')
    return FileResponse(folder / 'output.wav', media_type='audio/wav')


@app.get('/api/jobs/{key}/video')
def downloaded_video(key: str):
    folder = identified(JOBS, key)
    manifest = folder / 'media.json'
    if not manifest.exists():
        raise HTTPException(404, '尚未下载到原素材文件')
    name = read(manifest)['file']
    if Path(name).name != name or not (folder / name).is_file():
        raise HTTPException(404, '原素材文件不存在')
    return FileResponse(folder / name, filename='douyin-video' + Path(name).suffix, media_type='application/octet-stream')


app.mount('/', StaticFiles(directory=ROOT / 'static', html=True), name='ui')

if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default=os.environ.get('VITS_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=7860)
    args = parser.parse_args()
    print(f'VITS Studio: http://{args.host}:{args.port}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
