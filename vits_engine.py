"""Automatic CPU, single-GPU and distributed multi-GPU VITS fine tuning, compatible with Python 3.12."""
from __future__ import annotations
import math
import os
import tempfile
from datetime import timedelta
from pathlib import Path
import shutil
import sys
import time

from worker import DATA, ROOT, read, save, register_model, task_progress
from chinese_text import training_text, normalize_training_phones, punctuation_for_symbols
from training_modes import configure_training, validate_resume

VENDOR = ROOT / 'vendor/vits'
sys.path.insert(0, str(VENDOR))


def atomic_torch_save(value, path):
    import torch
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def load_weights(model, path, speaker=False, mode='single', reference=0):
    import torch
    # Local user-supplied checkpoints only; never fetch or deserialize arbitrary remote files.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    weights = checkpoint.get('model', checkpoint)
    own = model.state_dict()
    loaded = {}
    for key, value in own.items():
        if key not in weights:
            raise ValueError(f'底模不匹配，缺少参数：{key}')
        other = weights[key]
        if key == 'emb_g.weight' and speaker and other.shape[1:] == value.shape[1:]:
            if not 0 <= reference < other.shape[0]:
                raise ValueError('底模权重中不存在所选音色')
            if mode == 'single':
                loaded[key] = other[reference:reference+1].clone()
            elif mode == 'add' and value.shape[0] == other.shape[0]+1:
                loaded[key] = torch.cat((other, other[reference:reference+1]), dim=0).clone()
            elif mode == 'modify' and value.shape == other.shape:
                loaded[key] = other.clone()
            else:
                raise ValueError('底模音色数量与权重不匹配')
        elif other.shape == value.shape:
            loaded[key] = other
        else:
            raise ValueError(f'底模配置与权重不匹配：{key} {other.shape} != {value.shape}')
    model.load_state_dict(loaded, strict=True)


def make_models(cfg):
    from vits_fast_fine_tuning.models import SynthesizerTrn, MultiPeriodDiscriminator
    generator = SynthesizerTrn(len(cfg['symbols']), cfg['data']['filter_length']//2+1,
                              cfg['train']['segment_size']//cfg['data']['hop_length'],
                              n_speakers=cfg['data']['n_speakers'], **cfg['model'])
    discriminator = MultiPeriodDiscriminator(cfg['model'].get('use_spectral_norm', False))
    return generator, discriminator


def protect_speakers(generator, target):
    """Keep shared weights and all other embeddings unchanged, including AdamW decay."""
    import torch
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    weight = generator.emb_g.weight
    weight.requires_grad_(True)
    mask = torch.zeros_like(weight)
    mask[target] = 1
    weight.register_hook(lambda gradient: gradient * mask)


def train(job, folder):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    from vits_fast_fine_tuning import commons
    from vits_fast_fine_tuning.text import _clean_text
    from vits_fast_fine_tuning.mel_processing import spectrogram_torch

    options = job['options']
    project = DATA / 'projects' / job['project_id']
    state = read(project / 'project.json')
    base = Path(options['base'])
    base_cfg = read(base / 'config.json')
    if not base_cfg.get('symbols'):
        raise ValueError('底模 config.json 必须包含 symbols 符号表；请使用 VITS-fast-fine-tuning 配套底模')
    cleaners = base_cfg['data']['text_cleaners']
    allowed = {'chinese_cleaners', 'cjke_cleaners2', 'cjke_cleaners', 'cjks_cleaners', 'zh_ja_mixture_cleaners'}
    if not cleaners or any(c not in allowed for c in cleaners):
        raise ValueError(f'暂不支持此底模的文本处理方式：{cleaners}')
    if base_cfg['model'].get('gin_channels', 0) <= 0:
        raise ValueError('请选择支持说话人嵌入的 VITS-fast-fine-tuning 底模（gin_channels > 0）')
    cfg = configure_training(base_cfg, base, state['name'], options)
    training = project / 'training'
    training.mkdir(exist_ok=True)
    if options['resume'] and (training / 'resume.pth').exists():
        validate_resume(read(training / 'config.json'), cfg)
    torch.manual_seed(cfg['train'].get('seed', 1234))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    symbols = {s:i for i,s in enumerate(cfg['symbols'])}
    samples = []
    sample_rate = cfg['data']['sampling_rate']
    hop, n_fft = cfg['data']['hop_length'], cfg['data']['filter_length']
    segment_size = cfg['train']['segment_size']
    print('处理校对文本与训练音频…', flush=True)
    for clip in state['clips']:
        if not clip.get('enabled', True) or not clip['text'].strip():
            continue
        phones = _clean_text(f"[ZH]{training_text(clip['text'])}[ZH]", cleaners)
        phones = normalize_training_phones(phones, symbols)
        unknown = set(phones) - set(symbols)
        if unknown:
            raise ValueError(f'片段 {clip["id"]}（{clip["text"]}）的底模符号表缺少音素 {sorted(unknown)}，请检查底模的 symbols 与 text_cleaners 是否配套')
        ids = [symbols[s] for s in phones]
        if cfg['data'].get('add_blank', True):
            ids = commons.intersperse(ids, 0)
        audio, sr = sf.read(project / 'clips' / f"{clip['id']}.wav", dtype='float32')
        if sr != sample_rate:
            divisor = math.gcd(sr, sample_rate)
            audio = resample_poly(audio, sample_rate//divisor, sr//divisor)
        if len(audio) < segment_size + n_fft or len(ids) >= len(audio)//hop:
            print(f'跳过过短或文本过长片段：{clip["text"]}', flush=True)
            continue
        wave = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        spec = spectrogram_torch(wave, n_fft, sample_rate, hop, cfg['data']['win_length']).squeeze(0)
        samples.append((torch.tensor(ids, dtype=torch.long), spec, wave))
    if len(samples)<2:
        raise ValueError('有效训练片段不足 2 个，请补充至少 1 秒以上的清晰录音')
    # Hold one clip out. It never contributes to optimizer updates.
    validation = samples[-1:]
    samples = samples[:-1]
    save(training / 'dataset.json', {'training_clips':len(samples), 'validation_clips':len(validation), 'created':time.time()})

    devices = training_devices(torch)
    print('训练设备：'+(', '.join(f'GPU {i} ({torch.cuda.get_device_name(i)})' for i in devices) if devices else 'CPU'), flush=True)
    world_size = max(1, len(devices))
    if world_size > 1:
        print('多卡通信模式：'+('兼容模式（禁用直接 P2P）' if os.environ.get('NCCL_P2P_DISABLE') == '1' else '自动选择'),flush=True)
    print(f'每卡批次 {options["batch_size"]}，全局批次 {options["batch_size"]*world_size}；'+('多卡同步训练。' if world_size>1 else '单进程训练。'),flush=True)
    if world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError('当前 PyTorch 不支持分布式训练，请安装支持 distributed 的版本')
        torch.multiprocessing.set_sharing_strategy('file_system')
        # Each invocation gets a fresh, initially absent rendezvous file.
        with tempfile.TemporaryDirectory(prefix='distributed-',dir=folder) as temporary:
            rendezvous=(Path(temporary)/'rendezvous').resolve().as_uri()
            torch.multiprocessing.spawn(_train_rank, args=(devices,job,project,cfg,samples,validation,rendezvous), nprocs=world_size, join=True)
    else:
        _train_rank(0,devices,job,project,cfg,samples,validation,None)


def training_devices(torch):
    """Probe visible CUDA devices; unavailable CUDA falls back to CPU."""
    if not torch.cuda.is_available():
        return []
    usable=[]
    for index in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(index):
                probe=torch.ones(1,device=f'cuda:{index}')
                torch.cuda.synchronize(index)
                del probe
            usable.append(index)
        except RuntimeError as error:
            print(f'GPU {index} 无法使用，跳过：{error}',flush=True)
    return usable


def _train_rank(rank,devices,job,project,cfg,samples,validation,rendezvous):
    import torch
    import faulthandler
    trace_file = None
    if os.environ.get('VITS_DISTRIBUTED_DEBUG') == '1':
        trace_file = (Path(project)/'training'/f'rank-{rank}-trace.log').open('w')
        faulthandler.enable(file=trace_file)
        faulthandler.dump_traceback_later(60, repeat=True, file=trace_file)
    distributed=len(devices)>1
    device=torch.device(f'cuda:{devices[rank]}' if devices else 'cpu')
    if devices: torch.cuda.set_device(device)
    if distributed:
        backend='nccl' if torch.distributed.is_nccl_available() else 'gloo'
        torch.distributed.init_process_group(backend,init_method=rendezvous,rank=rank,world_size=len(devices),timeout=timedelta(minutes=5))
    try:
        _run_training(rank,devices,job,project,cfg,samples,validation,device)
    except BaseException:
        import traceback
        print(f'训练进程 {rank} 原始异常：',flush=True)
        traceback.print_exc()
        raise
    finally:
        if trace_file:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()
            trace_file.close()
        if distributed and torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


def wait_training_resume(job, rank, distributed, device):
    """All ranks pause together at a saved epoch boundary."""
    import torch
    folder = DATA / 'jobs' / job['id']
    control = folder / 'training-control.json'
    marker = folder / 'training-paused.json'
    announced = False
    while True:
        paused = bool(read(control).get('pause')) if rank == 0 and control.exists() else False
        if distributed:
            flag = torch.tensor([int(paused)], device=device)
            torch.distributed.broadcast(flag, src=0)
            paused = bool(flag.item())
        if not paused:
            if rank == 0:
                marker.unlink(missing_ok=True)
                if announced: print('训练已恢复，继续下一轮。', flush=True)
            return
        if rank == 0 and not announced:
            save(marker, {'paused': True})
            print('训练已暂停，当前轮检查点已保存。等待恢复训练…', flush=True)
        announced = True
        time.sleep(1)


def _run_training(rank,devices,job,project,cfg,samples,validation,device):
    import torch
    import torch.nn.functional as F
    from vits_fast_fine_tuning import commons
    from vits_fast_fine_tuning.mel_processing import spec_to_mel_torch, mel_spectrogram_torch
    from vits_fast_fine_tuning.losses import discriminator_loss, generator_loss, feature_loss, kl_loss
    options=job['options']; project=Path(project); training=project/'training'; base=Path(options['base'])
    distributed=len(devices)>1; world_size=max(1,len(devices))
    sample_rate=cfg['data']['sampling_rate']; hop=cfg['data']['hop_length']; n_fft=cfg['data']['filter_length']; segment_size=cfg['train']['segment_size']
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(cfg['train'].get('seed',1234))
    generator, discriminator = make_models(cfg)
    generator.to(device); discriminator.to(device)
    lr = options['learning_rate']
    tuning = cfg['finetune']
    protected = tuning['protect_existing']
    if protected:
        protect_speakers(generator, tuning['target_speaker'])
        print('保护原有音色：冻结共享参数，仅训练目标音色嵌入。音色拟合能力会受限。', flush=True)
    optimizer_g = torch.optim.AdamW([p for p in generator.parameters() if p.requires_grad], lr=lr,
                                   weight_decay=0 if protected else .01,
                                   betas=tuple(cfg['train'].get('betas',[.8,.99])), eps=cfg['train'].get('eps',1e-9))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=lr, betas=tuple(cfg['train'].get('betas',[.8,.99])), eps=cfg['train'].get('eps',1e-9))
    epoch_start, steps = 0, 0
    if options['resume'] and (training / 'resume.pth').exists():
        checkpoint = torch.load(training / 'resume.pth', map_location='cpu', weights_only=False)
        generator.load_state_dict(checkpoint['generator']); discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g']); optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        epoch_start, steps = checkpoint['epoch'], checkpoint['steps']
        del checkpoint
        for optim in (optimizer_g,optimizer_d):
            for group in optim.param_groups:
                group['lr'] = lr
        print(f'继续上次训练：已完成 {epoch_start} 轮 / {steps} 步', flush=True)
    else:
        load_weights(generator, base / 'G_0.pth', speaker=True, mode=tuning['mode'], reference=tuning['reference_speaker'])
        load_weights(discriminator, base / 'D_0.pth')
        print(f"已加载底模。训练模式 {tuning['mode']}，参考音色 {tuning['reference_speaker']}，目标音色 {tuning['target_speaker']}，输出 {cfg['data']['n_speakers']} 种音色。", flush=True)

    raw_g,raw_d=generator,discriminator
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        generator=DistributedDataParallel(raw_g,device_ids=[device.index],find_unused_parameters=True)
        discriminator=DistributedDataParallel(raw_d,device_ids=[device.index])
    torch.manual_seed(cfg['train'].get('seed',1234)+rank)
    print(f'训练进程 {rank+1}/{world_size} 已就绪：{device}',flush=True)
    def collate(items):
        max_text = max(x[0].shape[0] for x in items)
        max_spec = max(x[1].shape[1] for x in items)
        max_wave = max(x[2].shape[1] for x in items)
        x = torch.zeros(len(items),max_text,dtype=torch.long)
        spec = torch.zeros(len(items),n_fft//2+1,max_spec)
        y = torch.zeros(len(items),1,max_wave)
        xl, sl = [], []
        for i,(text,s,wave) in enumerate(items):
            x[i,:len(text)] = text; spec[i,:,:s.shape[1]]=s; y[i,:,:wave.shape[1]]=wave
            xl.append(len(text)); sl.append(s.shape[1])
        return [v.to(device) for v in (x,torch.tensor(xl),spec,torch.tensor(sl),y,torch.full((len(items),),tuning['target_speaker'],dtype=torch.long))]

    sampler = torch.utils.data.distributed.DistributedSampler(samples, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg['train'].get('seed',1234)) if distributed else None
    loader = torch.utils.data.DataLoader(samples, batch_size=options['batch_size'], shuffle=sampler is None, sampler=sampler, num_workers=0, collate_fn=collate)
    # Validation uses a fixed seed and a held-out recording, never optimizer updates.
    from training_validation import evaluate, record_validation
    validation_state = training/'validation.json'
    import hashlib
    fingerprint = hashlib.sha256()
    for item in validation:
        for tensor in item:
            fingerprint.update(tensor.numpy().tobytes())
    validation_key = fingerprint.hexdigest()
    previous = read(validation_state) if options['resume'] and validation_state.exists() else {}
    if previous.get('dataset') != validation_key:
        previous = {}
    def validate_epoch(epoch):
        nonlocal previous
        score = evaluate(raw_g, validation, cfg, device, tuning['target_speaker'])
        previous, improved = record_validation(previous, score, epoch, validation_key)
        if improved:
            atomic_torch_save(dict(model=raw_g.state_dict(), iteration=epoch,
                                   learning_rate=lr, optimizer=None), training/'G_best.pth')
        save(validation_state, previous)
        with (training/'validation-history.jsonl').open('a', encoding='utf-8') as history:
            import json
            history.write(json.dumps(previous['latest'], ensure_ascii=False)+'\n')
        print(f"[验证] 第 {epoch} 轮 | 独立原声 {len(validation)} 条 | Mel 平均绝对误差 {score:.6f}（越低越好） | "
              f"相对本次基线改善 {previous['latest']['improvement_percent']:+.2f}% | "
              f"历史最佳 {previous['best']:.6f}（第 {previous['best_epoch']} 轮） | "
              + ('有进步，已保存 G_best.pth' if improved else '未超过最佳，保留原最佳权重'), flush=True)
        print('[验证说明] 此为固定随机种子、原声条件重建的声学误差；改善比例不是音色相似度，实际朗读仍需试听。', flush=True)
    if rank == 0: validate_epoch(epoch_start)
    if distributed: torch.distributed.barrier()

    total_epochs = epoch_start + options['epochs']
    if rank == 0: print(f'{len(samples)} 个训练片段，1 个留出片段；本次训练 {options["epochs"]} 轮，每轮 {len(loader)} 步。', flush=True)
    for epoch in range(epoch_start+1, total_epochs+1):
        if sampler is not None: sampler.set_epoch(epoch)
        generator.train(); discriminator.train()
        for batch_idx,(x,xl,spec,sl,y,sid) in enumerate(loader):
            started = time.time()
            y_hat,l_length,_,ids_slice,_,z_mask,(_,z_p,m_p,logs_p,_,logs_q) = generator(x,xl,spec,sl,sid)
            mel = spec_to_mel_torch(spec,n_fft,cfg['data']['n_mel_channels'],sample_rate,cfg['data']['mel_fmin'],cfg['data'].get('mel_fmax'))
            target_mel = commons.slice_segments(mel,ids_slice,segment_size//hop)
            generated_mel = mel_spectrogram_torch(y_hat.squeeze(1),n_fft,cfg['data']['n_mel_channels'],sample_rate,hop,cfg['data']['win_length'],cfg['data']['mel_fmin'],cfg['data'].get('mel_fmax'))
            real = commons.slice_segments(y,ids_slice*hop,segment_size)
            optimizer_d.zero_grad(set_to_none=True)
            real_scores,fake_scores,_,_ = discriminator(real,y_hat.detach())
            loss_d,_,_ = discriminator_loss(real_scores,fake_scores)
            if not torch.isfinite(loss_d):
                raise ValueError('判别器损失非有限值，请降低学习率并检查录音')
            loss_d.backward(); torch.nn.utils.clip_grad_norm_(discriminator.parameters(),100); optimizer_d.step()
            optimizer_g.zero_grad(set_to_none=True)
            for parameter in raw_d.parameters(): parameter.requires_grad_(False)
            real_scores,fake_scores,real_maps,fake_maps = raw_d(real,y_hat)
            loss_g,_ = generator_loss(fake_scores)
            loss_mel = F.l1_loss(target_mel,generated_mel)*cfg['train'].get('c_mel',45)
            total = loss_g+feature_loss(real_maps,fake_maps)+loss_mel+torch.sum(l_length)+kl_loss(z_p,logs_q,m_p,logs_p,z_mask)*cfg['train'].get('c_kl',1)
            if not torch.isfinite(total):
                raise ValueError('生成器损失非有限值，请降低学习率并检查底模')
            total.backward(); torch.nn.utils.clip_grad_norm_(generator.parameters(),100); optimizer_g.step()
            for parameter in raw_d.parameters(): parameter.requires_grad_(True)
            steps+=1
            if rank == 0: print(f'Epoch {epoch}/{total_epochs} · Step {batch_idx+1}/{len(loader)} · G {total.item():.3f} · D {loss_d.item():.3f} · Mel {loss_mel.item():.3f} · {time.time()-started:.1f}s',flush=True)
        decay=cfg['train'].get('lr_decay',.999875)
        for optim in (optimizer_g,optimizer_d):
            for group in optim.param_groups:
                group['lr']*=decay
        if rank == 0: validate_epoch(epoch)
        if rank == 0:
            atomic_torch_save(dict(generator=raw_g.state_dict(),discriminator=raw_d.state_dict(),
                                   optimizer_g=optimizer_g.state_dict(),optimizer_d=optimizer_d.state_dict(),epoch=epoch,steps=steps), training/'resume.pth')
            atomic_torch_save(dict(model=raw_g.state_dict(),iteration=epoch,learning_rate=lr,optimizer=None),training/'G_latest.pth')
            atomic_torch_save(dict(model=raw_d.state_dict(),iteration=epoch,learning_rate=lr,optimizer=None),training/'D_latest.pth')
            save(training/'config.json', cfg)
            print(f'检查点已保存：第 {epoch} 轮。',flush=True)
        if distributed: torch.distributed.barrier()
        if epoch < total_epochs:
            wait_training_resume(job, rank, distributed, device)
    if rank == 0: print('训练完成。请到「试听与导出」转换 ONNX，然后输入文本试听。',flush=True)


def export(job, folder):
    project = DATA / 'projects' / job['project_id']
    training = project / 'training'
    weights = training/'G_best.pth' if (training/'G_best.pth').exists() else training/'G_latest.pth'
    print('导出权重：'+weights.name, flush=True)
    _export(job, folder, read(training/'config.json'), weights,
            (job['options'].get('name') or '').strip() or read(project/'project.json')['name'], job['project_id'])


def export_base(job, folder):
    from training_modes import base_speaker_count
    from speaker_catalog import model_speakers
    base = Path(job['options']['base']).expanduser()
    cfg = read(base/'config.json')
    cfg['data']['n_speakers'] = base_speaker_count(base, cfg)
    cfg['speakers'] = {str(s['id']): s['name'] or f"音色 {s['id']}"
                       for s in model_speakers(base, {'model': 'G_0.pth', 'n_speakers': cfg['data']['n_speakers']})}
    _export(job, folder, cfg, base/'G_0.pth', f'{base.name} · 底模', None)


def _export(job, folder, cfg, weights, name, origin):
    task_progress('loading', 0)
    import torch
    import onnx
    from vits_fast_fine_tuning.models import SynthesizerTrn
    from vits_fast_fine_tuning import utils
    from export_vits_fast_fine_tuning_onnx import OnnxModel, generate_tokens, generate_lexicon, add_meta_data
    save(folder/'export-config.json', cfg)
    hps = utils.get_hparams_from_file(str(folder/'export-config.json'))
    torch.set_num_threads(4)
    model = SynthesizerTrn(len(cfg['symbols']),cfg['data']['filter_length']//2+1,
                           cfg['train']['segment_size']//cfg['data']['hop_length'],n_speakers=cfg['data']['n_speakers'],**cfg['model'])
    load_weights(model, weights)
    model.eval()
    output = folder/'export'
    output.mkdir()
    task_progress('lexicon', 10)
    print('生成符号表和完整中文发音词典…',flush=True)
    generate_tokens(hps,str(output)); generate_lexicon(hps,str(output))
    # Use valid symbol IDs for tracing. No dependency on English/Japanese phonemizers.
    x=torch.tensor([[0, min(10,len(cfg['symbols'])-1),0,min(20,len(cfg['symbols'])-1),0]],dtype=torch.long)
    inputs=(x,torch.tensor([x.shape[1]]),torch.tensor([.667]),torch.tensor([1.]),torch.tensor([.8]),torch.tensor([0]))
    print('导出 ONNX（传统 TorchScript 导出器）…',flush=True)
    task_progress('onnx', 30)
    with torch.no_grad():
        torch.onnx.export(OnnxModel(model),inputs,str(output/'model.onnx'),dynamo=False,opset_version=17,
                          input_names=['x','x_length','noise_scale','length_scale','noise_scale_w','sid'],output_names=['y'],
                          dynamic_axes={'x':{0:'N',1:'L'},'x_length':{0:'N'},'y':{0:'N',2:'L'}})
    add_meta_data(str(output/'model.onnx'),dict(model_type='vits',language='Chinese',comment='VITS Studio',jieba=1,
                   add_blank=int(cfg['data'].get('add_blank',True)),n_speakers=cfg['data']['n_speakers'],
                   sample_rate=cfg['data']['sampling_rate'],punctuation=punctuation_for_symbols(cfg['symbols'])))
    onnx.checker.check_model(str(output/'model.onnx'))
    task_progress('resources', 60)
    resources=VENDOR/'source-for-output'
    if not (resources/'dict/jieba.dict.utf8').exists():
        raise ValueError('导出资源缺失，请还原项目内置的 vendor/vits/source-for-output 中文词典资源')
    shutil.copytree(resources,output,dirs_exist_ok=True)
    from speaker_catalog import normalize_names
    save(output/'speakers.json', {str(k): v for k,v in normalize_names(cfg['speakers']).items()})
    if job['options'].get('quantize'):
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print('生成 INT8 模型…',flush=True)
        task_progress('quantizing', 70)
        quantize_dynamic(str(output/'model.onnx'),str(output/'model.int8.onnx'),weight_type=QuantType.QUInt8)
    import numpy as np
    import onnxruntime as ort
    task_progress('validating', 75)
    session=ort.InferenceSession(str(output/'model.onnx'),providers=['CPUExecutionProvider'])
    for length in (3,9,31):
        print(f'验证 ONNX 动态文本长度：{length}',flush=True)
        values=dict(x=np.ones((1,length),dtype=np.int64),x_length=np.array([length],dtype=np.int64),
                    noise_scale=np.array([.667],dtype=np.float32),length_scale=np.array([1.],dtype=np.float32),
                    noise_scale_w=np.array([.8],dtype=np.float32),sid=np.array([0],dtype=np.int64))
        audio=session.run(None,values)[0]
        if audio.size==0 or not np.isfinite(audio).all():
            raise ValueError('ONNX 动态输入推理检查失败')
    for sid in sorted({cfg.get('finetune', {}).get('target_speaker', 0), cfg['data']['n_speakers']-1} - {0}):
        print(f'验证 ONNX 音色编号：{sid}', flush=True)
        values['sid'] = np.array([sid], dtype=np.int64)
        audio = session.run(None, values)[0]
        if audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError(f'ONNX 音色 {sid} 推理检查失败')
    result=register_model(output,name,origin=origin,progress=task_progress)
    save(folder/'result.json', {'model_id':result['id']})
    task_progress('completed', 100, model_id=result['id'])
