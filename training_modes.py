"""Speaker layout and resume compatibility shared by the API and training worker."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

from speaker_catalog import model_speakers


def base_speaker_count(base, cfg):
    count = int(cfg['data']['n_speakers'])
    if count <= 0:
        # Upstream fine-tuning templates set n_speakers=0 even for multi-speaker weights.
        import torch
        path = Path(base)/'G_0.pth'
        try:
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            except RuntimeError:
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        except (OSError, RuntimeError, pickle.UnpicklingError, EOFError) as error:
            raise ValueError(f'无法读取底模生成器权重：{error}') from error
        weights = checkpoint.get('model', checkpoint)
        embedding = weights.get('emb_g.weight')
        if embedding is None or embedding.ndim != 2:
            raise ValueError('底模权重缺少说话人嵌入，无法读取音色数量')
        count = embedding.shape[0]
    if not 1 <= count <= 10001:
        raise ValueError('底模必须包含 1 至 10001 个音色')
    return count


def configure_training(base_cfg, base, name, options):
    mode = options.get('mode', 'single')
    reference = options.get('reference_speaker', 0)
    count = base_speaker_count(base, base_cfg)
    if mode not in ('add', 'modify', 'single'):
        raise ValueError('未知训练模式')
    if not 1 <= count <= 10001 or not 0 <= reference < count:
        raise ValueError(f'参考/目标音色编号超出范围，可用范围 0 至 {count-1}')
    if mode == 'add' and count >= 10001:
        raise ValueError('底模音色数量已达到上限，无法新增音色')
    names = {str(s['id']): s['name'] or f"音色 {s['id']}"
             for s in model_speakers(Path(base), {'model': 'G_0.pth', 'n_speakers': count})}
    target = count if mode == 'add' else reference if mode == 'modify' else 0
    if mode == 'single':
        names = {'0': name}
    else:
        names[str(target)] = name
    cfg = copy.deepcopy(base_cfg)
    cfg['data']['n_speakers'] = 1 if mode == 'single' else count + (mode == 'add')
    # ID -> name supports duplicate display names without losing speakers.
    cfg['speakers'] = names
    cfg['finetune'] = dict(mode=mode, reference_speaker=reference, target_speaker=target,
                           protect_existing=mode != 'single' and options.get('protect_existing', True),
                           base=str(Path(base).resolve()))
    cfg['train']['learning_rate'] = options['learning_rate']
    cfg['train']['batch_size'] = options['batch_size']
    return cfg


def validate_resume(old, cfg):
    def signature(value):
        return hashlib.sha256(json.dumps([value['model'], value['symbols'], value['data']],
                                         sort_keys=True).encode()).hexdigest()
    if signature(old) != signature(cfg):
        raise ValueError('当前底模结构与已有检查点不同；请新建音色项目，或取消继续训练')
    previous = old.get('finetune')
    current = cfg['finetune']
    if previous is None:
        compatible = current['mode'] == 'single' and current['reference_speaker'] == 0
    else:
        compatible = previous == current
    if not compatible:
        raise ValueError('训练模式、参考/目标音色、参数保护或底模已改变；请新建音色项目，或取消继续训练')
