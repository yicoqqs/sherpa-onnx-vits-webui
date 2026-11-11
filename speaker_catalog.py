"""Read speaker labels from model sidecars without executing model-provided code."""
import json
from pathlib import Path


def normalize_names(value):
    result = {}
    if isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, str) and item.strip():
                result[index] = item.strip()
            elif isinstance(item, dict):
                try:
                    result[int(item['id'])] = str(item.get('name') or '').strip()
                except (KeyError, ValueError, TypeError):
                    continue
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, bool):
                continue
            if isinstance(item, int) or (isinstance(item, str) and item.isdigit() and not str(key).isdigit()):
                result[int(item)] = str(key).strip()  # name -> ID (VITS / Piper)
            elif str(key).isdigit() and isinstance(item, str):
                result[int(key)] = item.strip()  # ID -> name
    return {key:name for key,name in result.items() if key >= 0 and name}


def model_speakers(folder, info):
    try:
        count = max(1, min(10001, int(info.get('n_speakers') or 1)))
    except (ValueError, TypeError):
        count = 1
    names = normalize_names(info.get('speaker_names', {}))
    model = folder / Path(str(info['model']).replace('\\', '/'))
    resources = model.parent
    candidates = [resources/'speakers.json', resources/'speaker_names.json',
                  Path(str(model)+'.json'), resources/'config.json']
    for file in candidates:
        if not file.is_file() or file.stat().st_size > 2 * 1024**2:
            continue
        try:
            value = json.loads(file.read_text(encoding='utf-8-sig'))
        except (ValueError, OSError):
            continue
        mappings = []
        if isinstance(value, dict):
            for key in ('speakers', 'speaker_names', 'spk2id', 'speaker_id_map'):
                if key in value:
                    mappings.append(value[key])
            data = value.get('data', {})
            if isinstance(data, dict):
                mappings.extend(data[key] for key in ('speakers','spk2id') if key in data)
        if not mappings and (file.name in ('speakers.json', 'speaker_names.json') or isinstance(value, list)):
            mappings = [value]
        for mapping in mappings:
            for speaker_id, name in normalize_names(mapping).items():
                names.setdefault(speaker_id, name)
    return [dict(id=index, name=names.get(index, '')) for index in range(count)]
