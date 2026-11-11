"""Create a separate Chinese-base project from existing corrected recordings."""
from pathlib import Path
import copy
import json
import shutil
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import worker

def main():
    source_id=sys.argv[1];source=worker.DATA/'projects'/source_id
    source.resolve().relative_to((worker.DATA/'projects').resolve())
    if not source.is_dir():raise ValueError('Source project does not exist')
    existing=[p.parent for p in (worker.DATA/'projects').glob('*/project.json') if (j:=worker.read(p)).get('cloned_from')==source_id and j.get('base_variant')=='C']
    if existing:
        print(json.dumps({'project_id':existing[0].name,'reused':True}));return
    cfg=worker.read(worker.DATA/'base/c/config.json')
    if cfg['data']['text_cleaners']!=['chinese_cleaners']:raise ValueError('Chinese base config is not ready')
    key=uuid.uuid4().hex[:12];target=worker.DATA/'projects'/key;target.mkdir()
    try:
        for name in ('clips','uploads'):
            if (source/name).exists():shutil.copytree(source/name,target/name)
            else:(target/name).mkdir()
        state=copy.deepcopy(worker.read(source/'project.json'))
        state.update(id=key,name=(state['name']+' · 中文C对比')[:60],created=time.time(),base_model=str((worker.DATA/'base/c').resolve()),base_variant='C',cloned_from=source_id)
        worker.save(target/'project.json',state)
    except BaseException:
        assert target.resolve().parent==(worker.DATA/'projects').resolve()
        shutil.rmtree(target);raise
    print(json.dumps({'project_id':key,'name':state['name'],'clips':len(state['clips'])},ensure_ascii=False))

if __name__=='__main__':main()
