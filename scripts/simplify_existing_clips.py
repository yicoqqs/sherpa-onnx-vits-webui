"""One-off backed-up conversion of existing project transcripts to simplified Chinese."""
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from chinese_text import simplify_project
from worker import save

active=[]
for file in (ROOT/'data/jobs').glob('*/job.json'):
    job=json.loads(file.read_text(encoding='utf-8'))
    if job['status']=='running':
        active.append(job['id'])
if active:
    raise RuntimeError('任务仍在运行，请完成或停止任务后转换：'+','.join(active))
backup=ROOT/'data/backups'/('simplified-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
projects=clips=0
for file in (ROOT/'data/projects').glob('*/project.json'):
    value=json.loads(file.read_text(encoding='utf-8'))
    count=simplify_project(value)
    if count:
        destination=backup/file.parent.name/'project.json'
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,destination)
        save(file,value)
        clips+=count;projects+=1
print(json.dumps(dict(converted_projects=projects,converted_clips=clips,backup=str(backup) if clips else None),ensure_ascii=False))
