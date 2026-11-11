"""Reproduce a training job using an isolated copy of its state and checkpoints."""
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def main():
    import worker
    import vits_engine
    job=worker.read(ROOT/'data/jobs'/sys.argv[1]/'job.json')
    source=ROOT/'data/projects'/job['project_id']
    with tempfile.TemporaryDirectory(prefix='training-reproduce-',dir=ROOT/'data/tmp') as temporary:
        folder=Path(temporary); data=folder/'data'; project=data/'projects'/job['project_id']; project.mkdir(parents=True)
        (project/'clips').symlink_to(source/'clips',target_is_directory=True)
        worker.save(project/'project.json',worker.read(source/'project.json'))
        (project/'training').mkdir()
        for name in ('resume.pth','config.json'):
            os.link(source/'training'/name,project/'training'/name)
        job['options']['epochs']=2
        worker.save(ROOT/'data/logs/reproduce-location.json',{'project':str(project)})
        print('Isolated project:',project,flush=True)
        try:
            with patch.object(vits_engine,'DATA',data): vits_engine.train(job,folder)
        finally:
            for file in (project/'training').glob('rank-*-trace.log'):
                shutil.copyfile(file,ROOT/'data/logs'/('reproduce-'+file.name))
        print('REAL DATA REPRODUCTION PASSED',flush=True)

if __name__=='__main__': main()
