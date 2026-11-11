"""Validate export and synthesis progress with the real checkpoint in isolated storage."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def records(output):
    return [json.loads(x.removeprefix('@@TASK_PROGRESS@@')) for x in output.getvalue().splitlines() if x.startswith('@@TASK_PROGRESS@@')]

def main():
    import worker
    import vits_engine
    source=ROOT/'data/projects'/sys.argv[1]
    with tempfile.TemporaryDirectory(prefix='listen-progress-',dir=ROOT/'data/tmp') as temporary:
        folder=Path(temporary);data=folder/'data';project=data/'projects'/source.name;project.mkdir(parents=True)
        (project/'training').symlink_to(source/'training',target_is_directory=True)
        worker.save(project/'project.json',worker.read(source/'project.json'));(data/'models').mkdir()
        export=folder/'export-job';export.mkdir();output=io.StringIO()
        with patch.object(worker,'DATA',data),patch.object(vits_engine,'DATA',data):
            with contextlib.redirect_stdout(output):vits_engine.export({'project_id':source.name,'options':{'quantize':False}},export)
            progress=records(output);assert progress[-1]['percent']==100
            assert [p['percent'] for p in progress]==sorted(p['percent'] for p in progress)
            result=worker.read(export/'result.json');assert result['model_id']==progress[-1]['model_id']
            print('PASS: real ONNX export, package validation, progress stages:',[p['stage'] for p in progress],flush=True)
            synth=folder/'synth-job';synth.mkdir();output=io.StringIO()
            with contextlib.redirect_stdout(output):worker.synthesize({'options':{'model_id':result['model_id'],'speaker':0,'speed':1,'text':'你好，这是进度显示测试。现在正在生成第二句话。模型名称会显示精确到秒的创建时间。'}},synth)
            progress=records(output);assert progress[-1]['percent']==100;assert (synth/'output.wav').stat().st_size>44
            assert any(p['stage']=='generating' and p['percent']>15 for p in progress)
            print('PASS: real sherpa synthesis callback and WAV save, progress:',[(p['stage'],p['percent']) for p in progress],flush=True)
    print('LISTEN PROGRESS SMOKE PASSED',flush=True)

if __name__=='__main__':main()
