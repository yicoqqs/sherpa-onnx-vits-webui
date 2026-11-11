"""Exercise the real URL -> downloaded media -> training WAV path in isolated temporary data."""
import argparse
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import soundfile as sf
import worker

parser=argparse.ArgumentParser()
parser.add_argument('url')
args=parser.parse_args()
(ROOT/'data/tmp').mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='video-import-check-',dir=ROOT/'data/tmp') as temporary:
    data=Path(temporary)
    project=data/'projects/000000000001'
    (project/'uploads').mkdir(parents=True)
    folder=data/'job';folder.mkdir()
    worker.save(project/'project.json',dict(id=project.name,name='URL import check',clips=[],uploads=[]))
    with patch.object(worker,'DATA',data):
        worker.import_video(dict(project_id=project.name,options=dict(url=args.url)),folder)
    state=worker.read(project/'project.json')
    assert len(state['uploads'])==1
    upload=state['uploads'][0]
    audio=sf.info(project/'uploads'/upload['file'])
    assert audio.samplerate==16000 and audio.channels==1 and audio.duration>0
    print('VIDEO IMPORT CHECK PASSED',dict(name=upload['name'],seconds=round(audio.duration,2),sample_rate=audio.samplerate,channels=audio.channels),flush=True)
