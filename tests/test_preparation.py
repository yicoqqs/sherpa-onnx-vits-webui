import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import worker
from preparation import pending_uploads


class PendingTests(unittest.TestCase):
    def test_incremental_vocals_preserve_edits_and_use_separated_audio(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory(dir=worker.DATA/'tmp') as temp:
            root=Path(temp); project=root/'projects/p'; project.mkdir(parents=True)
            folder=root/'job'; folder.mkdir()
            old=dict(id='old',source_file='a',text='已校对',enabled=False,duration=1)
            worker.save(project/'project.json',dict(uploads=[dict(file='a',name='old'),dict(file='b',name='new')],clips=[old]))
            def separate(source,destination,*args):
                self.assertEqual(source.name,'b')
                sf.write(destination,np.ones(32000,dtype=np.float32)*.1,16000)
            def decode(source,destination):
                self.assertTrue(source.name.startswith('vocals-'))
                sf.write(destination,*sf.read(source))
            class Model:
                def __init__(self,*args,**kwargs): pass
                def transcribe(self,*args,**kwargs):
                    return iter([SimpleNamespace(start=0,end=1,text='你好',words=[])]),None
            with patch.object(worker,'DATA',root),patch.object(worker,'ffmpeg',decode),patch.object(worker,'whisper_model_path',return_value='cache'),patch('vocal_separation.separate_vocals',side_effect=separate),patch.dict('sys.modules',{'faster_whisper':SimpleNamespace(WhisperModel=Model)}):
                worker.prepare(dict(project_id='p',options=dict(whisper='small',mode='new',separate_vocals=True)),folder)
            state=worker.read(project/'project.json')
            self.assertEqual(state['clips'][0],old)
            self.assertEqual(len(state['clips']),2)
            self.assertTrue(state['clips'][1]['vocals_separated'])
            self.assertEqual(pending_uploads(state),[])

    def test_preserves_processed_sources_and_detects_new_upload(self):
        state = dict(uploads=[dict(file='a',name='same'),dict(file='b',name='same'),dict(file='c',name='new')],
                     clips=[dict(source_file='a',text='校对',enabled=False)],processed_uploads=['b'])
        self.assertEqual([u['file'] for u in pending_uploads(state)], ['c'])
        self.assertEqual(state['clips'][0]['text'], '校对')

    def test_legacy_names_are_only_used_when_unique(self):
        state=dict(uploads=[dict(file='a',name='old')],clips=[dict(source='old')])
        self.assertEqual(pending_uploads(state), [])
        state['uploads'].append(dict(file='b',name='old'))
        self.assertEqual(len(pending_uploads(state)), 2)
