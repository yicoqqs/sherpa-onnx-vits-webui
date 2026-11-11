import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import worker

class SynthesisProgressTests(unittest.TestCase):
    def run_synthesis(self, empty=False):
        def generate(text,sid,speed,callback):
            for value in (0,.25,.2,.75,1): self.assertEqual(callback([],value),1)
            return SimpleNamespace(samples=[] if empty else [.1]*16000,sample_rate=16000)
        engine=SimpleNamespace(num_speakers=1,generate=generate)
        output=io.StringIO()
        with tempfile.TemporaryDirectory() as temp, patch.object(worker,'read',return_value={'name':'test','model':'voice/model.onnx'}), patch.object(worker,'tts_for',return_value=engine), patch('soundfile.write') as write, contextlib.redirect_stdout(output):
            if empty:
                with self.assertRaises(ValueError):worker.synthesize({'options':{'model_id':'test','speaker':0,'speed':1,'text':'你好。'}},Path(temp))
            else:
                worker.synthesize({'options':{'model_id':'test','speaker':0,'speed':1,'text':'你好。'}},Path(temp));write.assert_called_once()
        return [json.loads(line.removeprefix('@@TASK_PROGRESS@@')) for line in output.getvalue().splitlines() if line.startswith('@@TASK_PROGRESS@@')]
    def test_real_callback_progress_is_monotonic_and_finishes_after_save(self):
        progress=self.run_synthesis();values=[p['percent'] for p in progress]
        self.assertEqual(values,sorted(set(values)))
        self.assertEqual(progress[0]['stage'],'loading')
        self.assertEqual(progress[-2]['stage'],'saving')
        self.assertEqual(progress[-1],{'stage':'completed','percent':100,'duration':1.0})
    def test_empty_audio_never_reports_completion(self):
        self.assertNotIn('completed',[p['stage'] for p in self.run_synthesis(empty=True)])
