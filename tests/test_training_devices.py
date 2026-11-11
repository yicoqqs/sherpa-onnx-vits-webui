import contextlib
import subprocess
import sys
import time
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import vits_engine
import app

class DeviceTests(unittest.TestCase):
    def fake_torch(self, count):
        cuda=Mock(); cuda.is_available.return_value=count>0; cuda.device_count.return_value=count
        cuda.device.side_effect=lambda _:contextlib.nullcontext()
        return SimpleNamespace(cuda=cuda,ones=Mock())
    def test_cpu(self):
        self.assertEqual(vits_engine.training_devices(self.fake_torch(0)),[])
    def test_single(self):
        self.assertEqual(vits_engine.training_devices(self.fake_torch(1)),[0])
    def test_multiple(self):
        self.assertEqual(vits_engine.training_devices(self.fake_torch(2)),[0,1])
    def test_unusable_device_skipped(self):
        torch=self.fake_torch(2); torch.ones.side_effect=[RuntimeError('device unavailable'),object()]
        self.assertEqual(vits_engine.training_devices(torch),[1])
    def test_all_devices_unavailable_use_cpu(self):
        torch=self.fake_torch(2); torch.ones.side_effect=RuntimeError('device unavailable')
        self.assertEqual(vits_engine.training_devices(torch),[])
    @unittest.skipIf(app.os.name=='nt','POSIX process groups')
    def test_cancel_stops_entire_worker_group(self):
        process=Mock(pid=123);process.poll.return_value=None
        with patch.object(app.os,'killpg') as kill:
            app.terminate_job(process)
            kill.assert_called_once_with(123,app.signal.SIGTERM)

    @unittest.skipIf(app.os.name=='nt','POSIX process groups')
    def test_real_worker_child_is_stopped(self):
        script="import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(child.pid,flush=True); time.sleep(60)"
        process=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,text=True,start_new_session=True)
        try:
            child=int(process.stdout.readline())
            app.terminate_job(process)
            process.wait(timeout=10)
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                stat=Path(f'/proc/{child}/stat')
                if not stat.exists() or stat.read_text().split()[2]=='Z': break
                time.sleep(.05)
            else: self.fail('Cancelled worker left a running child process')
        finally:
            try: app.os.killpg(process.pid,app.signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=10)
            process.stdout.close()
