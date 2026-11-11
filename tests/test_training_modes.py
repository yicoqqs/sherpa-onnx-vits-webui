import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from training_modes import base_speaker_count, configure_training, validate_resume
from vits_engine import load_weights, protect_speakers
from fastapi.testclient import TestClient
import app as studio


class TrainingModeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(data={'n_speakers': 3}, model={}, symbols=['a'], train={})
        self.options = dict(learning_rate=.001, batch_size=1, reference_speaker=2)

    def config(self, mode):
        return configure_training(self.cfg, '.', '新音色', dict(self.options, mode=mode))

    def test_layout_and_reference_validation(self):
        for mode, count, target in [('add', 4, 3), ('modify', 3, 2), ('single', 1, 0)]:
            cfg = self.config(mode)
            self.assertEqual(cfg['data']['n_speakers'], count)
            self.assertEqual(cfg['finetune']['target_speaker'], target)
            self.assertEqual(cfg['speakers'][str(target)], '新音色')
            self.assertEqual(len(cfg['speakers']), count)
        self.assertEqual(self.cfg['data']['n_speakers'], 3)
        with self.assertRaises(ValueError):
            configure_training(self.cfg, '.', 'x', dict(self.options, reference_speaker=3))

    def test_resume_rejects_different_target_and_protection(self):
        cfg = self.config('modify')
        validate_resume(cfg, cfg)
        for field, value in [('reference_speaker', 1), ('protect_existing', False), ('base', '/other')]:
            changed = copy.deepcopy(cfg)
            changed['finetune'][field] = value
            with self.assertRaises(ValueError):
                validate_resume(cfg, changed)
        legacy = self.config('single')
        del legacy['finetune']
        single = configure_training(self.cfg, '.', 'x', dict(self.options, reference_speaker=0))
        validate_resume(legacy, single)
        with self.assertRaises(ValueError):
            validate_resume(legacy, self.config('single'))

    def test_upstream_zero_count_is_inferred_from_weights(self):
        with tempfile.TemporaryDirectory(dir=studio.DATA/'tmp') as temporary:
            cfg = copy.deepcopy(self.cfg); cfg['data']['n_speakers'] = 0
            torch.save({'model': {'emb_g.weight': torch.ones(3, 4)}}, Path(temporary)/'G_0.pth')
            self.assertEqual(base_speaker_count(temporary, cfg), 3)
            result = configure_training(cfg, temporary, '新增', dict(self.options, mode='add'))
            self.assertEqual(result['data']['n_speakers'], 4)
            self.assertEqual(cfg['data']['n_speakers'], 0)

    def test_weight_initialization_and_protected_optimizer_resume(self):
        class Voice(torch.nn.Module):
            def __init__(self, count):
                super().__init__()
                self.emb_g = torch.nn.Embedding(count, 4)
                self.shared = torch.nn.Linear(4, 4)
            def forward(self, sid):
                return self.shared(self.emb_g(sid))
        original = Voice(3)
        with tempfile.TemporaryDirectory(dir=studio.DATA/'tmp') as temp:
            path = Path(temp)/'G.pth'
            torch.save({'model': original.state_dict()}, path)
            for mode, count, target in [('add', 4, 3), ('modify', 3, 2), ('single', 1, 0)]:
                model = Voice(count)
                load_weights(model, path, speaker=True, mode=mode, reference=2)
                self.assertTrue(torch.equal(model.emb_g.weight[target], original.emb_g.weight[2]))
                if mode == 'single':
                    continue
                protect_speakers(model, target)
                before = copy.deepcopy(model.state_dict())
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, weight_decay=0)
                for step in range(3):
                    optimizer.zero_grad()
                    model(torch.tensor([target])).square().sum().backward()
                    optimizer.step()
                    if step == 0:
                        resumed = Voice(count)
                        protect_speakers(resumed, target)
                        resumed.load_state_dict(model.state_dict())
                        next_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=.01, weight_decay=0)
                        next_optimizer.load_state_dict(optimizer.state_dict())
                        model, optimizer = resumed, next_optimizer
                for key, value in before.items():
                    if key != 'emb_g.weight':
                        self.assertTrue(torch.equal(model.state_dict()[key], value))
                for sid in range(count):
                    self.assertEqual(torch.equal(model.emb_g.weight[sid], before['emb_g.weight'][sid]), sid != target)

    def test_api_speakers_validation_and_resume_settings(self):
        with tempfile.TemporaryDirectory(dir=studio.DATA/'tmp') as temporary:
            root = Path(temporary)
            base = root/'base'; base.mkdir()
            cfg = dict(self.cfg, speakers={'甲': 0, '乙': 1, '丙': 2})
            studio.save(base/'config.json', cfg)
            for name in ('G_0.pth', 'D_0.pth'):
                (base/name).write_bytes(b'placeholder')
            project = root/'projects/000000000001'; project.mkdir(parents=True)
            studio.save(project/'project.json', dict(name='新音色', clips=[dict(text='你好'), dict(text='世界')]))
            client = TestClient(studio.app)
            with patch.object(studio, 'PROJECTS', root/'projects'), patch.object(studio, 'start_job', return_value={'id': 'test'}) as start:
                speakers = client.get('/api/base/speakers', params={'base': str(base)})
                self.assertEqual(speakers.json()['speakers'][2], {'id': 2, 'name': '丙'})
                url = '/api/projects/000000000001/train'
                body = dict(base=str(base), mode='add', reference_speaker=2)
                self.assertEqual(client.post(url, json=body).status_code, 200)
                self.assertEqual(start.call_args.args[2]['mode'], 'add')
                self.assertEqual(client.post(url, json=dict(body, reference_speaker=3)).status_code, 400)
                self.assertEqual(client.post(url, json=dict(body, mode='wrong')).status_code, 422)
                training = project/'training'; training.mkdir()
                options = dict(self.options, mode='add')
                saved = configure_training(cfg, base, '新音色', options)
                studio.save(training/'config.json', saved)
                (training/'resume.pth').write_bytes(b'placeholder')
                self.assertEqual(client.post(url, json=body).status_code, 200)
                self.assertEqual(client.post(url, json=dict(body, reference_speaker=1)).status_code, 400)
                self.assertEqual(client.get('/api/projects/000000000001').json()['training_settings'], saved['finetune'])
                self.assertEqual(client.post('/api/base/export', json={'base': str(base)}).status_code, 400)  # no symbols / embedding config
                cfg['model']['gin_channels'] = 4
                studio.save(base/'config.json', cfg)
                (base/'D_0.pth').unlink()  # direct export only needs the generator
                self.assertEqual(client.post('/api/base/export', json={'base': str(base)}).status_code, 200)
                self.assertEqual(start.call_args.args[:2], ('export_base', None))


if __name__ == '__main__':
    unittest.main()
