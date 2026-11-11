import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import worker
from chinese_text import normalize_training_phones, punctuation_for_symbols

class ChineseBaseTests(unittest.TestCase):
    def test_punctuation_matches_each_base_without_dropping_real_phones(self):
        self.assertEqual(normalize_training_phones('ㄋㄧˇ,ㄏㄠˇ?',set('ㄋㄧˇㄏㄠ，？')),'ㄋㄧˇ，ㄏㄠˇ？')
        self.assertEqual(normalize_training_phones('ni,hao?',set('ni,hao?')),'ni,hao?')
        self.assertEqual(normalize_training_phones('unknown',set('ㄋㄧ')),'unknown')
        self.assertEqual(punctuation_for_symbols(['_','ㄋ','，','。','？']),'， 。 ？')

    def test_weights_use_the_matching_config_for_each_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);configs=root/'vendor/vits/training_data/configs';configs.mkdir(parents=True)
            for name in ('c_finetune_speaker.json','cje_finetune_speaker.json'):
                worker.save(configs/name,worker.read(worker.ROOT/'vendor/vits/training_data/configs'/name))
            content=b'cached test weights';info={'oid':'sha256:'+hashlib.sha256(content).hexdigest(),'size':str(len(content))}
            worker.save(root/'vendor/vits/base-models.json',{'revision':'test','models':{v:{'G_0.pth':info,'D_0.pth':info} for v in ('C','CJE')}})
            for variant,cleaner in [('C','chinese_cleaners'),('CJE','cjke_cleaners2')]:
                with self.subTest(variant=variant):
                    destination=root/'data/base'/variant.lower();destination.mkdir(parents=True)
                    for name in ('G_0.pth','D_0.pth'):(destination/name).write_bytes(content)
                    folder=root/variant;folder.mkdir()
                    with patch.object(worker,'ROOT',root),patch.object(worker,'DATA',root/'data'),patch.object(worker.urllib.request,'urlopen') as network:
                        worker.download_base({'options':{'variant':variant}},folder)
                        network.assert_not_called()
                    self.assertEqual(worker.read(destination/'config.json')['data']['text_cleaners'],[cleaner])
                    self.assertEqual(worker.read(folder/'result.json'),{'base':str(destination.resolve()),'variant':variant})
