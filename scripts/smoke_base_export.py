"""Export an isolated multi-speaker base directly, without training or a discriminator."""
import hashlib
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import worker
    import vits_engine as engine
    from vits_fast_fine_tuning.models import SynthesizerTrn
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='base-export-smoke-', dir=ROOT/'data/tmp') as temporary:
        root = Path(temporary); base = root/'base'; base.mkdir(); (root/'models').mkdir()
        cfg = worker.read(ROOT/'vendor/vits/training_data/configs/c_finetune_speaker.json')
        cfg['model'].update(inter_channels=16,hidden_channels=16,filter_channels=32,n_heads=2,n_layers=2,
                            gin_channels=16,upsample_initial_channel=32,resblock_kernel_sizes=[3],
                            resblock_dilation_sizes=[[1,3,5]],n_layers_q=1)
        cfg['data'].update(n_speakers=0,filter_length=512,win_length=512)
        cfg['train']['segment_size']=2048
        cfg['speakers']={'甲':0,'乙':1,'丙':2}
        worker.save(base/'config.json',cfg)
        model=SynthesizerTrn(len(cfg['symbols']),257,8,n_speakers=3,**cfg['model'])
        torch.save({'model':model.state_dict()},base/'G_0.pth'); del model
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
        folder=root/'job'; folder.mkdir()
        with patch.object(worker,'DATA',root),patch.object(engine,'DATA',root):
            engine.export_base(dict(options={'base':str(base),'quantize':False}),folder)
        assert worker.read(folder/'export/speakers.json')=={'0':'甲','1':'乙','2':'丙'}
        manifest=next((root/'models').glob('*/model.json')); info=worker.read(manifest)
        assert int(info['n_speakers'])==3
        output=manifest.parent/info['model']
        tts=worker.tts_for(manifest.parent,output,output.parent/'tokens.txt')
        assert len(tts.generate('你好，底模导出测试。',sid=2).samples)>0
        assert list((root/'models').glob('*/voice.tar.bz2'))
        assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir()}
        print('PASS: direct base export, inferred speaker count, names, nonzero speaker synthesis, tar.bz2, unchanged source',flush=True)


if __name__=='__main__':
    main()
