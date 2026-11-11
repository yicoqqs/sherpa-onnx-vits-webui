"""Real isolated CPU/single GPU/DDP train-resume-export validation."""
from pathlib import Path
import argparse
import sys
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def main():
    import numpy as np
    import soundfile as sf
    import torch
    import worker
    import vits_engine as engine
    parser=argparse.ArgumentParser(); parser.add_argument('--export',action='store_true'); parser.add_argument('--base-variant',choices=['C','CJE'],default='CJE')
    parser.add_argument('--mode', choices=['single','add','modify'], default='single'); args=parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='device-smoke-',dir=ROOT/'data/tmp') as temporary:
        folder=Path(temporary); data=folder/'data'; base=data/'base'; base.mkdir(parents=True)
        project=data/'projects/000000000001'; (project/'clips').mkdir(parents=True); (data/'models').mkdir()
        cfg=worker.read(ROOT/'vendor/vits/training_data/configs'/('c_finetune_speaker.json' if args.base_variant=='C' else 'cje_finetune_speaker.json'))
        cfg['model'].update(inter_channels=16,hidden_channels=16,filter_channels=32,n_heads=2,n_layers=2,gin_channels=16,upsample_initial_channel=32,resblock_kernel_sizes=[3],resblock_dilation_sizes=[[1,3,5]],n_layers_q=1)
        cfg['data'].update(n_speakers=3,filter_length=512,win_length=512); cfg['train']['segment_size']=2048
        worker.save(base/'config.json',cfg)
        g,d=engine.make_models(cfg); original={k:v.clone() for k,v in g.state_dict().items()}; torch.save({'model':original},base/'G_0.pth'); torch.save({'model':d.state_dict()},base/'D_0.pth'); del g,d
        clips=[]
        for i in range(6):
            key=f'{i+1:012x}'; signal=(.1*np.sin(2*np.pi*(200+i*20)*np.arange(24000)/16000)).astype('float32')
            sf.write(project/'clips'/f'{key}.wav',signal,16000); clips.append(dict(id=key,text='你好《世界》。',duration=1.5,enabled=True))
        worker.save(project/'project.json',dict(id=project.name,name='Device smoke',created=0,clips=clips))
        job=dict(project_id=project.name,options=dict(base=str(base),epochs=2,batch_size=1,learning_rate=.0001,resume=False,mode=args.mode,reference_speaker=2))
        devices=engine.training_devices(torch)
        with patch.object(worker,'DATA',data),patch.object(engine,'DATA',data):
            engine.train(job,folder)
            checkpoint=torch.load(project/'training/resume.pth',map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==2 and checkpoint['steps']>=4
            assert not any(k.startswith('module.') for k in checkpoint['generator'])
            del checkpoint
            print('PASS: real automatic device training (2 epochs, multiple optimizer steps)',flush=True)
            job['options'].update(resume=True,epochs=1)
            # Continue on CPU even if the first run used multiple GPUs.
            with patch.object(engine,'training_devices',return_value=[]): engine.train(job,folder)
            checkpoint=torch.load(project/'training/resume.pth',map_location='cpu',weights_only=False)
            assert checkpoint['epoch']==3
            if args.mode != 'single':
                target=3 if args.mode=='add' else 2
                for key,value in original.items():
                    if key != 'emb_g.weight':
                        assert torch.equal(checkpoint['generator'][key],value), key
                    else:
                        for sid in range(3):
                            if sid!=target: assert torch.equal(checkpoint['generator'][key][sid],value[sid])
                        assert not torch.equal(checkpoint['generator'][key][target],value[2])
                print('PASS: shared weights and other speakers unchanged after real training and resume',flush=True)
            print('PASS: checkpoint and optimizer resume on CPU',flush=True)
            del checkpoint
            if devices:
                engine.train(job,folder)
                checkpoint=torch.load(project/'training/resume.pth',map_location='cpu',weights_only=False)
                assert checkpoint['epoch']==4
                del checkpoint
                print('PASS: CPU checkpoint resumes on automatic GPU devices',flush=True)
            if args.export:
                output=folder/'export-job'; output.mkdir()
                engine.export(dict(project_id=project.name,options={'quantize':False}),output)
                names=worker.read(output/'export/speakers.json')
                assert len(names)==(4 if args.mode=='add' else 3 if args.mode=='modify' else 1)
                assert list((data/'models').glob('*/voice.tar.bz2'))
                manifest=next((data/'models').glob('*/model.json'));info=worker.read(manifest)
                model=manifest.parent/info['model']
                tts=worker.tts_for(manifest.parent,model,model.parent/'tokens.txt')
                assert len(tts.generate('你好，这是中文底模的试听测试。').samples)>0
                print('PASS: ONNX dynamic lengths, sherpa load and model package',flush=True)
    print('DEVICE SMOKE PASSED',flush=True)

if __name__=='__main__': main()
