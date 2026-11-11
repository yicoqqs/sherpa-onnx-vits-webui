"""Validate real media/TTS/ASR and a tiny train/export cycle without changing user projects."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import soundfile as sf
import torch
import worker
import vits_engine

torch.set_num_threads(4)
with tempfile.TemporaryDirectory(prefix='deployment-smoke-', dir=ROOT/'data/tmp') as temporary:
    folder = Path(temporary)
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=2',
                    '-c:a','flac',str(folder/'test.mkv')],check=True)
    worker.ffmpeg(folder/'test.mkv',folder/'decoded.wav')
    assert sf.info(folder/'decoded.wav').samplerate == 16000
    print('PASS: MKV decode and WAV extraction',flush=True)

    candidates = list((ROOT/'data/models').glob('*/model.json'))
    candidates += list((ROOT/'data/trash').glob('*/model/model.json'))
    if candidates:
        manifest = next((p for p in candidates if worker.read(p).get('n_speakers') in (1,'1')),candidates[0])
        info=worker.read(manifest)
        model=manifest.parent/Path(str(info['model']).replace('\\','/'))
        engine=worker.tts_for(manifest.parent,model,model.parent/'tokens.txt')
        speech=engine.generate('你好，这是服务器部署后的语音试听测试。',sid=0,speed=1)
        assert len(speech.samples)>0
        sf.write(folder/'speech.wav',speech.samples,speech.sample_rate)
        worker.ffmpeg(folder/'speech.wav',folder/'speech-16k.wav')
        print('PASS: real sherpa-onnx Chinese synthesis',flush=True)
        from faster_whisper import WhisperModel
        whisper=WhisperModel(worker.whisper_model_path('tiny'),device='cpu',compute_type='int8')
        wave,_=sf.read(folder/'speech-16k.wav',dtype='float32')
        segments,_=whisper.transcribe(wave,language='zh',vad_filter=True,beam_size=1)
        recognized=list(segments)
        assert recognized
        print(f'PASS: cached Tiny Whisper transcription ({len(recognized)} segments)',flush=True)
        del whisper,engine
    else:
        raise RuntimeError('No migrated ONNX package available for the TTS/ASR check')

    data=folder/'data'
    base=data/'base';base.mkdir(parents=True)
    project=data/'projects/000000000001';(project/'clips').mkdir(parents=True)
    (data/'models').mkdir()
    config=worker.read(ROOT/'vendor/vits/training_data/configs/cje_finetune_speaker.json')
    config['model'].update(inter_channels=16,hidden_channels=16,filter_channels=32,n_heads=2,
                           n_layers=2,gin_channels=16,upsample_initial_channel=32,
                           resblock_kernel_sizes=[3],resblock_dilation_sizes=[[1,3,5]],n_layers_q=1)
    config['data'].update(n_speakers=1,filter_length=512,win_length=512)
    config['train']['segment_size']=2048
    worker.save(base/'config.json',config)
    generator,discriminator=vits_engine.make_models(config)
    torch.save({'model':generator.state_dict()},base/'G_0.pth')
    torch.save({'model':discriminator.state_dict()},base/'D_0.pth')
    del generator,discriminator
    clips=[]
    for index in range(3):
        key=f'{index+1:012x}'
        signal=(.1*np.sin(2*np.pi*(200+index*20)*np.arange(24000)/16000)).astype('float32')
        sf.write(project/'clips'/f'{key}.wav',signal,16000)
        clips.append(dict(id=key,text='你好。',duration=1.5,enabled=True))
    worker.save(project/'project.json',dict(id=project.name,name='Deployment test',created=0,clips=clips))
    job=dict(project_id=project.name,options=dict(base=str(base),epochs=1,batch_size=2,learning_rate=.0001,resume=False))
    with patch.object(worker,'DATA',data),patch.object(vits_engine,'DATA',data):
        vits_engine.train(job,folder)
        print('PASS: real VITS forward/backward and checkpoint save',flush=True)
        job['options']['resume']=True
        vits_engine.train(job,folder)
        print('PASS: checkpoint and optimizer resume',flush=True)
        export=folder/'export-job';export.mkdir()
        vits_engine.export(dict(project_id=project.name,options={'quantize':False}),export)
        assert list((data/'models').glob('*/voice.tar.bz2'))
        print('PASS: ONNX export, variable-length inference, sherpa load and packaging',flush=True)
print('DEPLOYMENT SMOKE PASSED',flush=True)
