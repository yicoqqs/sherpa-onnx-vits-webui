import io
import itertools
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import contextlib
import re
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import app as studio
import worker
from video_source import normalize_video_url
from speaker_catalog import model_speakers
from chinese_text import to_simplified, simplify_project, training_text


class SimplifiedChineseTests(unittest.TestCase):
    def test_training_removes_silent_brackets_but_keeps_words_and_pauses(self):
        self.assertEqual(training_text('讀《三國演義》，他說：“你好！”'), '读三国演义，他说：你好！')
        self.assertEqual(training_text('【测试】（旁白）「名称」『标题』〈作品〉，123。'), '测试旁白名称标题作品，123。')

    def test_mixed_script_is_normalized_without_changing_english_or_numbers(self):
        self.assertEqual(to_simplified('這裡是臺灣，發展經驗。Python 3.12，2026年！'), '这里是台湾，发展经验。Python 3.12，2026年！')

    def test_conversion_preserves_original_and_is_idempotent(self):
        value={'clips':[{'id':'1','text':'正確的發展戰略','enabled':False,'duration':2.5}]}
        self.assertEqual(simplify_project(value),1)
        self.assertEqual(value['clips'][0]['text'],'正确的发展战略')
        self.assertEqual(value['clips'][0]['original_text'],'正確的發展戰略')
        self.assertFalse(value['clips'][0]['enabled'])
        self.assertEqual(value['clips'][0]['duration'],2.5)
        self.assertEqual(simplify_project(value),0)


class SpeakerCatalogTests(unittest.TestCase):
    def test_missing_names_fall_back_to_ids(self):
        with tempfile.TemporaryDirectory(dir=studio.DATA) as temp:
            self.assertEqual(model_speakers(Path(temp),{'model':'voice/model.onnx','n_speakers':'3'}),
                             [{'id':0,'name':''},{'id':1,'name':''},{'id':2,'name':''}])

    def test_vits_and_piper_names_and_partial_mapping(self):
        with tempfile.TemporaryDirectory(dir=studio.DATA) as temp:
            root=Path(temp);root.joinpath('voice').mkdir()
            studio.save(root/'voice/config.json',{'speakers':{'女声':0,'男声':2}})
            speakers=model_speakers(root,{'model':'voice/model.onnx','n_speakers':3})
            self.assertEqual([s['name'] for s in speakers],['女声','','男声'])
            studio.save(root/'voice/model.onnx.json',{'speaker_id_map':{'Piper voice':1}})
            speakers=model_speakers(root,{'model':'voice/model.onnx','n_speakers':3})
            self.assertEqual([s['name'] for s in speakers],['女声','Piper voice','男声'])

    def test_explicit_sidecar_takes_priority_and_invalid_json_is_ignored(self):
        with tempfile.TemporaryDirectory(dir=studio.DATA) as temp:
            root=Path(temp);root.joinpath('voice').mkdir()
            studio.save(root/'voice/speakers.json',{'0':'我的音色'})
            (root/'voice/config.json').write_text('invalid json',encoding='utf-8')
            self.assertEqual(model_speakers(root,{'model':'voice/model.onnx','n_speakers':1}),[{'id':0,'name':'我的音色'}])
            self.assertEqual(model_speakers(root,{'model':'voice\\model.onnx','n_speakers':1}),[{'id':0,'name':'我的音色'}])


class VideoImportTests(unittest.TestCase):
    def test_bilibili_links_strip_tracking_but_preserve_selected_part(self):
        self.assertEqual(normalize_video_url('https://www.bilibili.com/video/BV1kE411K71B/?spm_id_from=333.337&vd_source=tracking'), 'https://www.bilibili.com/video/BV1kE411K71B/')
        self.assertEqual(normalize_video_url('https://m.bilibili.com/video/BV1kE411K71B/?p=2&spm_id_from=test'), 'https://www.bilibili.com/video/BV1kE411K71B/?p=2')
        self.assertEqual(normalize_video_url('https://www.bilibili.com/video/av123/?p=01'), 'https://www.bilibili.com/video/av123/?p=1')
        self.assertEqual(normalize_video_url('分享 https://b23.tv/abc123'), 'https://b23.tv/abc123')
        for url in ('https://www.bilibili.com/video/BV1kE411K71B/?p=0','https://www.bilibili.com/video/BV1kE411K71B/?p=no','https://www.bilibili.com/video/BV1kE411K71B/?p=1&p=2','https://bilibili.com.evil.test/video/BV1kE411K71B/','https://www.bilibili.com/playlist/123'):
            with self.assertRaises(ValueError): normalize_video_url(url)

    def test_bilibili_uses_audio_track_and_platform_specific_access_message(self):
        from yt_dlp.utils import DownloadError
        with tempfile.TemporaryDirectory(dir=studio.DATA) as temp:
            folder=Path(temp)
            with patch('yt_dlp.YoutubeDL') as downloader,contextlib.redirect_stdout(io.StringIO()):
                downloader.return_value.__enter__.return_value.extract_info.side_effect=DownloadError('HTTP 412 login required')
                with self.assertRaisesRegex(ValueError,'哔哩哔哩'):
                    worker.import_video(dict(project_id='000000000001',options=dict(url='https://www.bilibili.com/video/BV1kE411K71B/')),folder)
                options=downloader.call_args.args[0]
                self.assertTrue(options['format'].startswith('bestaudio/'))
                self.assertEqual(options['http_headers']['Referer'],'https://www.bilibili.com/')
                self.assertEqual(options['proxy'],'')
                self.assertTrue(options['noplaylist'])
                self.assertNotIn('max_downloads',options)  # yt-dlp raises after a successful download at this limit.
                self.assertEqual(options['playlistend'],1)

    def test_url_normalization_and_restricted_hosts(self):
        self.assertEqual(normalize_video_url('分享 https://www.douyin.com/video/7665731138861747508?modeFrom=searchResult'), 'https://www.douyin.com/video/7665731138861747508')
        self.assertEqual(normalize_video_url('https://v.douyin.com/abc123/'), 'https://v.douyin.com/abc123/')
        direct='https://v11-weba.douyinvod.com/signed/media-video-avc1/?a=6383&cd=0|0|0|3&signature=x=='
        self.assertEqual(normalize_video_url(direct),direct)
        self.assertEqual(normalize_video_url(direct.replace('&','\\&')),direct)
        for url in ('http://127.0.0.1/video/1','https://douyin.com.evil.test/video/1','https://v1.douyinvod.com.evil.test/video','https://www.douyin.com/user/abc','file:///test.mp4'):
            with self.assertRaises(ValueError): normalize_video_url(url)

    def test_video_without_audio_has_actionable_error(self):
        result=SimpleNamespace(returncode=1,stderr='Output file #0 does not contain any stream')
        with patch.object(worker.subprocess,'run',return_value=result):
            with self.assertRaisesRegex(ValueError,'没有音轨'):
                worker.ffmpeg('video-only.mp4','audio.wav')

    @unittest.skipUnless(__import__('shutil').which('ffmpeg'), 'FFmpeg required')
    def test_download_extract_register_and_cookie_cleanup(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory(dir=studio.DATA) as temp:
            root=Path(temp); project=root/'projects/000000000001'
            (project/'uploads').mkdir(parents=True); folder=root/'job'; folder.mkdir()
            worker.save(project/'project.json',dict(clips=[],uploads=[]))
            cookies=root/'cookies.txt'; cookies.write_text('# Netscape HTTP Cookie File\n',encoding='utf-8')
            source=folder/'source.wav'
            sf.write(source,.1*np.sin(np.arange(32000)*.1),16000)
            class Downloader:
                def __init__(self,options): self.options=options
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def extract_info(self,url,download):
                    self.options['progress_hooks'][0](dict(status='finished',downloaded_bytes=100,total_bytes=100))
                    Path(self.options['cookiefile']).write_text('changed copy',encoding='utf-8')
                    return dict(id='1',title='测试视频',ext='wav')
                def prepare_filename(self,info): return str(source)
            output=io.StringIO()
            with patch.object(worker,'DATA',root),patch('yt_dlp.YoutubeDL',Downloader),contextlib.redirect_stdout(output):
                worker.import_video(dict(project_id='000000000001',options=dict(url='https://www.douyin.com/video/1',cookies_path=str(cookies))),folder)
            uploads=worker.read(project/'project.json')['uploads']
            self.assertEqual(len(uploads),1)
            self.assertEqual(sf.info(project/'uploads'/uploads[0]['file']).samplerate,16000)
            self.assertEqual(uploads[0]['source_url'],'https://www.douyin.com/video/1')
            self.assertEqual(cookies.read_text(encoding='utf-8'),'# Netscape HTTP Cookie File\n')
            self.assertFalse((folder/'session-cookies.txt').exists())
            self.assertIn('@@MEDIA_PROGRESS@@',output.getvalue())

    def test_douyin_access_error_leaves_project_unchanged(self):
        from yt_dlp.utils import DownloadError
        with tempfile.TemporaryDirectory(dir=studio.DATA) as temp:
            root=Path(temp); project=root/'projects/000000000001'; project.mkdir(parents=True)
            worker.save(project/'project.json',dict(clips=[],uploads=[])); folder=root/'job';folder.mkdir()
            with patch.object(worker,'DATA',root),patch('yt_dlp.YoutubeDL') as downloader,contextlib.redirect_stdout(io.StringIO()):
                downloader.return_value.__enter__.return_value.extract_info.side_effect=DownloadError('Fresh cookies are needed')
                with self.assertRaisesRegex(ValueError,'Cookies'):
                    worker.import_video(dict(project_id='000000000001',options=dict(url='https://www.douyin.com/video/1')),folder)
            self.assertEqual(worker.read(project/'project.json')['uploads'],[])


class PrepareProgressTests(unittest.TestCase):
    def test_model_download_reports_real_file_bytes(self):
        def download(repo, filename, **kwargs):
            with kwargs['tqdm_class'](total=100, unit='B') as progress:
                progress.update(50)
                progress.display()
                progress.update(50)
            return '/cached/snapshot/' + filename
        output=io.StringIO()
        hub=SimpleNamespace(snapshot_download=lambda *a,**k: (_ for _ in ()).throw(FileNotFoundError()),hf_hub_download=download)
        with patch.dict('sys.modules',{'huggingface_hub':hub}),contextlib.redirect_stdout(output):
            path=worker.whisper_model_path('small')
        events=[json.loads(line.split('@@VITS_PROGRESS@@',1)[1]) for line in output.getvalue().splitlines() if line.startswith('@@VITS_PROGRESS@@')]
        self.assertTrue(any(e.get('downloaded_bytes')==50 and e.get('percent')==50 for e in events))
        self.assertTrue(any(e.get('file')=='model.bin' and e.get('percent')==100 for e in events))
        self.assertEqual(Path(path).name,'snapshot')

    def test_progress_uses_audio_duration_and_finishes_after_saving(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory(dir=studio.DATA) as temp:
            root=Path(temp); project=root/'projects/000000000001'
            (project/'uploads').mkdir(parents=True)
            worker.save(project/'project.json',dict(uploads=[dict(file='one.wav',name='一.wav'),dict(file='two.wav',name='二.wav')],clips=[]))
            folder=root/'job'; folder.mkdir()
            def decode(source,destination):
                sf.write(destination,np.zeros(32000,dtype=np.float32),16000)
            class Model:
                def __init__(self,*args,**kwargs): pass
                def transcribe(self,*args,**kwargs):
                    segments=[SimpleNamespace(start=i,end=i+1,text='你好，發展',words=[SimpleNamespace(start=i,end=i+1,word='你好，發展')]) for i in (0,1)]
                    return iter(segments),None
            output=io.StringIO()
            with patch.object(worker,'DATA',root),patch.object(worker,'ffmpeg',decode),patch.object(worker,'whisper_model_path',return_value='cached-model'),patch.dict('sys.modules',{'faster_whisper':SimpleNamespace(WhisperModel=Model)}),contextlib.redirect_stdout(output):
                worker.prepare(dict(project_id='000000000001',options={'whisper':'tiny'}),folder)
            events=[json.loads(line.split('@@VITS_PROGRESS@@',1)[1]) for line in output.getvalue().splitlines() if line.startswith('@@VITS_PROGRESS@@')]
            self.assertEqual(events[0]['stage'],'loading')
            self.assertIsNone(events[0]['percent'])
            values=[event['percent'] for event in events if event['percent'] is not None]
            self.assertEqual(values,sorted(values))
            self.assertIn(25,values)
            self.assertEqual(events[-1]['percent'],100)
            self.assertEqual(events[-1]['total_seconds'],4)
            self.assertEqual(events[-1]['clips'],4)
            self.assertEqual(len(worker.read(project/'project.json')['clips']),4)
            self.assertTrue(all(c['text']=='你好，发展' and c['original_text']=='你好，發展' for c in worker.read(project/'project.json')['clips']))
            self.assertTrue(all(value<100 for value in values[:-1]))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=studio.DATA)
        root = Path(self.temp.name)
        self.patches = []
        for name in ('PROJECTS','MODELS','JOBS'):
            directory = root / name.lower()
            directory.mkdir()
            item = patch.object(studio, name, directory)
            item.start(); self.patches.append(item)
        self.client = TestClient(studio.app)
        self.project = self.client.post('/api/projects',json={'name':'测试声音'}).json()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_project_survives_reload(self):
        data=self.client.get('/api/projects/'+self.project['id']).json()
        self.assertEqual(data['name'],'测试声音')

    def test_frontend_script_ids_match_page_and_assets_revalidate(self):
        page=self.client.get('/')
        script=self.client.get('/app.js')
        ids=set(re.findall(r'id="([A-Za-z0-9_-]+)"',page.text))
        referenced=set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)",script.text))
        self.assertEqual(referenced-ids,set(),'页面和脚本必须同步部署，不得引用缺失的控件')
        self.assertIn('train-progress',ids)
        self.assertIn('export-progress',ids)
        self.assertIn('synthesize-progress',ids)
        self.assertIn('/app.js?v=',page.text)
        for path in ('/','/index.html','/app.js','/style.css'):
            self.assertEqual(self.client.get(path).headers.get('cache-control'),'no-cache, must-revalidate')

    def test_export_job_exposes_exact_generated_model(self):
        key='000000000099';folder=studio.JOBS/key;folder.mkdir()
        studio.save(folder/'job.json',dict(id=key,kind='export',status='completed',created=0))
        studio.save(folder/'result.json',{'model_id':'000000000123'})
        self.assertEqual(self.client.get(f'/api/jobs/{key}').json()['result']['model_id'],'000000000123')

    def test_base_download_defaults_to_chinese_and_preserves_explicit_cje(self):
        with patch.object(studio,'start_job',return_value={'id':'test'}) as start:
            self.assertEqual(self.client.post('/api/base/download').status_code,200)
            start.assert_called_with('download_base',None,{'variant':'C'})
            self.assertEqual(self.client.post('/api/base/download',json={'variant':'CJE'}).status_code,200)
            start.assert_called_with('download_base',None,{'variant':'CJE'})
            self.assertEqual(self.client.post('/api/base/download',json={'variant':'unknown'}).status_code,422)

    def test_model_api_includes_named_and_unnamed_speakers(self):
        folder=studio.MODELS/'000000000001';(folder/'voice').mkdir(parents=True)
        studio.save(folder/'model.json',dict(id='000000000001',name='测试模型',created=0,model='voice/model.onnx',n_speakers=2))
        studio.save(folder/'voice/speakers.json',{'1':'旁白'})
        models=self.client.get('/api/models').json()
        self.assertEqual(models[0]['speakers'],[{'id':0,'name':''},{'id':1,'name':'旁白'}])

    def test_delete_imported_model_and_restore_keeps_source_and_other_models(self):
        key='000000000001';folder=studio.MODELS/key;(folder/'voice').mkdir(parents=True)
        source=Path(self.temp.name)/'original.tar.bz2';source.write_bytes(b'original package')
        info=dict(id=key,name='导入音色',created=0,origin='import',model='voice/model.onnx',n_speakers=1)
        studio.save(folder/'model.json',info)
        (folder/'voice/model.onnx').write_bytes(b'weights');(folder/'voice.tar.bz2').write_bytes(b'archive')
        other=studio.MODELS/'000000000002';other.mkdir();(other/'keep').write_bytes(b'keep')
        response=self.client.delete('/api/models/'+key)
        self.assertEqual(response.status_code,200)
        self.assertFalse(folder.exists());self.assertTrue(other.exists())
        self.assertEqual(source.read_bytes(),b'original package')
        self.assertEqual(self.client.get('/api/models/'+key+'/download').status_code,404)
        token=response.json()['undo']
        restored=self.client.post('/api/trash/'+token+'/restore')
        self.assertEqual(restored.json(),info)
        self.assertEqual((folder/'voice/model.onnx').read_bytes(),b'weights')
        self.assertEqual(self.client.get('/api/models/'+key+'/download').status_code,200)

    def test_delete_exported_model_preserves_training_and_can_restore(self):
        key='000000000001';folder=studio.MODELS/key;folder.mkdir()
        info=dict(id=key,origin=self.project['id'])
        studio.save(folder/'model.json',info)
        (folder/'voice.tar.bz2').write_bytes(b'exported archive')
        training=studio.PROJECTS/self.project['id']/'training';training.mkdir()
        (training/'G_latest.pth').write_bytes(b'checkpoint')
        response=self.client.delete('/api/models/'+key)
        self.assertEqual(response.status_code,200)
        self.assertFalse(folder.exists())
        self.assertEqual((training/'G_latest.pth').read_bytes(),b'checkpoint')
        restored=self.client.post('/api/trash/'+response.json()['undo']+'/restore')
        self.assertEqual(restored.json(),info)
        self.assertEqual((folder/'voice.tar.bz2').read_bytes(),b'exported archive')

    def test_running_job_blocks_model_delete_and_restore(self):
        key='000000000001';folder=studio.MODELS/key;folder.mkdir()
        studio.save(folder/'model.json',dict(id=key,origin='import'))
        with patch.object(studio,'active',{'task':object()}):
            self.assertEqual(self.client.delete('/api/models/'+key).status_code,409)
        deleted=self.client.delete('/api/models/'+key).json()
        with patch.object(studio,'active',{'task':object()}):
            self.assertEqual(self.client.post('/api/trash/'+deleted['undo']+'/restore').status_code,409)

    def test_delete_and_restore_project_preserves_checkpoints_and_exported_models(self):
        key=self.project['id']; folder=studio.PROJECTS/key
        (folder/'training').mkdir(); (folder/'training/G_latest.pth').write_bytes(b'checkpoint')
        exported=studio.MODELS/'000000000002'; exported.mkdir(); (exported/'model.onnx').write_bytes(b'exported')
        response=self.client.delete('/api/projects/'+key)
        self.assertEqual(response.status_code,200)
        self.assertFalse(folder.exists()); self.assertTrue(exported.exists())
        self.assertEqual(self.client.get('/api/projects').json(),[])
        token=response.json()['undo']
        self.assertEqual(self.client.post('/api/trash/'+token+'/restore').status_code,200)
        self.assertEqual((folder/'training/G_latest.pth').read_bytes(),b'checkpoint')
        self.assertEqual(self.client.post('/api/trash/'+token+'/restore').status_code,409)

    def test_delete_upload_removes_only_associated_clips_and_undo_restores_edits(self):
        key=self.project['id'];folder=studio.PROJECTS/key;(folder/'clips').mkdir()
        state=studio.read(folder/'project.json')
        state['uploads']=[dict(file='a.wav',name='same.wav',bytes=1),dict(file='b.wav',name='same.wav',bytes=1)]
        state['clips']=[dict(id='000000000001',text='已校对文字',enabled=False,source='same.wav',source_file='a.wav'),dict(id='000000000002',text='保留',enabled=True,source='same.wav',source_file='b.wav')]
        studio.save(folder/'project.json',state)
        for name in ('a.wav','b.wav'): (folder/'uploads'/name).write_bytes(b'audio')
        for c in state['clips']: (folder/'clips'/(c['id']+'.wav')).write_bytes(b'clip')
        response=self.client.delete(f'/api/projects/{key}/uploads/a.wav')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['removed_clips'],1)
        self.assertEqual([c['id'] for c in response.json()['project']['clips']],['000000000002'])
        self.assertFalse((folder/'uploads/a.wav').exists());self.assertTrue((folder/'uploads/b.wav').exists())
        restored=self.client.post('/api/trash/'+response.json()['undo']+'/restore').json()
        self.assertEqual(restored,state)
        self.assertTrue((folder/'clips/000000000001.wav').exists())

    def test_single_clip_delete_and_legacy_source_association(self):
        key=self.project['id'];folder=studio.PROJECTS/key;(folder/'clips').mkdir()
        state=studio.read(folder/'project.json');state['uploads']=[dict(file='a.wav',name='原录音.wav',bytes=1)]
        state['clips']=[dict(id='000000000001',text='旧片段',source='原录音.wav')]
        studio.save(folder/'project.json',state)
        (folder/'uploads/a.wav').write_bytes(b'audio');(folder/'clips/000000000001.wav').write_bytes(b'clip')
        response=self.client.delete(f'/api/projects/{key}/clips/000000000001')
        self.assertEqual(response.json()['project']['clips'],[])
        self.assertTrue((folder/'uploads/a.wav').exists())
        self.client.post('/api/trash/'+response.json()['undo']+'/restore')
        response=self.client.delete(f'/api/projects/{key}/uploads/a.wav')
        self.assertEqual(response.json()['removed_clips'],1)

    def test_running_task_blocks_deletion(self):
        key=self.project['id']
        with patch.object(studio,'active',{'active-job':object()}):
            self.assertEqual(self.client.delete('/api/projects/'+key).status_code,409)
        self.assertTrue((studio.PROJECTS/key).exists())

    def test_video_import_validates_url_and_queues_canonical_source(self):
        url='/api/projects/'+self.project['id']+'/video'
        self.assertEqual(self.client.post(url,json={'url':'http://127.0.0.1/private'}).status_code,400)
        with patch.object(studio,'start_job',return_value={'status':'running'}) as start:
            result=self.client.post(url,json={'url':'https://www.douyin.com/video/123?modeFrom=searchResult'})
            self.assertEqual(result.status_code,200)
            self.assertEqual(start.call_args.args[2]['url'],'https://www.douyin.com/video/123')

    def test_original_media_can_be_downloaded_even_if_audio_extraction_fails(self):
        folder=studio.JOBS/'000000000001';folder.mkdir()
        (folder/'source.mp4').write_bytes(b'test-video')
        studio.save(folder/'media.json',{'file':'source.mp4'})
        response=self.client.get('/api/jobs/000000000001/video')
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.content,b'test-video')

    def test_upload_batch_rolls_back_on_invalid_extension(self):
        response=self.client.post('/api/projects/'+self.project['id']+'/upload',files=[
            ('files',('recording.wav',b'RIFFtest','audio/wav')),
            ('files',('bad.exe',b'bad','application/octet-stream'))])
        self.assertEqual(response.status_code,400)
        self.assertEqual(list((studio.PROJECTS/self.project['id']/'uploads').iterdir()),[])
        self.assertNotIn('uploads',studio.read(studio.PROJECTS/self.project['id']/'project.json'))

    def test_upload_and_prepare_validation(self):
        key=self.project['id']
        self.assertEqual(self.client.post(f'/api/projects/{key}/prepare',json={}).status_code,400)
        self.assertEqual(self.client.post(f'/api/projects/{key}/upload',files=[('files',('voice.wav',b'RIFFtest','audio/wav'))]).status_code,200)
        self.assertEqual(len(self.client.get(f'/api/projects/{key}').json()['uploads']),1)

    def test_cross_origin_and_dns_rebinding_rejected(self):
        self.assertEqual(self.client.post('/api/projects',json={'name':'evil'},headers={'Origin':'https://example.com'}).status_code,403)
        self.assertEqual(self.client.get('/api/projects',headers={'Host':'evil.example'}).status_code,403)

    def test_configured_server_ip_is_allowed(self):
        with patch.dict('os.environ',{'VITS_ALLOWED_HOSTS':'192.168.0.150'}):
            self.assertEqual(self.client.get('/api/projects',headers={'Host':'192.168.0.150:7860'}).status_code,200)
            self.assertEqual(self.client.post('/api/projects',json={'name':'LAN'},headers={'Host':'192.168.0.150:7860','Origin':'http://192.168.0.150:7860'}).status_code,200)
            self.assertEqual(self.client.get('/api/projects',headers={'Host':'192.168.0.151:7860'}).status_code,403)

    def test_training_requires_data_and_checkpoint(self):
        key=self.project['id']
        self.assertEqual(self.client.post(f'/api/projects/{key}/train',json={'base':'missing'}).status_code,400)
        self.assertEqual(self.client.post(f'/api/projects/{key}/export',json={}).status_code,400)

    def test_recovery_marks_interrupted_jobs(self):
        folder=studio.JOBS/'000000000001'; folder.mkdir()
        studio.save(folder/'job.json',dict(status='running'))
        (folder/'session-cookies.txt').write_text('temporary copy',encoding='utf-8')
        studio.recover()
        self.assertEqual(studio.read(folder/'job.json')['status'],'interrupted')
        self.assertFalse((folder/'session-cookies.txt').exists())

    def test_clip_editor_preserves_metadata(self):
        folder=studio.PROJECTS/self.project['id']
        value=studio.read(folder/'project.json')
        value['clips']=[dict(id='000000000001',text='你好',duration=2,enabled=True)]
        studio.save(folder/'project.json',value)
        url=f"/api/projects/{self.project['id']}/clips"
        response=self.client.put(url,json={'clips':[dict(id='000000000001',text='您好',enabled=False)]})
        self.assertEqual(response.json()['clips'][0]['duration'],2)
        self.assertFalse(response.json()['clips'][0]['enabled'])
        self.assertEqual(self.client.put(url,json={'clips':[dict(id='000000000001',text='坏|文本')]}).status_code,400)

    def test_saving_traditional_text_normalizes_to_simplified(self):
        folder=studio.PROJECTS/self.project['id']; value=studio.read(folder/'project.json')
        value['clips']=[dict(id='000000000001',text='旧文本',duration=2,enabled=True)]
        studio.save(folder/'project.json',value)
        response=self.client.put(f"/api/projects/{self.project['id']}/clips",json={'clips':[dict(id='000000000001',text='這是聲音訓練',enabled=True)]})
        self.assertEqual(response.json()['clips'][0]['text'],'这是声音训练')
        self.assertEqual(studio.read(folder/'project.json')['clips'][0]['text'],'这是声音训练')

    def test_archive_path_traversal_rejected(self):
        root=Path(self.temp.name); archive=root/'bad.tar.bz2'; job_dir=root/'job'; job_dir.mkdir()
        with tarfile.open(archive,'w:bz2') as tar:
            info=tarfile.TarInfo('../escaped.txt'); info.size=4; tar.addfile(info,io.BytesIO(b'evil'))
        with self.assertRaisesRegex(ValueError,'不安全'):
            worker.import_model({'options':{'path':str(archive)}},job_dir)
        self.assertFalse((root/'escaped.txt').exists())


class NumericalTests(unittest.TestCase):
    def test_numba_alignment_matches_exhaustive_search(self):
        import numpy as np
        import torch
        import vits_engine
        from vits_fast_fine_tuning.monotonic_align import maximum_path
        random=np.random.default_rng(42)
        for frames,texts in ((5,2),(7,3),(4,4)):
            scores=random.normal(size=(frames,texts)).astype(np.float32)
            candidates=[]
            for transitions in itertools.combinations(range(1,frames),texts-1):
                indices=np.searchsorted(transitions,np.arange(frames),side='right')
                candidates.append((sum(scores[y,indices[y]] for y in range(frames)),indices))
            best=max(candidates,key=lambda c:c[0])[0]
            result=maximum_path(torch.tensor(scores[None]),torch.ones(1,frames,texts)).numpy()[0]
            self.assertTrue(np.all(result.sum(axis=1)==1))
            self.assertAlmostEqual(float((result*scores).sum()),float(best),places=5)

    def test_dynamic_relative_embeddings_equal_original_formula(self):
        import torch
        import torch.nn.functional as F
        import vits_engine
        from vits_fast_fine_tuning.attentions import MultiHeadAttention
        layer=MultiHeadAttention(16,16,2,window_size=4)
        for length in (1,3,5,9,31):
            padding=max(length-5,0); start=max(5-length,0)
            expected=F.pad(layer.emb_rel_k,(0,0,padding,padding))[:,start:start+2*length-1]
            actual=layer._get_relative_embeddings(layer.emb_rel_k,length)
            torch.testing.assert_close(actual,expected)


if __name__=='__main__':
    unittest.main()
