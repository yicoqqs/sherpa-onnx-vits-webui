const $ = s => document.querySelector(s);
const selectedModels = new Set();
const state = {project:null, models:[], job:null, timer:null, polling:false, dirty:false, clipView:{project:null,group:null,pages:{}}};
const escapeHtml = v => String(v ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(message,error=false){const t=$('#toast');t.textContent=message;t.className=error?'error':'';t.hidden=false;clearTimeout(state.timer);state.timer=setTimeout(()=>t.hidden=true,6500);}
async function api(url,options={}){const response=await fetch(url,options);if(!response.ok){let data;try{data=await response.json();}catch{data={detail:response.statusText};}throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));}return response.json();}
const json=(method,data)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
let speakerBase='', speakerRequest=0;
function updateTrainingMode(){
 const mode=$('#training-mode').value, protect=$('#protect-existing').checked;
 $('#reference-label').textContent=mode==='modify'?'要修改的音色':'参考音色';
 $('#protect-label').hidden=mode==='single';
 const description={add:'复制参考音色为新的音色编号，保留原有音色。',modify:'用当前项目录音训练所选音色编号，保留其他音色。',single:'从参考音色开始完整微调，只输出当前项目的一个音色。'}[mode];
 $('#training-mode-help').textContent=description+(mode==='single'?'':protect?' 冻结共享参数，仅调整目标音色嵌入；保护其他音色，但对目标声音的拟合能力有限。':' 完整微调会改变共享参数，其他音色可能变化。')+' 结果另存，原始底模保持不变。切换设置后请新建项目或取消继续训练。';
}
async function loadBaseSpeakers(preferred=null){
 const base=$('#base').value.trim(), request=++speakerRequest;
 const previous=preferred??(speakerBase===base?$('#reference-speaker').value:null);
 speakerBase='';$('#reference-speaker').disabled=true;
 $('#reference-speaker').innerHTML='<option value="">正在读取音色…</option>';
 try{
  if(!base)throw new Error('请填写训练底模目录');
  const result=await api('/api/base/speakers?'+new URLSearchParams({base}));
  if(request!==speakerRequest||base!==$('#base').value.trim())return;
  $('#reference-speaker').innerHTML=result.speakers.map(s=>`<option value="${s.id}">${s.id} · ${escapeHtml(s.name||'未命名音色')}</option>`).join('');
  if(previous!=null&&result.speakers.some(s=>String(s.id)===String(previous)))$('#reference-speaker').value=previous;
  $('#reference-speaker').disabled=false;speakerBase=base;
 }catch(error){
  if(request===speakerRequest)$('#reference-speaker').innerHTML='<option value="">读取失败，请检查底模目录</option>';
  throw error;
 }
}
async function restoreTrainingSettings(){
 const settings=state.project?.training_settings;
 $('#training-mode').value=settings?.mode||'add';
 $('#protect-existing').checked=settings?.protect_existing??true;
 if(settings?.base)$('#base').value=settings.base;
 updateTrainingMode();
 if($('#base').value.trim())await loadBaseSpeakers(settings?.reference_speaker??0).catch(e=>toast(e.message,true));
}
$('#base').addEventListener('input',()=>{++speakerRequest;speakerBase='';$('#reference-speaker').disabled=true;$('#reference-speaker').innerHTML='<option value="">目录已改变，请读取底模音色</option>';});
$('#base').addEventListener('change',()=>loadBaseSpeakers().catch(e=>toast(e.message,true)));
$('#training-mode').addEventListener('change',updateTrainingMode);
$('#protect-existing').addEventListener('change',updateTrainingMode);
updateTrainingMode();
function action(selector,fn){$(selector).addEventListener('click',async()=>{const b=$(selector);b.disabled=true;try{await fn();}catch(e){toast(e.message,true);}finally{b.disabled=selector==='#delete-project'?(!state.project||state.job?.status==='running'):["#export","#export-base","#synthesize","#delete-selected-models"].includes(selector)&&state.job?.status==='running';if(selector==='#delete-selected-models')updateBulkModels();}});}
function page(name){document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!==`page-${name}`);document.querySelectorAll('.nav').forEach(n=>n.classList.toggle('active',n.dataset.page===name));$('#page-title').textContent={create:'声音训练',listen:'试听与导出',settings:'运行环境'}[name];localStorage.setItem('vits.page',name);}
document.querySelectorAll('.nav').forEach(n=>n.addEventListener('click',()=>page(n.dataset.page)));
function selected(){if(!state.project)throw new Error('请先创建或选择一个音色项目');return state.project.id;}
async function loadProjects(){const items=await api('/api/projects');const previous=state.project?.id||localStorage.getItem('vits.project');$('#project').innerHTML=items.length?items.map(p=>`<option value="${p.id}">${escapeHtml(p.name)}</option>`).join(''):'<option value="">新建你的第一个音色</option>';if(items.length){$('#project').value=items.some(p=>p.id===previous)?previous:items[0].id;await loadProject();}else{clearProject();}}
async function loadProject(){const id=$('#project').value;if(!id)return;state.project=await api(`/api/projects/${id}`);localStorage.setItem('vits.project',id);$('#material-search').value='';renderProject();await restoreTrainingSettings();}
$('#project').addEventListener('change',async()=>{if(state.dirty&&!confirm('校对尚未保存，仍要切换项目吗？')){$('#project').value=state.project.id;return;}try{await loadProject();}catch(e){toast(e.message,true);}});
function groupClips(project){
 const uploads=project.uploads||[], groups=new Map();
 project.clips.forEach((clip,index)=>{
  const matches=uploads.filter(upload=>upload.name===clip.source);
  const upload=clip.source_file?uploads.find(upload=>upload.file===clip.source_file):(matches.length===1?matches[0]:null);
  const key=clip.source_file?'file:'+clip.source_file:upload?'file:'+upload.file:'legacy:'+(clip.source||'');
  if(!groups.has(key))groups.set(key,{key,name:upload?.name||clip.source||'未标注来源的素材',file:upload?.file||'',clips:[]});
  groups.get(key).clips.push({clip,index});
 });
 return [...groups.values()];
}
function renderProject(){const p=state.project;if(p.base_model){$('#base').value=p.base_model;localStorage.setItem('vits.base',p.base_model);}$('#delete-project').disabled=state.job?.status==='running';$('#export-project').textContent='当前音色：'+p.name;state.dirty=false;$('#save-clips').textContent='保存校对';renderUploads();$('#clip-count').textContent=p.clips.length;renderClips();updateSummary();renderPrepareProgress(state.job);renderMediaProgress(state.job);renderTrainProgress(state.job);}
function renderUploads(){
 const uploads=state.project?.uploads||[], query=$('#material-search').value.trim().toLowerCase();
 const filtered=uploads.filter(f=>f.name.toLowerCase().includes(query));
 $('#material-count').textContent=query?`${filtered.length} / ${uploads.length} 个素材`:`${uploads.length} 个素材`;
 const processed=new Set(state.project?.processed_uploads||[]);
 (state.project?.clips||[]).forEach(c=>{if(c.source_file)processed.add(c.source_file);});
 $('#uploads').innerHTML=filtered.map(f=>{const clips=(state.project.clips||[]).filter(c=>c.source_file===f.file||(!c.source_file&&c.source===f.name));const ready=processed.has(f.file)||clips.length>0;
 return `<div class="file-item"><div class="material-name" title="${escapeHtml(f.name)}"><strong>${escapeHtml(f.name)}</strong><small>${(f.bytes/1024/1024).toFixed(1)} MB · ${ready?`${clips.length} 个片段`:'待切分'}</small></div><button class="delete-material" data-file="${escapeHtml(f.file)}" aria-label="删除素材 ${escapeHtml(f.name)}" ${state.job?.status==='running'?'disabled':''}>删除</button></div>`;}).join('')||`<div class="empty-small">${query?'没有匹配的素材':'还没有录音，先添加你的声音素材。'}</div>`;
}
$('#material-search').addEventListener('input',renderUploads);
function updateSummary(){const clips=state.project?.clips||[];const enabled=clips.filter(c=>c.enabled!==false&&c.text.trim());const seconds=enabled.reduce((n,c)=>n+c.duration,0);$('#data-summary').innerHTML=`<span class="summary-icon">≋</span><div><strong>${enabled.length?`${enabled.length} 个可用片段 · ${(seconds/60).toFixed(1)} 分钟`:'等待准备训练数据'}</strong><small>${seconds&&seconds<180?'素材不足 3 分钟，建议补充录音以改善效果。':'校对并保存后，即可开始训练。'}</small></div>`;}
function renderClips(){
 const saveButton=$('#save-clips');
 const p=state.project, view=state.clipView;
 if(view.project!==p.id){view.project=p.id;view.group=null;view.pages={};}
 const groups=groupClips(p);
 if(!groups.length){$('#clips').innerHTML='<div class="empty-state"><strong>暂无录音片段，请先上传并切分素材。</strong></div>';if(saveButton)$('#clips').prepend(saveButton);return;}
 const group=groups.find(g=>g.key===view.group)||groups[0];view.group=group.key;
 const total=Math.ceil(group.clips.length/20), page=Math.min(Math.max(view.pages[group.key]||1,1),total);view.pages[group.key]=page;
 const start=(page-1)*20;
 const pager=()=>`<div class="clip-pagination"><span>第 ${start+1}–${Math.min(start+20,group.clips.length)} 条，共 ${group.clips.length} 条 · 每页 20 条</span><div><button class="button secondary small" data-clip-page="${page-1}" ${page===1?'disabled':''}>上一页</button><label>第 <select aria-label="选择片段页码" class="clip-page-select">${Array.from({length:total},(_,i)=>`<option value="${i+1}" ${i+1===page?'selected':''}>${i+1}</option>`).join('')}</select> / ${total} 页</label><button class="button secondary small" data-clip-page="${page+1}" ${page===total?'disabled':''}>下一页</button></div></div>`;
 $('#clips').innerHTML=`<div class="clip-workspace clip-dropdown-workspace"><div class="clip-source-picker"><label for="clip-source-select">选择来源素材（${groups.length} 个）</label><div class="clip-source-controls"><button type="button" class="button secondary small" data-source-step="-1" ${groups.indexOf(group)===0?'disabled':''} aria-label="上一个素材">↑ 上一个</button><select id="clip-source-select" title="${escapeHtml(group.name)}">${groups.map((g,i)=>`<option value="${escapeHtml(g.key)}" ${g.key===group.key?'selected':''}>${i+1}. ${escapeHtml(g.name)} · ${g.clips.length} 个片段</option>`).join('')}</select><button type="button" class="button secondary small" data-source-step="1" ${groups.indexOf(group)===groups.length-1?'disabled':''} aria-label="下一个素材">↓ 下一个</button><button type="button" class="button secondary small danger delete-source-material" data-file="${escapeHtml(group.file)}" ${state.job?.status==='running'||!group.file?'disabled':''} title="${group.file?'删除当前素材及其关联切片，可撤销':'此旧片段未关联到素材，请在上方素材管理中处理'}">删除当前素材</button></div><div class="clip-source-current">${groups.indexOf(group)+1} / ${groups.length} · ${escapeHtml(group.name)} · ${group.clips.length} 个片段</div></div><section id="clip-panel" aria-label="录音切片">${pager()}<div class="clip-grid">${group.clips.slice(start,start+20).map(({clip:c,index:i})=>`<div class="clip" data-id="${escapeHtml(c.id)}"><input type="checkbox" aria-label="启用片段 ${i+1}" ${c.enabled!==false?'checked':''}><span class="clip-number">${String(i+1).padStart(2,'0')}</span><audio controls preload="none" src="/api/audio/${p.id}/${c.id}"></audio><input type="text" aria-label="片段 ${i+1} 转写" value="${escapeHtml(c.text)}"><button class="delete-clip" aria-label="删除片段 ${i+1}" ${state.job?.status==='running'?'disabled':''}>删除</button></div>`).join('')}</div>${pager()}</section></div>`;
 if(saveButton)$('#clips .clip-source-controls').append(saveButton);

}
$('#clips').addEventListener('input',e=>{const row=e.target.closest('.clip');if(!row)return;const clip=state.project.clips.find(c=>c.id===row.dataset.id);if(!clip)return;clip.text=row.querySelector('input[type=text]').value;clip.enabled=row.querySelector('input[type=checkbox]').checked;state.dirty=true;$('#save-clips').textContent='保存校对 · 未保存';updateSummary();});
$('#clips').addEventListener('click',e=>{const button=e.target.closest('[data-clip-page]');if(button){state.clipView.pages[state.clipView.group]=Number(button.dataset.clipPage);renderClips();}});
$('#clips').addEventListener('click',e=>{
 const button=e.target.closest('[data-source-step]');if(!button||button.disabled)return;
 const groups=groupClips(state.project),index=groups.findIndex(g=>g.key===state.clipView.group);
 const next=index+Number(button.dataset.sourceStep);if(next<0||next>=groups.length)return;
 state.clipView.group=groups[next].key;renderClips();
 $('#clip-source-select').focus({preventScroll:true});
});
$('#clips').addEventListener('change',e=>{if(e.target.id==='clip-source-select'){state.clipView.group=e.target.value;renderClips();$('#clip-source-select').focus({preventScroll:true});}});
$('#clips').addEventListener('change',e=>{if(e.target.matches('.clip-page-select')){state.clipView.pages[state.clipView.group]=Number(e.target.value);renderClips();}});

action('#new-project',async()=>{$('#new-project-form').hidden=!$('#new-project-form').hidden;$('#project-name').focus();});
action('#create-project',async()=>{const name=$('#project-name').value.trim();if(!name)throw new Error('请填写音色名称');state.project=await api('/api/projects',json('POST',{name}));await loadProjects();$('#new-project-form').hidden=true;toast('音色项目已创建');});

function clearProject(){const saveButton=$('#save-clips');if(saveButton)document.querySelector('.clips-card .card-heading').append(saveButton);state.project=null;state.dirty=false;$('#material-search').value='';$('#material-count').textContent='0 个素材';localStorage.removeItem('vits.project');$('#delete-project').disabled=true;$('#uploads').innerHTML='<div class="empty-small">还没有录音，先创建音色并添加素材。</div>';$('#clips').innerHTML='<div class="empty-state"><strong>先创建一个音色项目</strong></div>';$('#clip-count').textContent='0';$('#save-clips').textContent='保存校对';$('#export-project').textContent='当前音色：未选择';updateSummary();renderPrepareProgress(state.job);renderMediaProgress(state.job);renderTrainProgress(state.job);}
function rememberDeletion(token){localStorage.setItem('vits.deleted',token);$('#undo-delete').hidden=false;}
action('#delete-project',async()=>{const key=selected();if(!confirm(`删除音色「${state.project.name}」及其素材、片段和训练检查点？\n将移入本地回收目录，可撤销。已导出的模型仍会保留。`))return;const result=await api(`/api/projects/${key}`,{method:'DELETE'});rememberDeletion(result.undo);clearProject();await loadProjects();toast('音色已移入回收目录');});
action('#undo-delete',async()=>{const token=localStorage.getItem('vits.deleted');if(!token)return;const project=await api(`/api/trash/${token}/restore`,{method:'POST'});localStorage.removeItem('vits.deleted');$('#undo-delete').hidden=true;state.project=project;localStorage.setItem('vits.project',project.id);await loadProjects();toast('删除已撤销');});
function confirmRemoval(message){
 return new Promise(resolve=>{
  const dialog=document.createElement('dialog');dialog.className='delete-dialog';
  dialog.setAttribute('aria-labelledby','delete-dialog-title');dialog.setAttribute('aria-describedby','delete-dialog-description');
  const header=document.createElement('div');header.className='delete-dialog-header';
  const title=document.createElement('h2');title.id='delete-dialog-title';title.textContent='确认删除';
  const subtitle=document.createElement('p');subtitle.textContent='请确认要移除的内容';header.append(title,subtitle);
  const description=document.createElement('p');description.id='delete-dialog-description';description.className='delete-dialog-description';description.textContent=message;
  const note=document.createElement('div');note.className='delete-dialog-note';note.textContent='删除后将移入回收目录，可通过「撤销上次删除」恢复。';
  const actions=document.createElement('div');actions.className='delete-dialog-actions';
  const cancel=document.createElement('button');cancel.type='button';cancel.className='button secondary';cancel.textContent='取消';
  const accept=document.createElement('button');accept.type='button';accept.className='button delete-dialog-confirm';accept.textContent='确认删除';
  let finished=false;
  const finish=value=>{if(finished)return;finished=true;dialog.close();dialog.remove();resolve(value);};
  cancel.onclick=()=>finish(false);accept.onclick=()=>finish(true);
  dialog.addEventListener('cancel',e=>{e.preventDefault();finish(false);});
  actions.append(cancel,accept);dialog.append(header,description,note,actions);document.body.append(dialog);dialog.showModal();cancel.focus();
 });
}
async function removeItem(button,kind,id){
 const key=selected();
 const name=kind==='uploads'?(state.project.uploads||[]).find(f=>f.file===id)?.name:'当前片段';
 const scope=kind==='uploads'?'该素材及其关联切分片段':'该片段';
 if(!await confirmRemoval(`删除「${name}」？\n${scope}将移入回收目录，可撤销。${state.dirty?'未保存的校对将丢失。':''}`))return;
 button.disabled=true;toast('正在删除…');
 try{const result=await api(`/api/projects/${key}/${kind}/${encodeURIComponent(id)}`,{method:'DELETE'});rememberDeletion(result.undo);state.project=result.project;renderProject();toast('已删除，可点击「撤销上次删除」恢复');}catch(e){toast(e.message,true);button.disabled=false;}
}
$('#uploads').addEventListener('click',e=>{const button=e.target.closest('.delete-material');if(button)removeItem(button,'uploads',button.dataset.file).catch(e=>toast(e.message,true));});
$('#clips').addEventListener('click',e=>{const button=e.target.closest('.delete-source-material');if(button&&!button.disabled&&button.dataset.file)removeItem(button,'uploads',button.dataset.file).catch(e=>toast(e.message,true));});
$('#clips').addEventListener('click',e=>{const button=e.target.closest('.delete-clip');if(button)removeItem(button,'clips',button.closest('.clip').dataset.id).catch(e=>toast(e.message,true));});
$('#undo-delete').hidden=!localStorage.getItem('vits.deleted');

async function uploadFiles(files){if(!files.length)return;const key=selected();const data=new FormData();for(const file of files)data.append('files',file);$('#dropzone').style.opacity='.5';toast('正在上传录音…');try{state.project=await api(`/api/projects/${key}/upload`,{method:'POST',body:data});renderProject();toast('录音上传完成');}finally{$('#dropzone').style.opacity='';$('#files').value='';}}
$('#files').addEventListener('change',()=>uploadFiles([...$('#files').files]).catch(e=>toast(e.message,true)));
$('#dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#files').click();}});
for(const event of ['dragenter','dragover'])$('#dropzone').addEventListener(event,e=>{e.preventDefault();$('#dropzone').classList.add('dragover');});
for(const event of ['dragleave','drop'])$('#dropzone').addEventListener(event,e=>{e.preventDefault();$('#dropzone').classList.remove('dragover');});
$('#dropzone').addEventListener('drop',e=>uploadFiles([...e.dataTransfer.files]).catch(e=>toast(e.message,true)));
async function saveClips(){const clips=state.project.clips.map(c=>({id:c.id,text:c.text,enabled:c.enabled!==false}));state.project=await api(`/api/projects/${selected()}/clips`,json('PUT',{clips}));state.dirty=false;$('#save-clips').textContent='保存校对';const saved=new Map(state.project.clips.map(c=>[c.id,c.text]));document.querySelectorAll('.clip').forEach(row=>{if(saved.has(row.dataset.id))row.querySelector('input[type=text]').value=saved.get(row.dataset.id);});updateSummary();}
action('#save-clips',async()=>{selected();await saveClips();toast('校对已保存');});
const kinds={video:'视频素材导入',download_base:'推荐底模下载',prepare:'录音切分与转写',train:'VITS 音色训练',export:'ONNX 模型导出',export_base:'底模导出',import:'导入语音包',synthesize:'文本试听'};
const statuses={running:'运行中',completed:'已完成',failed:'失败',cancelled:'已停止',interrupted:'已中断'};
async function watch(job){state.job=job;renderJob(job);await pollJob();}
function renderJob(j){const pause=$('#pause-train');pause.hidden=j.kind!=='train'||j.status!=='running';pause.textContent=j.training_paused?'恢复训练':j.pause_requested?'取消暂停（等待本轮保存）':'暂停训练';pause.disabled=false;$('#delete-project').disabled=!state.project||j.status==='running';document.querySelectorAll('.delete-material,.delete-clip,.delete-model').forEach(b=>b.disabled=j.status==='running');document.querySelectorAll('.delete-source-material').forEach(b=>b.disabled=j.status==='running'||!b.dataset.file);renderPrepareProgress(j);renderMediaProgress(j);renderTrainProgress(j);renderTaskProgress(j,'export');renderTaskProgress(j,'base-export');const baseLink=$('#base-export-download');baseLink.hidden=!(j.kind==='export_base'&&j.status==='completed'&&j.result?.model_id);if(!baseLink.hidden)baseLink.href=`/api/models/${j.result.model_id}/download`;$('#export-base').disabled=j.status==='running';renderTaskProgress(j,'synthesize');$('#export').disabled=j.status==='running';$('#synthesize').disabled=j.status==='running';updateBulkModels();$('#task-state').textContent=statuses[j.status]||j.status;$('#task-dot').className=`status-dot ${j.status}`;$('#task-description').textContent=`${kinds[j.kind]} · ${new Date(j.created*1000).toLocaleString()}${j.status==='failed'?' · 请查看下方日志中的原因':''}`;$('#cancel').disabled=j.status!=='running';if(j.log){const log=$('#log');const nearBottom=log.scrollHeight-log.scrollTop-log.clientHeight<60;log.textContent=j.log.split('\n').filter(line=>!line.startsWith('@@VITS_PROGRESS@@')&&!line.startsWith('@@MEDIA_PROGRESS@@')&&!line.startsWith('@@TASK_PROGRESS@@')).join('\n');if(nearBottom)log.scrollTop=log.scrollHeight;}else $('#log').textContent='正在启动任务…';}

function modelTimestamp(seconds){return new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(new Date(seconds*1000));}
function taskProgress(job){const lines=(job.log||'').split('\n');for(let i=lines.length-1;i>=0;i--){if(lines[i].startsWith('@@TASK_PROGRESS@@')){try{return JSON.parse(lines[i].slice('@@TASK_PROGRESS@@'.length));}catch{}}}return null;}
function renderTaskProgress(job,kind){
 const panel=$(`#${kind}-progress`);if(!panel)return;
 if(!job||(job.kind!==kind&&!(kind==='base-export'&&job.kind==='export_base'))){panel.hidden=true;return;}panel.hidden=false;
 const p=taskProgress(job), running=job.status==='running';
 const percent=job.status==='completed'?100:(p?.percent==null?null:Math.max(0,Math.min(99,p.percent)));
 const stages={loading:'正在加载模型',lexicon:'正在生成符号表与词典',onnx:'正在转换 ONNX 模型',resources:'正在整理语音包资源',quantizing:'正在生成 INT8 模型',validating:'正在验证 ONNX 推理',validating_package:'正在验证 sherpa-onnx 加载',copying:'正在复制模型文件',packing:'正在压缩语音包',generating:'正在生成语音',saving:'正在保存试听音频',completed:'正在完成任务'};
 $(`#${kind}-stage`).textContent=running?(stages[p?.stage]||'正在启动任务'):({completed:kind==='synthesize'?'试听生成完成':'转换打包完成',failed:'任务失败，请查看日志',cancelled:'任务已停止',interrupted:'任务已中断'}[job.status]||job.status);
 $(`#${kind}-percent`).textContent=percent==null?(running?'处理中…':'—'):`${percent}%`;
 const bar=$(`#${kind}-bar`);bar.classList.toggle('indeterminate',running&&percent==null);bar.classList.toggle('stopped',!running&&job.status!=='completed');
 if(percent==null)bar.removeAttribute('aria-valuenow');else bar.setAttribute('aria-valuenow',String(percent));
 bar.setAttribute('aria-valuetext',$(`#${kind}-stage`).textContent);
 $(`#${kind}-fill`).style.width=percent==null?(running?'35%':'0%'):`${percent}%`;
 $(`#${kind}-detail`).textContent=`${kind==='export'?'阶段进度 · ':''}${running?'已用时':'用时'} ${formatDuration((job.finished||Date.now()/1000)-job.created)}${p?.duration?' · 音频 '+formatDuration(p.duration):''}`;
}
function clearPlayer(){const audio=$('#output-audio');audio.pause();audio.removeAttribute('src');audio.load();$('#player').hidden=true;}
function formatDuration(seconds){seconds=Math.max(0,Math.floor(seconds||0));return seconds>=60?`${Math.floor(seconds/60)} 分 ${seconds%60} 秒`:`${seconds} 秒`;}
function renderPrepareProgress(job){
  const panel=$('#prepare-progress');
 if(!panel)return;
  if(!job||job.kind!=='prepare'||job.project_id!==state.project?.id){panel.hidden=true;return;}
  panel.hidden=false;
  const lines=(job.log||'').split('\n');let progress=null;
  for(let i=lines.length-1;i>=0;i--){if(lines[i].startsWith('@@VITS_PROGRESS@@')){try{progress=JSON.parse(lines[i].slice('@@VITS_PROGRESS@@'.length));break;}catch{}}}
  // Tasks started before progress reporting was added remain visible without inventing a percentage.
  if(!progress){const matches=[...(job.log||'').matchAll(/片段 (\d+) ·/g)];progress={stage:(job.log||'').includes('正在处理：')?'transcribing':'loading',percent:null,clips:matches.length?Number(matches[matches.length-1][1]):0};}
  const running=job.status==='running';
  const labels={separating:'正在分离人声与背景音乐',downloading:'正在下载转写模型',loading:'正在加载转写模型',decoding:'正在解码录音',transcribing:'正在识别语音并切分',saving:'正在保存录音片段',completed:'切分与转写完成'};
  const terminal={completed:'切分与转写完成',failed:'处理失败，请查看任务日志',cancelled:'处理已停止',interrupted:'处理已中断'};
  const percent=job.status==='completed'?100:progress.percent;
  $('#prepare-stage').textContent=running?(labels[progress.stage]||'正在准备'):terminal[job.status]||job.status;
  $('#prepare-percent').textContent=percent==null?(running?'处理中…':'—'):`${percent}%${progress.stage==='downloading'?' · 当前文件':''}`;
  const bar=$('#prepare-bar');bar.classList.toggle('indeterminate',running&&percent==null);bar.classList.toggle('stopped',!running&&job.status!=='completed');
  if(percent==null)bar.removeAttribute('aria-valuenow');else bar.setAttribute('aria-valuenow',percent);
  bar.setAttribute('aria-valuetext',$('#prepare-stage').textContent+(percent==null?'':`，${percent}%`));
  $('#prepare-fill').style.width=percent==null?(running?'35%':'0%'):`${percent}%`;
  const details=[];
  if(progress.file_index)details.push(`录音 ${progress.file_index}/${progress.files_total}`);
  if(progress.file)details.push(progress.file);
  if(progress.total_seconds!=null)details.push(`已处理 ${formatDuration(progress.processed_seconds)} / ${formatDuration(progress.total_seconds)}`);
  if(progress.stage==='downloading'){if(progress.total_bytes)details.push(`${(progress.downloaded_bytes/1024**2).toFixed(1)} / ${(progress.total_bytes/1024**2).toFixed(1)} MB`);}else details.push(`已生成 ${progress.clips||0} 个片段`);
  $('#prepare-detail').textContent=details.join(' · ');
  const elapsed=formatDuration((job.finished||Date.now()/1000)-job.created);
  $('#prepare-timing').textContent=`${running?'已用时':'用时'} ${elapsed}${running&&['loading','downloading'].includes(progress.stage)?' · 首次使用需要下载模型，可复用已下载的缓存':''}`;
}


function trainingProgress(job){
 const log=job.log||'';
 const resume=log.match(/继续上次训练：已完成 (\d+) 轮/);
 const start=resume?Number(resume[1]):0;
 const matches=[...log.matchAll(/Epoch (\d+)\/(\d+) · Step (\d+)\/(\d+)/g)];
 const last=matches[matches.length-1];
 if(!last)return null;
 const [,epoch,total,step,steps]=last.map(Number);
 const rounds=total-start;
 if(rounds<=0||steps<=0)return null;
 const done=(epoch-start-1)*steps+step;
 return {epoch:epoch-start,rounds,step,steps,percent:Math.max(0,Math.min(99,Math.floor(done/(rounds*steps)*100)))};
}
function renderTrainProgress(job){
 const panel=$('#train-progress');
 if(!panel)return;
 if(!job||job.kind!=='train'||job.project_id!==state.project?.id){panel.hidden=true;return;}
 panel.hidden=false;
 const running=job.status==='running', progress=trainingProgress(job);
 const percent=job.status==='completed'?100:progress?.percent;
 const labels={completed:'训练完成',failed:'训练失败，请查看任务日志',cancelled:'训练已停止',interrupted:'训练已中断'};
 $('#train-stage').textContent=running?(job.training_paused?'训练已暂停，可恢复':job.pause_requested?'等待当前轮保存后暂停':progress?'正在训练音色':'正在加载模型与准备训练数据'):labels[job.status]||job.status;
 $('#train-percent').textContent=percent==null?(running?'准备中…':'—'):`${percent}%`;
 const bar=$('#train-bar');bar.classList.toggle('indeterminate',running&&percent==null);bar.classList.toggle('stopped',!running&&job.status!=='completed');
 if(percent==null)bar.removeAttribute('aria-valuenow');else bar.setAttribute('aria-valuenow',percent);
 bar.setAttribute('aria-valuetext',$('#train-stage').textContent+(percent==null?'':`，${percent}%`));
 $('#train-fill').style.width=percent==null?(running?'35%':'0%'):`${percent}%`;
 $('#train-detail').textContent=progress?`本次第 ${progress.epoch} / ${progress.rounds} 轮 · 当前轮第 ${progress.step} / ${progress.steps} 步`:'首次更新进度需要等待模型加载和首个训练步骤完成。';
 $('#train-timing').textContent=`${running?'已用时':'用时'} ${formatDuration((job.finished||Date.now()/1000)-job.created)}${running&&progress?.percent===99?' · 正在完成训练并保存检查点':''}`;
}

function renderMediaProgress(job){
 const sourceLink=$('#video-source-download');if(sourceLink)sourceLink.hidden=true;
 const panel=$('#video-progress');
 if(!panel)return;
 if(!job||job.kind!=='video'||job.project_id!==state.project?.id){panel.hidden=true;return;}
 panel.hidden=false;
 if(sourceLink&&(job.status==='completed'||(job.log||'').includes('@@MEDIA_PROGRESS@@{"stage": "extracting"'))){sourceLink.hidden=false;sourceLink.href=`/api/jobs/${job.id}/video`;}
 let progress={stage:'resolving'};const lines=(job.log||'').split('\n');
 for(let i=lines.length-1;i>=0;i--){if(lines[i].startsWith('@@MEDIA_PROGRESS@@')){try{progress=JSON.parse(lines[i].slice('@@MEDIA_PROGRESS@@'.length));break;}catch{}}}
 const running=job.status==='running';const percent=job.status==='completed'?100:progress.percent;
 const labels={resolving:'正在解析视频链接',downloading:'正在下载视频素材',extracting:'正在提取音频',completed:'音频已加入素材列表'};
 $('#video-stage').textContent=running?labels[progress.stage]||'正在处理':({completed:labels.completed,failed:'导入失败，请查看任务日志',cancelled:'导入已停止',interrupted:'导入已中断'}[job.status]||job.status);
 $('#video-percent').textContent=percent==null?(running?'处理中…':'—'):`${percent}%`;
 const bar=$('#video-bar');bar.classList.toggle('indeterminate',running&&percent==null);bar.classList.toggle('stopped',!running&&job.status!=='completed');
 if(percent==null)bar.removeAttribute('aria-valuenow');else bar.setAttribute('aria-valuenow',percent);
 $('#video-fill').style.width=percent==null?(running?'35%':'0%'):`${percent}%`;
 const details=[];if(progress.title)details.push(progress.title);if(progress.downloaded_bytes!=null)details.push(`${(progress.downloaded_bytes/1024**2).toFixed(1)}${progress.total_bytes?' / '+(progress.total_bytes/1024**2).toFixed(1):''} MB`);if(progress.duration)details.push(`音频 ${formatDuration(progress.duration)}`);details.push(`用时 ${formatDuration((job.finished||Date.now()/1000)-job.created)}`);$('#video-detail').textContent=details.join(' · ');
}
action('#import-video',async()=>{const key=selected();const url=$('#video-url').value.trim();if(!url)throw new Error('请先粘贴视频链接');await watch(await api(`/api/projects/${key}/video`,json('POST',{url,cookies_path:$('#video-cookies').value.trim()})));});

async function pollJob(){if(state.polling||!state.job)return;state.polling=true;try{const previous=state.job;const j=await api(`/api/jobs/${state.job.id}`);state.job=j;renderJob(j);if(j.status!=='running'&&previous.status==='running'){if(j.status==='completed'){toast(`${kinds[j.kind]}已完成`);if(['prepare','video'].includes(j.kind)&&state.project?.id===j.project_id)await loadProject();if(['export','export_base','import'].includes(j.kind))await loadModels(j.result?.model_id);if(j.kind==='download_base')await loadEnvironment(j.result?.base);if(j.kind==='synthesize'){await loadHistory();$('#player').hidden=false;$('#output-audio').src=`/api/jobs/${j.id}/audio`;$('#audio-download').href=`/api/jobs/${j.id}/audio`;}}else if(j.status==='failed')toast('任务失败，请查看日志',true);}}catch(e){$('#task-description').textContent=`读取任务状态失败：${e.message}，稍后重试`; }finally{state.polling=false;}}
setInterval(()=>{if(state.job?.status==='running')pollJob();},1600);
action('#prepare',async()=>{const key=selected();if(state.dirty)await saveClips();if($('#prepare-mode').value==='all'&&state.project.clips.length&&!confirm('重新切分会替换现有片段和校对内容，是否继续？'))return;await watch(await api(`/api/projects/${key}/prepare`,json('POST',{whisper:$('#whisper').value,language:'zh',mode:$('#prepare-mode').value,separate_vocals:$('#separate-vocals').checked})));});
action('#download-base',async()=>{await watch(await api('/api/base/download',json('POST',{variant:'C'})));});
action('#load-base-speakers',()=>loadBaseSpeakers());
action('#export-base',async()=>{const base=$('#base').value.trim();if(!base)throw new Error('请填写训练底模目录');await watch(await api('/api/base/export',json('POST',{base,quantize:$('#base-quantize').checked})));});
action('#train',async()=>{const key=selected();if(speakerBase!==$('#base').value.trim())await loadBaseSpeakers();if($('#reference-speaker').disabled||$('#reference-speaker').value==='')throw new Error('请先读取并选择底模音色');await saveClips();await watch(await api(`/api/projects/${key}/train`,json('POST',{base:$('#base').value.trim(),epochs:Number($('#epochs').value),batch_size:Number($('#batch').value),learning_rate:Number($('#lr').value),resume:$('#resume').checked,mode:$('#training-mode').value,reference_speaker:Number($('#reference-speaker').value),protect_existing:$('#protect-existing').checked})));localStorage.setItem('vits.base',$('#base').value);});
action('#cancel',async()=>{await api(`/api/jobs/${state.job.id}/cancel`,{method:'POST'});await pollJob();});
action('#export',async()=>{await watch(await api(`/api/projects/${selected()}/export`,json('POST',{quantize:$('#quantize').checked,name:$('#export-name').value.trim()})));});

function updateSpeakerOptions(){
 const model=state.models.find(m=>m.id===$('#model').value);
 const speakers=model?.speakers?.length?model.speakers:[{id:0,name:''}];
 const options=$('#speaker-select');
 options.innerHTML=speakers.map(s=>`<option value="${s.id}">${escapeHtml(s.name?`${s.id} · ${s.name}`:String(s.id))}</option>`).join('');
 $('#speaker').min='0';$('#speaker').max=String(speakers[speakers.length-1].id);
 const id=Number($('#speaker').value);
 if(!Number.isInteger(id)||!speakers.some(s=>s.id===id))$('#speaker').value=String(speakers[0].id);
 options.value=$('#speaker').value;
}
$('#model').addEventListener('change',()=>{clearPlayer();updateSpeakerOptions();updateBulkModels();});
$('#speaker-select').addEventListener('change',()=>{$('#speaker').value=$('#speaker-select').value;});
$('#speaker').addEventListener('input',()=>{const value=$('#speaker').value;$('#speaker-select').value=value;});

async function loadModels(preferredId=null){const old=$('#model').value;const previous=preferredId||old;state.models=await api('/api/models');for(const id of selectedModels)if(!state.models.some(m=>m.id===id))selectedModels.delete(id);$('#model-count').textContent=state.models.length;$('#model').innerHTML=state.models.length?state.models.map(m=>`<option value="${m.id}">${escapeHtml(m.name)} · ${modelTimestamp(m.created)}</option>`).join(''):'<option value="">请先导入或导出模型</option>';if(state.models.some(m=>m.id===previous))$('#model').value=previous;$('#models').innerHTML=state.models.length?state.models.map(m=>`<div class="model-item"><label class="check-label model-selection"><input type="checkbox" class="select-model" data-id="${m.id}" aria-label="选择模型 ${escapeHtml(m.name)}" ${selectedModels.has(m.id)?'checked':''}>选择此模型</label><strong>◒ ${escapeHtml(m.name)}</strong><small>${escapeHtml(m.type)} · 创建于 ${modelTimestamp(m.created)}（北京时间）</small><div class="model-actions"><a href="/api/models/${m.id}/download">下载语音包 ↓</a>${`<button class="button danger small delete-model" data-id="${m.id}" ${state.job?.status==='running'?'disabled':''} aria-label="删除模型 ${escapeHtml(m.name)}">删除</button>`}</div></div>`).join(''):'<div class="empty-state"><strong>还没有模型</strong><p>完成训练并导出，或导入已有 VITS 语音包。</p></div>';if(preferredId&&old!==preferredId)clearPlayer();updateSpeakerOptions();updateBulkModels();}

$('#models').addEventListener('click',async event=>{
 const button=event.target.closest('.delete-model');if(!button)return;
 const model=state.models.find(m=>m.id===button.dataset.id);if(!model)return;
 if(!await confirmRemoval(`删除模型「${model.name}」？\n创建时间：${modelTimestamp(model.created)}\n模型库中的模型和语音包将移入回收目录，可撤销。训练素材、检查点及外部原始文件会保留。`))return;
 button.disabled=true;
 try{
  const result=await api(`/api/models/${model.id}`,{method:'DELETE'});
  saveModelUndo([result.undo]);
  if($('#model').value===model.id){$('#output-audio').pause();$('#output-audio').removeAttribute('src');$('#output-audio').load();$('#player').hidden=true;}
  await loadModels();toast('模型已移入回收目录，可撤销');
 }catch(error){toast(error.message,true);button.disabled=false;}
});
function modelUndoTokens(){try{const tokens=JSON.parse(localStorage.getItem('vits.deletedModels')||'[]');return Array.isArray(tokens)?tokens:[];}catch{return [];}}
function saveModelUndo(tokens){localStorage.removeItem('vits.deletedModel');localStorage.setItem('vits.deletedModels',JSON.stringify(tokens));$('#undo-model-delete').hidden=!tokens.length;$('#undo-model-delete').textContent=`撤销模型删除（${tokens.length} 个）`;}
if(!modelUndoTokens().length&&localStorage.getItem('vits.deletedModel'))saveModelUndo([localStorage.getItem('vits.deletedModel')]);
else saveModelUndo(modelUndoTokens());
function updateBulkModels(){
 const count=selectedModels.size, running=state.job?.status==='running';
 $('#selected-model-count').textContent=`已选 ${count} 个`;
 $('#delete-selected-models').disabled=!count||running;
 $('#select-all-models').checked=state.models.length>0&&count===state.models.length;
 $('#select-all-models').indeterminate=count>0&&count<state.models.length;
 $('#select-all-models').disabled=!state.models.length;
}
$('#models').addEventListener('change',event=>{const input=event.target.closest('.select-model');if(!input)return;if(input.checked)selectedModels.add(input.dataset.id);else selectedModels.delete(input.dataset.id);updateBulkModels();});
$('#select-all-models').addEventListener('change',event=>{selectedModels.clear();if(event.target.checked)state.models.forEach(m=>selectedModels.add(m.id));document.querySelectorAll('.select-model').forEach(input=>input.checked=selectedModels.has(input.dataset.id));updateBulkModels();});
action('#delete-selected-models',async()=>{
 const ids=[...selectedModels];if(!ids.length)return;
 if(!await confirmRemoval(`将所选 ${ids.length} 个模型及语音包移入回收目录？\n可撤销；训练素材和检查点保留。`))return;
 const tokens=[];let failure=null;
 for(const id of ids){try{const result=await api(`/api/models/${id}`,{method:'DELETE'});tokens.push(result.undo);selectedModels.delete(id);if($('#model').value===id)clearPlayer();saveModelUndo(tokens);}catch(error){failure=error;break;}}
 await loadModels();updateBulkModels();
 if(failure)throw new Error(`已删除 ${tokens.length} 个，其余未删除：${failure.message}`);
 toast(`已将 ${tokens.length} 个模型移入回收目录，可批量撤销`);
});
action('#undo-model-delete',async()=>{
 const tokens=modelUndoTokens();let last=null;
 while(tokens.length){last=await api(`/api/trash/${tokens[0]}/restore`,{method:'POST'});tokens.shift();saveModelUndo(tokens);}
 await loadModels(last?.id);toast('模型已恢复');
});

action('#import-model',async()=>{if(!$('#import-path').value.trim())throw new Error('请填写本地语音包路径');await watch(await api('/api/models/import',json('POST',{path:$('#import-path').value.trim()})));});
action('#synthesize',async()=>{if(!$('#model').value)throw new Error('请先选择模型');const speaker=Number($('#speaker').value);if(!$('#speaker').value.trim()||!Number.isInteger(speaker)||speaker<0||speaker>Number($('#speaker').max))throw new Error('请输入当前模型支持的整数说话人 ID');clearPlayer();await watch(await api('/api/synthesize',json('POST',{model_id:$('#model').value,text:$('#text').value,speaker:Number($('#speaker').value),speed:Number($('#speed').value)})));});
$('#speed').addEventListener('input',()=>$('#speed-value').textContent=Number($('#speed').value).toFixed(1)+'×');
async function loadEnvironment(preferredBase=null){const e=await api('/api/environment');if(e.executable.startsWith('/')){$('#import-path').placeholder=e.data_directory+'/imports/voice.tar.bz2';$('#base').placeholder=e.data_directory+'/base/c';$('#video-cookies').placeholder=e.data_directory+'/cookies/douyin.txt';}$('#env-badge').textContent=`Python ${e.python} · ${e.cuda?'CUDA':'CPU'}`;$('#device').textContent=e.device;const entries=[['Python',e.python],['解释器',e.executable],['PyTorch',e.torch],['计算设备',e.device],['FFmpeg',e.ffmpeg?'已就绪':'未安装'],['VITS 源码',e.engine?'已就绪':'缺失'],...Object.entries(e.dependencies).map(([k,v])=>[k,v?'已安装':'未安装'])];$('#environment').innerHTML=entries.map(([k,v])=>`<div class="env-row"><span>${escapeHtml(k)}</span><span>${escapeHtml(v)}</span></div>`).join('');if(preferredBase){$('#base').value=preferredBase;localStorage.setItem('vits.base',preferredBase);}else if(!$('#base').value&&e.base_models.length)$('#base').value=e.base_models.find(p=>p.replaceAll('\\','/').endsWith('/c'))||e.base_models[0];if($('#base').value.trim()&&speakerBase!==$('#base').value.trim())await loadBaseSpeakers().catch(e=>toast(e.message,true));}
window.addEventListener('beforeunload',e=>{if(state.dirty){e.preventDefault();e.returnValue='';}});
(async()=>{page(localStorage.getItem('vits.page')||'create');$('#base').value=localStorage.getItem('vits.base')||'';await loadEnvironment();await Promise.all([loadProjects(),loadModels(),loadHistory()]);const jobs=await api('/api/jobs');if(jobs.length){state.job=jobs[0];if(state.job.status==='running'&&state.job.project_id&&state.job.project_id!==state.project?.id&&[...$('#project').options].some(o=>o.value===state.job.project_id)){$('#project').value=state.job.project_id;await loadProject();}renderJob(await api(`/api/jobs/${jobs[0].id}`));if(jobs[0].status==='running')await pollJob();}})().catch(e=>toast(e.message,true));

let listenHistory=[], historyQueue=[];
const historySelected=new Set();
function updateHistorySelection(){
 const n=historySelected.size;
 $('#history-selected').textContent=`已选 ${n} 条`;
 $('#history-play').disabled=!n;$('#history-delete').disabled=!n;
 $('#history-all').checked=!!listenHistory.length&&n===listenHistory.length;
 $('#history-all').indeterminate=n>0&&n<listenHistory.length;
}
async function loadHistory(){
 const [history,models]=await Promise.all([api('/api/listen-history'),api('/api/models')]);
 listenHistory=history.map(h=>({...h,model_created:h.model_created??models.find(m=>m.id===h.options.model_id)?.created}));
 for(const id of historySelected)if(!listenHistory.some(h=>h.id===id))historySelected.delete(id);
 $('#history-count').textContent=listenHistory.length;
 $('#listen-history').innerHTML=listenHistory.map(h=>`<div class="history-item"><input type="checkbox" class="history-select" aria-label="选择试听：${escapeHtml(h.options.text)}" data-id="${escapeHtml(h.id)}" ${historySelected.has(h.id)?'checked':''}><div class="history-content"><div class="history-meta"><strong title="${escapeHtml(h.model_name)}">${escapeHtml(h.model_name)}</strong><time>模型生成：${h.model_created?escapeHtml(new Date(h.model_created*1000).toLocaleString()):'未知'}</time><time>试听：${escapeHtml(new Date(h.created*1000).toLocaleString())}</time><span>音色 ${escapeHtml(h.options.speaker)}</span><span>${escapeHtml(h.options.speed)}×</span></div><details class="history-text"><summary>${escapeHtml(h.options.text)}</summary><p>${escapeHtml(h.options.text)}</p></details></div><div class="history-actions"><button class="button secondary small history-one" data-id="${escapeHtml(h.id)}" aria-label="播放试听：${escapeHtml(h.options.text)}">▷ 播放</button><a href="/api/jobs/${encodeURIComponent(h.id)}/audio" download="试听-${escapeHtml(h.id)}.wav" aria-label="下载试听：${escapeHtml(h.options.text)}">下载</a></div></div>`).join('')||'<p class="field-help">生成成功的试听会自动保存在这里。</p>';
 updateHistorySelection();
}
function stopHistory(clear=false){historyQueue=[];const a=$('#history-player');a.pause();if(clear){a.removeAttribute('src');a.load();$('#history-playing').textContent='选择一条历史试听开始播放';}else if(a.getAttribute('src'))a.currentTime=0;updateHistoryTransport();}
async function nextHistory(){
 const id=historyQueue.shift();if(!id){updateHistoryTransport();return;}
 const h=listenHistory.find(h=>h.id===id);if(!h)return nextHistory();
 $('#output-audio').pause();const a=$('#history-player');a.src=`/api/jobs/${encodeURIComponent(id)}/audio`;
 $('#history-playing').textContent=`${h.model_name} · ${h.options.text}`;
 try{await a.play();}catch(e){historyQueue=[];toast('播放未开始，请点击左侧播放按钮',true);}
}
$('#history-player').addEventListener('ended',nextHistory);
$('#history-player').addEventListener('error',()=>{historyQueue=[];toast('历史音频加载失败，请刷新后重试',true);});
$('#history-stop').addEventListener('click',()=>stopHistory());
$('#history-play').addEventListener('click',()=>{historyQueue=listenHistory.filter(h=>historySelected.has(h.id)).map(h=>h.id);nextHistory();});
$('#listen-history').addEventListener('change',e=>{if(!e.target.matches('.history-select'))return;const id=e.target.dataset.id;if(e.target.checked)historySelected.add(id);else historySelected.delete(id);updateHistorySelection();});
$('#listen-history').addEventListener('click',e=>{const b=e.target.closest('.history-one');if(b){historyQueue=[b.dataset.id];nextHistory();}});
$('#history-all').addEventListener('change',e=>{historySelected.clear();if(e.target.checked)listenHistory.forEach(h=>historySelected.add(h.id));document.querySelectorAll('.history-select').forEach(i=>i.checked=historySelected.has(i.dataset.id));updateHistorySelection();});
action('#history-delete',async()=>{const ids=[...historySelected];if(!ids.length||!confirm(`删除所选 ${ids.length} 条试听及音频？删除后无法恢复。`))return;stopHistory(true);if(ids.includes(state.job?.id))clearPlayer();await api('/api/listen-history/delete',json('POST',{ids}));historySelected.clear();await loadHistory();toast('所选试听已删除');});

action('#pause-train',async()=>{if(!state.job)return;await api(`/api/jobs/${state.job.id}/training-pause`,json('POST',{pause:!state.job.pause_requested}));await pollJob();});

function historyTime(seconds){if(!Number.isFinite(seconds))return '0:00';return `${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,'0')}`;}
function updateHistoryTransport(){
 const a=$('#history-player'), ready=!!a.getAttribute('src'), duration=Number.isFinite(a.duration)?a.duration:0;
 $('#history-toggle').disabled=!ready;$('#history-toggle').textContent=a.paused?'▷':'Ⅱ';$('#history-toggle').setAttribute('aria-label',a.paused?'播放历史试听':'暂停历史试听');
 $('#history-current').textContent=historyTime(a.currentTime);$('#history-duration').textContent=historyTime(duration);
 const percent=duration?a.currentTime/duration*100:0,seek=$('#history-seek');seek.value=percent;seek.disabled=!duration;seek.style.setProperty('--played',`${percent}%`);
}
for(const event of ['timeupdate','loadedmetadata','durationchange','play','pause','ended','emptied'])$('#history-player').addEventListener(event,updateHistoryTransport);
$('#history-toggle').addEventListener('click',async()=>{const a=$('#history-player');if(a.paused){if(a.ended)a.currentTime=0;$('#output-audio').pause();try{await a.play();}catch(e){toast('播放失败，请重新选择试听',true);}}else a.pause();});
$('#history-seek').addEventListener('input',e=>{const a=$('#history-player');if(Number.isFinite(a.duration)){a.currentTime=a.duration*Number(e.target.value)/100;updateHistoryTransport();}});
