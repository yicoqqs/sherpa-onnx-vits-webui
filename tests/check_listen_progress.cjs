const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('static/app.js','utf8');
const nodes=new Map();function node(s){if(!nodes.has(s))nodes.set(s,{hidden:true,textContent:'',innerHTML:'',value:'old',style:{},classList:{toggle(){}},setAttribute(k,v){this[k]=v},removeAttribute(k){delete this[k]}});return nodes.get(s);}
const context=vm.createContext({Intl,Date,document:{querySelector:node},console});
vm.runInContext(source.slice(0,source.indexOf('function toast'))+source.slice(source.indexOf('function modelTimestamp'),source.indexOf('function clearPlayer'))+source.slice(source.indexOf('function formatDuration'),source.indexOf('function renderPrepareProgress')),context);
assert.equal(vm.runInContext('modelTimestamp(0)',context),'1970-01-01 08:00:00');
vm.runInContext(`renderTaskProgress({kind:'export',status:'running',created:0,log:'@@TASK_PROGRESS@@{"stage":"onnx","percent":30}\\n@@TASK_PROGRESS@@broken'},'export')`,context);
assert.equal(node('#export-progress').hidden,false);assert.equal(node('#export-percent').textContent,'30%');assert(node('#export-stage').textContent.includes('ONNX'));
vm.runInContext(`renderTaskProgress({kind:'export',status:'failed',created:0,finished:10,log:'@@TASK_PROGRESS@@{"stage":"onnx","percent":30}'},'export')`,context);
assert(node('#export-stage').textContent.includes('失败'));assert.equal(node('#export-percent').textContent,'30%');
vm.runInContext(`renderTaskProgress({kind:'synthesize',status:'completed',created:0,finished:10,log:''},'synthesize')`,context);
assert.equal(node('#synthesize-percent').textContent,'100%');
vm.runInContext(`renderTaskProgress({kind:'train'},'export')`,context);assert.equal(node('#export-progress').hidden,true);
vm.runInContext(source.slice(source.indexOf('async function loadModels'),source.indexOf("$('#models').addEventListener")),context);
context.clearPlayer=()=>{context.cleared=true;};context.updateSpeakerOptions=()=>{};
context.api=async()=>[{id:'new',name:'同名模型',created:60,type:'VITS'},{id:'old',name:'同名模型',created:0,type:'VITS'}];
(async()=>{await vm.runInContext(`loadModels('new')`,context);assert.equal(node('#model').value,'new');assert(node('#model').innerHTML.includes('1970-01-01 08:01:00'));assert(node('#models').innerHTML.includes('1970-01-01 08:00:00'));assert(context.cleared);assert.equal((node('#models').innerHTML.match(/class="button danger small delete-model"/g)||[]).length,2);console.log('PASS: second timestamps, exact export auto-selection, progress parsing, completion and failure states');})().catch(e=>{console.error(e);process.exitCode=1});
