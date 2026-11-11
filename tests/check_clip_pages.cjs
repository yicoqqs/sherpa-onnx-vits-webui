const fs=require('fs'),vm=require('vm'),assert=require('assert');
const code=fs.readFileSync('static/app.js','utf8');
const elements=new Map();
function el(s){if(!elements.has(s))elements.set(s,{innerHTML:'',textContent:'',addEventListener(type,fn){this[type]=fn;},focus(){}});return elements.get(s);}
const ctx=vm.createContext({document:{querySelector:el},console});
vm.runInContext(code.slice(0,code.indexOf('function toast'))+code.slice(code.indexOf('function groupClips'),code.indexOf('function renderProject'))+code.slice(code.indexOf('function renderClips'),code.indexOf("action('#new-project'"))+'\nfunction updateSummary(){}',ctx);
vm.runInContext(`state.project={id:'p',uploads:[{file:'a',name:'同名.wav'},{file:'b',name:'同名.wav'}],clips:Array.from({length:45},(_,i)=>({id:'c'+i,source_file:i<41?'a':'b',text:'原文'+i,enabled:true}))};renderClips();`,ctx);
function html(){return el('#clips').innerHTML;}
assert.equal((html().match(/class="clip" data-id/g)||[]).length,20);
assert.equal((html().match(/role="tab" /g)||[]).length,2);
vm.runInContext(`state.clipView.pages[state.clipView.group]=3;renderClips();`,ctx);
assert.equal((html().match(/class="clip" data-id/g)||[]).length,1);
// Exercise actual edit handler, then navigate away and back.
const text={value:'跨页校对'},enabled={checked:false};
el('#clips').input({target:{closest:()=>({dataset:{id:'c40'},querySelector:s=>s==='input[type=text]'?text:enabled})}});
vm.runInContext(`state.clipView.group='file:b';renderClips();state.clipView.group='file:a';renderClips();`,ctx);
assert(html().includes('value="跨页校对"'));
assert(vm.runInContext('state.dirty && state.project.clips[40].enabled===false',ctx));
vm.runInContext(`state.project.clips.splice(40,1);renderClips();`,ctx);
assert(vm.runInContext('state.clipView.pages["file:a"]===2',ctx));
vm.runInContext(`state.project={id:'new',clips:[],uploads:[]};renderClips();`,ctx);
assert(html().includes('暂无录音片段'));
console.log('PASS: 20 per page, distinct source tabs, last page, cross-page edits, disabled state, deletion clamp, empty project');

const saveStart=code.indexOf('async function saveClips()');
vm.runInContext(code.slice(saveStart,code.indexOf("action('#save-clips'",saveStart)),ctx);
vm.runInContext(`state.project={id:'save-test',clips:Array.from({length:45},(_,i)=>({id:'s'+i,text:i===40?'另页修改':'文本',enabled:i!==40}))};`,ctx);
let submitted;
ctx.selected=()=> 'save-test';ctx.json=(method,data)=>({method,body:data});ctx.api=async(url,options)=>{submitted=options.body.clips;return {id:'save-test',clips:submitted};};
ctx.document.querySelectorAll=()=>[];
(async()=>{await vm.runInContext('saveClips()',ctx);assert.equal(submitted.length,45);assert.equal(submitted[40].text,'另页修改');assert.equal(submitted[40].enabled,false);console.log('PASS: save submits all 45 clips including edits outside visible page');})().catch(e=>{console.error(e);process.exitCode=1;});
