import {studies,paintStudy} from './art-studies.js';
import {BrowserCanvasRuntime} from './canvas-runtime.js';
const $=id=>document.getElementById(id);
let clip=window.PLASMA_BOOTSTRAP,index=0,scientific=false,playing=false,last=0,request=0;
const panels=[];
for(const [kind,spec] of Object.entries(studies)){
 const figure=document.createElement('figure'),canvas=document.createElement('canvas'),caption=document.createElement('figcaption'),heading=document.createElement('strong'),text=document.createElement('p');
 canvas.width=720;canvas.height=480;canvas.setAttribute('aria-label',spec.name+' interpretation of selected plasma field');heading.textContent=spec.name;text.textContent=spec.description;caption.append(heading,text);figure.append(canvas,caption);$('styles').append(figure);
 const runtime=new BrowserCanvasRuntime(canvas,{vibrant:true},1701);runtime.scientific=true;panels.push({kind,canvas,runtime});
}
function stop(){playing=false;$('play').textContent='Play';}
function draw(){const f=clip.frames[index];for(const p of panels){if(scientific){p.runtime.field=$('field').value;p.runtime.scientificFrame(f);}else paintStudy(p.canvas,f,$('field').value,p.kind,1701);}$('frame').value=index;$('position').textContent=`85604 · y=${f.geometry.plane_y??18} · frame ${f.source.frame_index}`;}
$('field').onchange=draw;
$('view').onclick=()=>{scientific=!scientific;$('view').textContent=scientific?'Show creative':'Show scientific';$('view').setAttribute('aria-pressed',String(scientific));$('scale').textContent=scientific?'Scientific reference: low → high maps navy → teal → gold → red.':'Creative views: each palette maps the selected field’s training-normalized intensity.';draw();};
$('frame').oninput=()=>{stop();index=Number($('frame').value);draw();};$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';};
document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
const loaded=new Map();
async function selectSection(url){const ticket=++request;stop();$('error').textContent='';try{if(!loaded.has(url)){const response=await fetch(url);if(!response.ok)throw Error();loaded.set(url,await response.json());}const next=loaded.get(url);if(next.split!=='art_train')throw Error();if(ticket!==request)return;clip=next;index=0;$('frame').max=clip.frames.length-1;draw();}catch{$('error').textContent='Could not load the selected training section.';}}
draw();fetch('sections.generated.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(async manifest=>{const select=$('section');select.replaceChildren();for(const section of manifest.sections){const option=document.createElement('option');option.value=section.url;option.textContent=`y=${section.y}`;option.selected=section.y===18;select.append(option);}select.disabled=false;select.onchange=()=>selectSection(select.value);await selectSection(select.value);}).catch(()=>{$('error').textContent='Section manifest unavailable; showing the first training frame only.';});
function tick(t){if(playing&&t-last>650){last=t;index=(index+1)%clip.frames.length;draw();}requestAnimationFrame(tick);}requestAnimationFrame(tick);
