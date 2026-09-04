import {BrowserCanvasRuntime} from './canvas-runtime.js';
import {createPainter} from './painter.generated.js';
const $=id=>document.getElementById(id);
const style={paper:'#f6f0df',grain:.015,bleed:.16,persistence:.55,vibrant:true,washStrength:.68};
let clip=window.PLASMA_BOOTSTRAP,index=0,playing=false,last=0;
let sectionRequest=0;
const paint=new BrowserCanvasRuntime($('painting'),style,clip.seed);
const science=new BrowserCanvasRuntime($('scientific'),style,clip.seed);science.scientific=true;
const operations=[];
const api={reset(){}};
for(const op of ['createPaper','setPalette','washRegion','strokePath','dryBrushPath','dab','poolPigment','scatterGrain','fadeLayer','composite']) api[op]=(args={})=>operations.push({op,args});
const painter=createPainter(Object.freeze(api),style);
function draw(){
  // Reconstruct from clip start so scrubbing has the same seeded pigment history as playback.
  paint.seed=clip.seed;paint.persistent.getContext('2d').clearRect(0,0,768,512);painter.reset(clip.seed);
  const state={};
  for(let i=0;i<=index;i++){operations.length=0;painter.renderFrame(clip.frames[i],i,state);paint.render(clip.frames[i],operations);}
  science.render(clip.frames[index],[]);$('frame').value=index;
  $('position').textContent=`85604 · y=${clip.frames[index].geometry.plane_y ?? 18} · frame ${clip.frames[index].source.frame_index}`;
  $('legend').style.background=`linear-gradient(90deg,${Array.from({length:11},(_,i)=>`rgb(${paint.color(i/10).join(',')})`).join(',')})`;
}
$('field').onchange=()=>{paint.field=science.field=$('field').value;draw();};
$('strength').oninput=()=>{style.washStrength=Number($('strength').value)/100;draw();};
$('frame').oninput=()=>{playing=false;$('play').textContent='Play';index=Number($('frame').value);draw();};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play';};
document.addEventListener('visibilitychange',()=>{if(document.hidden){playing=false;$('play').textContent='Play';}});
draw();
fetch('features.generated.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(d=>{if(d.split!=='art_train')throw Error();clip=d;$('frame').max=clip.frames.length-1;draw();}).catch(()=>{$('error').textContent='The training clip could not be loaded.';});
function tick(now){if(playing&&now-last>650){last=now;index=(index+1)%clip.frames.length;draw();}requestAnimationFrame(tick);}requestAnimationFrame(tick);

fetch('sections.generated.json').then(r=>{if(!r.ok)throw Error();return r.json();}).then(manifest=>{
  const label=document.createElement('label');label.textContent='Section ';
  const select=document.createElement('select');select.setAttribute('aria-label','Field-aligned cross-section');
  for(const section of manifest.sections){const option=document.createElement('option');option.value=section.url;option.textContent=section.label;option.selected=section.y===18;select.append(option);}
  label.append(select);document.querySelector('.toolbar').prepend(label);
  const note=document.createElement('p');note.textContent='Each section uses its own training-fitted color scale. Views are radial × periodic field-aligned coordinates; equal colors across sections do not imply equal physical values.';$('status').before(note);
  select.onchange=async()=>{const ticket=++sectionRequest;playing=false;$('play').textContent='Play';select.disabled=true;
    try{const response=await fetch(select.value);if(!response.ok)throw Error();const next=await response.json();if(next.split!=='art_train')throw Error();if(ticket!==sectionRequest)return;clip=next;index=0;$('frame').max=clip.frames.length-1;draw();}
    catch{$('error').textContent='Could not load this cross-section.';}finally{select.disabled=false;}
  };
}).catch(()=>{});
