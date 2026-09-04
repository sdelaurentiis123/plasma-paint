const $=id=>document.getElementById(id);
let candidates=[],frame=0,playing=false,last=0,revision=0;
const cache=new Map();
function image(url){if(!cache.has(url))cache.set(url,new Promise((resolve,reject)=>{const i=new Image();i.onload=()=>resolve(i);i.onerror=reject;i.src=url;}));return cache.get(url);}
async function draw(){const ticket=++revision;const pictures=await Promise.all(candidates.map(c=>image(c.frames[frame])));if(ticket!==revision)return;for(let i=0;i<pictures.length;i++)$(`painting-${i}`).src=pictures[i].src;$('position').textContent=`85604 · frame ${candidates[0].indices[frame]}`;$('frame').value=frame;}
function stop(){playing=false;$('play').textContent='Play all';}
function fail(){stop();$('error').textContent='Could not load the saved candidate renders. Refresh after the local gallery build completes.';}
try{const r=await fetch('candidates.generated.json');if(!r.ok)throw Error();candidates=(await r.json()).candidates;
 for(const [i,c] of candidates.entries()){
  const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption'),label=document.createElement('strong'),link=document.createElement('a');
  img.id=`painting-${i}`;img.alt=`Model-generated painter ${c.label}, TCV training frame`;label.textContent=`Painter ${c.label} · seed ${c.seed}`;
  link.href=c.code;link.target='_blank';link.rel='noopener';link.textContent='View generated code ↗';caption.append(label,link);figure.append(img,caption);$('gallery').append(figure);
 }
 await draw();$('play').disabled=false;$('frame').disabled=false;$('frame').max=candidates[0].frames.length-1;
 $('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause all':'Play all';};
 $('frame').oninput=()=>{stop();frame=Number($('frame').value);draw().catch(fail);};
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
 function tick(t){if(playing&&t-last>450){last=t;frame=(frame+1)%candidates[0].frames.length;draw().catch(fail);}requestAnimationFrame(tick);}requestAnimationFrame(tick);
}catch{fail();}
