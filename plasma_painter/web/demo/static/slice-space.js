import {BrowserCanvasRuntime} from './canvas-runtime.js';
const $=id=>document.getElementById(id),canvas=$('space'),ctx=canvas.getContext('2d');
const colorRuntime=new BrowserCanvasRuntime(document.createElement('canvas'),{},1701);
let sections=[],frame=0,angle=.44,pitch=.2,tour=false,drag=null,last=0,cache=new Map();
function raster(section){const key=`${section.y}:${frame}:${$('field').value}`;if(!cache.has(key))cache.set(key,colorRuntime.raster(section.clip.frames[frame].rasters[$('field').value]));return cache.get(key);}
function world(u,v,i){const scene=$('scene').value,s=i-3.5;
  if(scene==='stack')return [(u-.5)*2.3,(v-.5)*2.6,s*.48];
  if(scene==='ribbon')return [(u-.5)*2.1+s*.56,(v-.5)*2.3+Math.sin(i*.8)*.35,Math.sin(i*.8)*.6];
  const a=s*.20;return [(u+.12)*2.4*Math.cos(a)-1.4,(v-.5)*2.5,(u+.12)*2.4*Math.sin(a)];
}
function project(p){const x=p[0]*Math.cos(angle)+p[2]*Math.sin(angle),z=-p[0]*Math.sin(angle)+p[2]*Math.cos(angle),y=p[1]*Math.cos(pitch)-z*Math.sin(pitch),depth=p[1]*Math.sin(pitch)+z*Math.cos(pitch);const scale=canvas.height*.95/(4.4+depth);return [canvas.width*.5+x*scale,canvas.height*.49+y*scale,depth];}
function draw(){ctx.fillStyle='#101b24';ctx.fillRect(0,0,canvas.width,canvas.height);const faces=[];colorRuntime.field=$('field').value;
  sections.forEach((section,i)=>{const r=raster(section),nx=20,nz=26;
    for(let x=0;x<nx;x++)for(let z=0;z<nz;z++){
      const u=x/nx,v=z/nz,points=[[u,v],[u+1/nx,v],[u+1/nx,v+1/nz],[u,v+1/nz]].map(p=>project(world(...p,i)));
      const ix=Math.min(r.nx-1,Math.floor((u+.5/nx)*r.nx)),iz=Math.min(r.nz-1,Math.floor((v+.5/nz)*r.nz)),value=r.bytes[ix*r.nz+iz]/255,col=colorRuntime.color(value);
      for(const tri of [[0,1,2],[0,2,3]])faces.push({points:tri.map(j=>points[j]),color:col,shade:tri[1]===1?.93:1,depth:tri.reduce((s,j)=>s+points[j][2],0)/3});
    }
  });
  faces.sort((a,b)=>b.depth-a.depth);
  for(const face of faces){ctx.beginPath();face.points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=`rgb(${face.color.map(c=>Math.round(c*face.shade)).join(',')})`;ctx.fill();ctx.strokeStyle='rgba(10,26,32,.15)';ctx.lineWidth=.4;ctx.stroke();}
  sections.forEach((s,i)=>{const p=project(world(0,0,i));ctx.fillStyle='#fff6d8';ctx.font='14px system-ui';ctx.fillText(`y=${s.y}`,p[0],p[1]-8);});
  $('state').textContent=`85604 · frame ${frame} · ${sections.length}/8 sections · ${$('scene').selectedOptions[0].textContent}`;
}
for(const id of ['field','scene'])$(id).onchange=draw;
$('frame').oninput=()=>{frame=Number($('frame').value);draw();};
$('orbit').oninput=()=>{angle=Number($('orbit').value)*Math.PI/180;tour=false;$('tour').textContent='Start camera tour';draw();};
$('tour').onclick=()=>{tour=!tour;$('tour').textContent=tour?'Pause camera tour':'Start camera tour';};
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);tour=false;$('tour').textContent='Start camera tour';};
canvas.onpointermove=e=>{if(drag){angle+=(e.clientX-drag[0])*.006;pitch=Math.max(-.9,Math.min(.9,pitch+(e.clientY-drag[1])*.004));drag=[e.clientX,e.clientY];draw();}};
canvas.onpointerup=canvas.onpointercancel=()=>{drag=null;};
document.addEventListener('visibilitychange',()=>{if(document.hidden){tour=false;$('tour').textContent='Start camera tour';}});
function tick(t){if(tour&&t-last>65){angle+=.008;pitch=.16+Math.sin(angle*.7)*.12;last=t;draw();}requestAnimationFrame(tick);}requestAnimationFrame(tick);
try{const response=await fetch('sections.generated.json');if(!response.ok)throw Error();const manifest=await response.json();
  for(const section of manifest.sections){const r=await fetch(section.url);if(!r.ok)throw Error();const clip=await r.json();if(clip.split!=='art_train')throw Error();sections.push({...section,clip});draw();}
}catch{$('state').textContent='Could not load the training sections. Build the section cache and refresh.';}
