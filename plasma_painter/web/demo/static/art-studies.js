// Hand-authored field-driven auditions, not trained artist policies.
export const studies={
 impasto:{name:'Van Gogh-inspired · directional impasto',description:'Layered short strokes follow scalar contour tangents; cool shadows and golden highlights encode intensity.',palette:[[19,32,83],[34,70,147],[29,127,148],[130,173,98],[238,192,61],[248,226,133]]},
 facets:{name:'Picasso-inspired · faceted planes',description:'Angular planes sample the field in place, without relocating structures.',palette:[[37,46,58],[59,101,112],[124,153,143],[197,183,131],[208,125,77],[237,213,169]]},
 tonal:{name:'Manet-inspired · tonal brushwork',description:'Broad overlapping strokes compress the field into distinct light and dark masses.',palette:[[29,33,38],[52,72,68],[101,113,99],[167,163,133],[224,208,171],[250,238,207]]},
 dots:{name:'Seurat-inspired · optical dots',description:'Separate colored touches mix optically; their radius and color follow the scalar.',palette:[[35,39,102],[61,97,164],[48,148,152],[143,174,100],[239,174,78],[247,219,145]]}
};
function decode(record){const b=Uint8Array.from(atob(record.data),c=>c.charCodeAt(0)),[nx,nz]=record.shape;return {nx,nz,at(x,z){return b[Math.max(0,Math.min(nx-1,x))*nz+(z%nz+nz)%nz]/255;}};}
function rng(seed){return()=>{seed=(seed+0x6d2b79f5)>>>0;let t=seed;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
function rgb(palette,v){const p=Math.max(0,Math.min(1,v))*(palette.length-1),i=Math.min(palette.length-2,Math.floor(p)),t=p-i;return palette[i].map((c,k)=>Math.round(c*(1-t)+palette[i+1][k]*t));}
export function paintStudy(canvas,frame,field,kind,seed=1701){
 const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,r=decode(frame.rasters[field]),random=rng(seed),palette=studies[kind].palette;
 const sample=(u,v)=>r.at(Math.round(u*(r.nx-1)),Math.floor(v*r.nz));
 ctx.globalAlpha=1;ctx.fillStyle=kind==='impasto'?'#182749':'#f4e9ce';ctx.fillRect(0,0,w,h);ctx.lineCap='round';
 if(kind==='facets'){
  const nx=22,nz=27,points=[];
  for(let x=0;x<=nx;x++){points[x]=[];for(let z=0;z<=nz;z++)points[x][z]=[(x+(x>0&&x<nx?(random()-.5)*.45:0))/nx,(z+(z>0&&z<nz?(random()-.5)*.45:0))/nz];}
  for(let x=0;x<nx;x++)for(let z=0;z<nz;z++){const p=[points[x][z],points[x+1][z],points[x+1][z+1],points[x][z+1]];
   for(const tri of [[0,1,2],[0,2,3]]){const u=tri.reduce((s,i)=>s+p[i][0],0)/3,v=tri.reduce((s,i)=>s+p[i][1],0)/3;ctx.beginPath();tri.forEach((j,i)=>i?ctx.lineTo(p[j][0]*w,p[j][1]*h):ctx.moveTo(p[j][0]*w,p[j][1]*h));ctx.closePath();ctx.fillStyle=`rgb(${rgb(palette,sample(u,v)).join(',')})`;ctx.fill();ctx.lineWidth=.65;ctx.strokeStyle='rgba(25,40,45,.4)';ctx.stroke();}
  }
 }else{
  const step=kind==='dots'?5:kind==='tonal'?12:8;
  for(let y=2;y<h;y+=step)for(let x=2;x<w;x+=step){const jitter=kind==='dots'?1.2:step*.34,px=x+(random()-.5)*jitter,py=y+(random()-.5)*jitter,u=px/w,v=py/h,value=sample(u,v),ix=Math.round(u*(r.nx-1)),iz=Math.floor(v*r.nz),dx=r.at(ix+1,iz)-r.at(ix-1,iz),dz=r.at(ix,iz+1)-r.at(ix,iz-1),angle=Math.atan2(dz,dx)+Math.PI/2;
   const c=rgb(palette,kind==='tonal'?Math.round(value*5)/5:value);
   if(kind==='dots'){const col=rgb(palette,value+(random()-.5)*.16);ctx.fillStyle=`rgb(${col.join(',')})`;ctx.beginPath();ctx.arc(px,py,1.1+value*1.5,0,2*Math.PI);ctx.fill();}
   else{const length=kind==='tonal'?step*(1.2+Math.min(1,Math.hypot(dx,dz)*5)):step*(1.2+value*1.1);ctx.save();ctx.translate(px,py);ctx.rotate(angle);ctx.strokeStyle=`rgb(${c.join(',')})`;ctx.lineWidth=kind==='tonal'?step*.95:step*.67;ctx.beginPath();ctx.moveTo(-length/2,0);ctx.quadraticCurveTo(0,-step*.25,length/2,0);ctx.stroke();if(kind==='impasto'){ctx.strokeStyle=`rgba(${c.map(v=>Math.min(255,v+35)).join(',')},.65)`;ctx.lineWidth=1.1;ctx.beginPath();ctx.moveTo(-length*.4,-1.5);ctx.quadraticCurveTo(0,-step*.2,length*.4,-1.5);ctx.stroke();}ctx.restore();}
  }
 }
 ctx.setLineDash([3,5]);ctx.strokeStyle='rgba(255,245,216,.52)';ctx.lineWidth=1;const sx=frame.geometry.separatrix_face_u*w;ctx.beginPath();ctx.moveTo(sx,0);ctx.lineTo(sx,h);ctx.stroke();ctx.setLineDash([]);
}
