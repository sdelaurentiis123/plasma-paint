// Exercise actual raster-to-mark code without a browser or external services.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
const source=fs.readFileSync('plasma_painter/web/demo/static/art-studies.js','utf8');
const {studies,paintStudy}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const dir='plasma_painter/web/demo/static/';
const manifest=JSON.parse(fs.readFileSync(dir+'sections.generated.json'));
function trace(frame,field,kind){const hash=crypto.createHash('sha256');
 const ctx=new Proxy({}, {get(target,key){return (...args)=>{for(const a of args)if(typeof a==='number')assert(Number.isFinite(a));hash.update(JSON.stringify([key,args]));};},set(target,key,value){hash.update(JSON.stringify([key,value]));return true;}});
 paintStudy({width:120,height:80,getContext:()=>ctx},frame,field,kind,1701);return hash.digest('hex');
}
for(const section of manifest.sections){const clip=JSON.parse(fs.readFileSync(dir+section.url));assert.equal(clip.split,'art_train');
 for(const field of ['density','density_fluctuation','potential','electron_temperature']){
  const fingerprints=[];for(const kind of Object.keys(studies)){const first=trace(clip.frames[0],field,kind);assert.equal(first,trace(clip.frames[0],field,kind));assert.notEqual(first,trace(clip.frames[7],field,kind));fingerprints.push(first);}assert.equal(new Set(fingerprints).size,4);
 }
}
console.log('All 8 sections × 4 fields × 4 styles: finite, deterministic, distinct, and responsive to changed frames.');
