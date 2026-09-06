import vm from "node:vm";
import { performance } from "node:perf_hooks";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const request = JSON.parse(Buffer.concat(chunks).toString("utf8"));
const transformed = request.code.replace(/\bexport\s+function\s+createPainter/, "function createPainter");
const allowed = new Set([
  "createPaper", "mark", "paintStroke", "setPalette", "washRegion", "strokePath", "dryBrushPath",
  "dab", "poolPigment", "scatterGrain", "fadeLayer", "composite"
]);
let current = [];
let phase = "factory";
let randomState = request.seed >>> 0;
const random = () => {
  randomState += 0x6d2b79f5;
  let t = randomState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const api = { reset(seed) { randomState = seed >>> 0; } };
let activeFields = null;
let queries = 0;
let queryCounts = Object.create(null);
function query(field, x, z, offsetX=0, offsetZ=0) {
  if (phase !== 'renderFrame' || !activeFields) throw new Error('Field queries are available only in free-paint renderFrame');
  if (++queries > 40000) throw new Error('field query cap exceeded');
  if (!Number.isFinite(x) || !Number.isFinite(z) || x<0 || x>1 || z<0 || z>1) throw new Error('sample coordinates must be in [0,1]');
  if (typeof field!=='string' || !Object.hasOwn(activeFields,field)) throw new Error('Unknown scalar field');
  const grid=activeFields[field];
  if (!grid || !Array.isArray(grid.values)) throw new Error('Unknown scalar field');
  queryCounts[field]=(queryCounts[field]||0)+1;
  const fx=Math.max(0,Math.min(grid.nx-1,x*(grid.nx-1)+offsetX));
  const fz=((z*(grid.nz-1)+offsetZ)%grid.nz+grid.nz)%grid.nz;
  const i=Math.floor(fx), j=Math.floor(fz), u=fx-i, v=fz-j;
  const k=Math.min(grid.nx-1,i+1), l=(j+1)%grid.nz;
  return (1-u)*((1-v)*grid.values[i*grid.nz+j]+v*grid.values[i*grid.nz+l])+u*((1-v)*grid.values[k*grid.nz+j]+v*grid.values[k*grid.nz+l]);
}
api.sample=(field,x,z)=>query(field,x,z);
api.gradient=(field,x,z)=>{
  const grid=activeFields && typeof field==='string' && Object.hasOwn(activeFields,field) ? activeFields[field] : null;
  if (!grid) throw new Error('Unknown scalar field');
  const fx=x*(grid.nx-1), span=Math.min(grid.nx-1,fx+1)-Math.max(0,fx-1);
  const dx=(query(field,x,z,1)-query(field,x,z,-1))*(grid.nx-1)/span;
  const dz=(query(field,x,z,0,1)-query(field,x,z,0,-1))*(grid.nz-1)/2;
  return Object.freeze(Object.assign(Object.create(null),{dx,dz}));
};
for (const name of allowed) {
  api[name] = (args = {}) => {
    if (phase !== "renderFrame") throw new Error(`${name}: drawing is only allowed inside renderFrame, not ${phase}`);
    if (request.profile === "stroke_only" && !["createPaper", "mark"].includes(name)) {
      throw new Error(`${name}: stroke_only permits only createPaper and mark`);
    }
    if (request.profile === 'free_paint' && !['createPaper','paintStroke'].includes(name)) throw new Error('free_paint permits createPaper and paintStroke');
    if (!args || typeof args !== "object" || Array.isArray(args)) {
      throw new Error(`${name}: expected ONE options object; emit each mark in a separate api.mark call`);
    }
    if (name === "mark") {
      if (!Array.isArray(args.points) || args.points.length < 2 || args.points.length > 8 ||
          !args.points.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) {
        throw new Error("mark.points: expected 2..8 numeric [x,z] pairs, not objects or RGB tuples");
      }
      if (!Number.isInteger(args.sample_id)) throw new Error("mark.sample_id: use the original stroke sample's integer id");
    }
    if (name==='paintStroke') {
      if (!Array.isArray(args.points) || args.points.length<2 || args.points.length>64 ||
          !args.points.every(p=>Array.isArray(p) && p.length===2 && p.every(v=>Number.isFinite(v) && v>=0 && v<=1)))
        throw new Error('paintStroke.points requires 2..64 finite [x,y] pairs in [0,1]; a one-point dab is not a stroke');
      if (!/^#[0-9a-fA-F]{6}$/.test(args.color)) throw new Error('paintStroke.color requires #rrggbb, not HSL or RGB strings');
    }
    if (current.length >= request.maxOperations) throw new Error("operation cap exceeded");
    const copy = structuredClone(args);
    if (Array.isArray(copy.points) && copy.points.length > request.maxPathPoints) {
      throw new Error("path cap exceeded");
    }
    current.push({ op: name, args: copy });
  };
}
Object.freeze(api);
// Never expose host constructors/functions/Errors through the drawing interface.
// Candidate code and its input objects live in the VM realm; string code generation
// is disabled there. The sole host bridge has no prototype and returns null-prototype
// envelopes containing primitives (or the null-prototype numeric gradient object).
const hostInvoke = (name,args) => {
  const envelope=Object.create(null);
  try {
    if (!Array.isArray(args)) throw new Error('API arguments must be an array');
    if (name==='__random') envelope.value=random();
    else if (typeof name==='string' && Object.hasOwn(api,name)) envelope.value=api[name](...args);
    else throw new Error('Unknown API operation');
  } catch (error) {
    envelope.error='Painter API call failed';
    try { if (typeof error.message==='string') envelope.error=error.message; } catch {}
  }
  return Object.freeze(envelope);
};
Object.setPrototypeOf(hostInvoke,null);
Object.freeze(hostInvoke);
const context = vm.createContext({__hostInvoke:hostInvoke}, {
  codeGeneration:{strings:false,wasm:false},microtaskMode:'afterEvaluate'
});
new vm.Script(`
  this.__api=Object.create(null);
  for(const name of ${JSON.stringify(Object.keys(api))}) {
    __api[name]=function(...args) {
      const result=__hostInvoke(name,args);
      if(result.error) throw new Error(result.error);
      return result.value;
    };
    Object.freeze(__api[name]);
  }
  Object.freeze(__api);
  Math.random=function(){return __hostInvoke('__random',[]).value;};
  Object.freeze(Math);
  this.__state=Object.create(null);
`).runInContext(context,{timeout:request.vmTimeoutMs});
const script = new vm.Script(`${transformed}\n;this.__createPainter = createPainter;`, { filename: "candidate-painter.js" });
script.runInContext(context, { timeout: request.vmTimeoutMs });
context.__styleJSON = JSON.stringify(request.style);
const painter = new vm.Script("__createPainter(__api, Object.freeze(JSON.parse(__styleJSON)))").runInContext(context, { timeout: request.vmTimeoutMs });
if (!painter || typeof painter.reset !== "function" || typeof painter.renderFrame !== "function") {
  throw new Error("factory returned an invalid painter");
}
const started = performance.now();
context.__painter = painter;
context.__seed = request.seed;
phase = "reset";
new vm.Script("__painter.reset(__seed)").runInContext(context, { timeout: request.vmTimeoutMs });
const operationsByFrame = [];
const queryCountsByFrame = [];
for (let index = 0; index < request.frames.length; index += 1) {
  current = [];
  activeFields=request.queryFields?.[index] || null;
  queries=0;
  queryCounts=Object.create(null);
  phase = "renderFrame";
  const frame = structuredClone(request.frames[index]);
  const call = new vm.Script("__painter.renderFrame(JSON.parse(__frameJSON), __time, __state)");
  context.__painter = painter;
  context.__frameJSON = JSON.stringify(frame);
  context.__time = index;
  call.runInContext(context, { timeout: request.vmTimeoutMs });
  operationsByFrame.push(current);
  queryCountsByFrame.push(queryCounts);
}
process.stdout.write(JSON.stringify({ operationsByFrame,queryCountsByFrame,elapsedMs: performance.now() - started }));
