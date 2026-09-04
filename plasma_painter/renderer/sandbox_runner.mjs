import vm from "node:vm";
import { performance } from "node:perf_hooks";

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const request = JSON.parse(Buffer.concat(chunks).toString("utf8"));
const transformed = request.code.replace(/\bexport\s+function\s+createPainter/, "function createPainter");
const allowed = new Set([
  "createPaper", "setPalette", "washRegion", "strokePath", "dryBrushPath",
  "dab", "poolPigment", "scatterGrain", "fadeLayer", "composite"
]);
let current = [];
let randomState = request.seed >>> 0;
const random = () => {
  randomState += 0x6d2b79f5;
  let t = randomState;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const api = { reset(seed) { randomState = seed >>> 0; } };
for (const name of allowed) {
  api[name] = (args = {}) => {
    if (current.length >= request.maxOperations) throw new Error("operation cap exceeded");
    const copy = structuredClone(args);
    if (Array.isArray(copy.points) && copy.points.length > request.maxPathPoints) {
      throw new Error("path cap exceeded");
    }
    current.push({ op: name, args: copy });
  };
}
Object.freeze(api);
const safeMath = Object.create(null);
for (const name of Object.getOwnPropertyNames(Math)) {
  const descriptor = Object.getOwnPropertyDescriptor(Math, name);
  if (descriptor && "value" in descriptor) safeMath[name] = descriptor.value;
}
safeMath.random = random;
Object.freeze(safeMath);
const context = vm.createContext({ Math: safeMath, JSON, Object, Array, Number, String, Boolean });
const script = new vm.Script(`${transformed}\n;this.__createPainter = createPainter;`, { filename: "candidate-painter.js" });
script.runInContext(context, { timeout: request.vmTimeoutMs });
const painter = context.__createPainter(api, Object.freeze(structuredClone(request.style)));
if (!painter || typeof painter.reset !== "function" || typeof painter.renderFrame !== "function") {
  throw new Error("factory returned an invalid painter");
}
const persistentState = Object.create(null);
const started = performance.now();
painter.reset(request.seed);
const operationsByFrame = [];
for (let index = 0; index < request.frames.length; index += 1) {
  current = [];
  const frame = structuredClone(request.frames[index]);
  const call = new vm.Script("__painter.renderFrame(__frame, __time, __state)");
  context.__painter = painter;
  context.__frame = frame;
  context.__time = index;
  context.__state = persistentState;
  call.runInContext(context, { timeout: request.vmTimeoutMs });
  operationsByFrame.push(current);
}
process.stdout.write(JSON.stringify({ operationsByFrame, elapsedMs: performance.now() - started }));

