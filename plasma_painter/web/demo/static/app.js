import {BrowserCanvasRuntime} from './canvas-runtime.js';
import {createPainter} from './painter.generated.js';

const style = {paper: '#f2ede2', grain: .045, bleed: .35, persistence: .86};
const canvas = document.querySelector('#plasma-canvas');
const runtime = new BrowserCanvasRuntime(canvas, style, window.PLASMA_BOOTSTRAP.seed);
const operations = [];
const api = {reset(seed) { runtime.seed = seed >>> 0; }};
for (const name of ['createPaper', 'setPalette', 'washRegion', 'strokePath', 'dryBrushPath', 'dab', 'poolPigment', 'scatterGrain', 'fadeLayer', 'composite']) api[name] = (args = {}) => operations.push({op: name, args});
const painter = createPainter(Object.freeze(api), Object.freeze(style));
const state = {}; painter.reset(window.PLASMA_BOOTSTRAP.seed);
let clip = window.PLASMA_BOOTSTRAP, index = 0;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
let playing = !reducedMotion, timer = null;

function draw() {
  const frame = clip.frames[index]; if (!frame) return;
  operations.length = 0; painter.renderFrame(frame, index, state); runtime.render(frame, operations);
  document.querySelector('#scrubber').value = index;
  document.querySelector('#frame-caption').textContent = `TCV shot ${frame.source.shot} · frame ${frame.source.frame_index} · ${frame.time.microseconds.toFixed(2)} μs · fixed y=18 plane`;
  if (!reducedMotion) document.querySelector('.fallback').style.visibility = 'hidden';
}
function schedule() { clearInterval(timer); if (playing) timer = setInterval(() => { index = (index + 1) % clip.frames.length; draw(); }, 640); }
draw();
fetch('features.generated.json').then(response => response.json()).then(data => { clip = data; document.querySelector('#scrubber').max = clip.frames.length - 1; draw(); schedule(); });
document.querySelector('#play').onclick = () => { playing = !playing; document.querySelector('#play').textContent = playing ? 'Pause' : 'Play'; schedule(); };
document.querySelector('#scrubber').oninput = event => { index = Number(event.target.value); draw(); };
document.querySelector('#field').onchange = event => { runtime.field = event.target.value; draw(); };
document.querySelector('#scientific').onclick = event => { runtime.scientific = !runtime.scientific; event.target.textContent = runtime.scientific ? 'View painting' : 'View scientific rendering'; draw(); };
const dialog = document.querySelector('#code-dialog');
document.querySelector('#code').onclick = async () => { document.querySelector('#program-source').textContent = await fetch('painter.generated.js').then(response => response.text()); dialog.showModal(); };
document.querySelector('#close-code').onclick = () => dialog.close();
