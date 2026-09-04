const $ = id => document.getElementById(id);
let data, selected = 0, offset = 0, playing = false, last = 0, revision = 0;
const images = new Map();
function loadImage(url) {
  if (!images.has(url)) images.set(url, new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url;
  }));
  return images.get(url);
}
async function draw() {
  const ticket = ++revision, clip = data.clips[selected], frame = clip.frames[offset];
  const pair = await Promise.all([loadImage(frame.painting), loadImage(frame.scientific)]);
  if (ticket !== revision) return;
  $('painting').src = pair[0].src; $('scientific').src = pair[1].src;
  $('position').textContent = `85604 · frame ${frame.index}`;
  $('frame').value = offset;
}
function stop() { playing = false; $('play').textContent = 'Play'; }
function tick(now) {
  if (playing && now - last > 420) { last = now; offset = (offset + 1) % data.clips[selected].frames.length; draw().catch(fail); }
  requestAnimationFrame(tick);
}
function fail() { stop(); $('error').textContent = 'Could not load the recorded results. Rebuild the local demo and refresh.'; }
try {
  const response = await fetch('results.generated.json');
  if (!response.ok) throw new Error('Missing results');
  data = await response.json();
  data.clips.forEach((clip, i) => { const option = document.createElement('option'); option.value = i; option.textContent = `${clip.split.replace('art_', '')} · ${clip.frames[0].index}–${clip.frames.at(-1).index}`; $('clip').append(option); });
  $('scope').textContent = `${data.evaluation_clips} independent test clips · ${data.seeds.length} matched seeds · 8 frames per clip. Baseline only.`;
  for (const [value, label] of [[`${(100*data.valid_rate).toFixed(0)}%`, 'Valid renders'], [data.metrics.coarse_spearman.mean.toFixed(3), 'Pixel intensity correlation'], [data.metrics.temporal.mean.toFixed(3), 'Temporal score']]) {
    const item = document.createElement('div'); item.className = 'metric'; const strong = document.createElement('strong'); strong.textContent = value; const span = document.createElement('span'); span.textContent = label; item.append(strong, span); $('metrics').append(item);
  }
  for (const [key, metric] of Object.entries(data.metrics)) { const row = document.createElement('tr'); for (const value of [key.replaceAll('_', ' '), metric.mean.toFixed(3), `${metric.low.toFixed(3)}–${metric.high.toFixed(3)}`]) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); } $('details').append(row); }
  data.methods.forEach(method => { const p = document.createElement('p'); p.textContent = `${method.name}: ${method.status.replaceAll('_', ' ')}`; $('methods').append(p); });
  $('hash').textContent = `Recorded evaluation program: ${data.program_hash}. Median low-resolution render: ${data.median_render_ms.toFixed(1)} ms per clip.`;
  await draw();
  for (const id of ['clip','frame','play']) $(id).disabled = false;
  $('clip').onchange = async () => { stop(); selected = Number($('clip').value); offset = 0; $('frame').max = data.clips[selected].frames.length-1; await draw().catch(fail); };
  $('frame').oninput = () => { stop(); offset = Number($('frame').value); draw().catch(fail); };
  $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  // Start paused for everyone; explicit play also respects reduced-motion preferences.
  requestAnimationFrame(tick);
} catch { fail(); }
