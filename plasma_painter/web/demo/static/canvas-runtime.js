export class BrowserCanvasRuntime {
  constructor(canvas, style, seed) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d'); this.style = style;
    this.seed = seed >>> 0; this.field = 'density'; this.scientific = false;
    this.persistent = document.createElement('canvas');
    this.persistent.width = canvas.width; this.persistent.height = canvas.height;
  }
  random() { this.seed = (this.seed + 0x6d2b79f5) >>> 0; let t = this.seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
  point(p) { return [p[0] * this.canvas.width, p[1] * this.canvas.height]; }
  raster(record) { const raw = atob(record.data), bytes = new Uint8ClampedArray(raw.length); for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i); return {bytes, nx: record.shape[0], nz: record.shape[1]}; }
  drawRaster(frame, opacity, bleed) {
    const r = this.raster(frame.rasters[this.field] || frame.rasters.density), tiny = document.createElement('canvas'); tiny.width = r.nx; tiny.height = r.nz;
    const context = tiny.getContext('2d'), image = context.createImageData(r.nx, r.nz);
    for (let x = 0; x < r.nx; x += 1) for (let z = 0; z < r.nz; z += 1) { const value = r.bytes[x * r.nz + z] / 255, i = (z * r.nx + x) * 4; image.data[i] = 35 + 179 * value; image.data[i + 1] = 58 + 112 * value; image.data[i + 2] = 77 + 21 * value; image.data[i + 3] = (45 + 210 * value) * opacity; }
    context.putImageData(image, 0, 0); this.ctx.save(); this.ctx.filter = `blur(${1 + bleed * 4}px)`; this.ctx.imageSmoothingEnabled = true; this.ctx.drawImage(tiny, 0, 0, this.canvas.width, this.canvas.height); this.ctx.restore();
  }
  scientificFrame(frame) {
    const r = this.raster(frame.rasters[this.field] || frame.rasters.density), tiny = document.createElement('canvas'); tiny.width = r.nx; tiny.height = r.nz;
    const context = tiny.getContext('2d'), image = context.createImageData(r.nx, r.nz);
    for (let x = 0; x < r.nx; x += 1) for (let z = 0; z < r.nz; z += 1) { const value = r.bytes[x * r.nz + z], i = (z * r.nx + x) * 4; image.data[i] = image.data[i + 1] = image.data[i + 2] = value; image.data[i + 3] = 255; }
    context.putImageData(image, 0, 0); this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); this.ctx.drawImage(tiny, 0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = '#c94d36'; this.ctx.lineWidth = 2; const x = frame.geometry.separatrix_face_u * this.canvas.width; this.ctx.beginPath(); this.ctx.moveTo(x, 0); this.ctx.lineTo(x, this.canvas.height); this.ctx.stroke();
  }
  render(frame, operations) {
    if (this.scientific) { this.scientificFrame(frame); return; }
    let paper = this.style.paper, grain = this.style.grain;
    for (const item of operations) if (item.op === 'createPaper') { paper = item.args.color || paper; grain = item.args.grain ?? grain; }
    this.ctx.fillStyle = paper; this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    const image = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    for (let i = 0; i < image.data.length; i += 4) { const noise = (this.random() - .5) * grain * 110; image.data[i] += noise; image.data[i + 1] += noise; image.data[i + 2] += noise; }
    this.ctx.putImageData(image, 0, 0);
    for (const item of operations) {
      const args = item.args || {};
      if (item.op === 'washRegion') this.drawRaster(frame, args.opacity || .15, args.bleed || .2);
      else if (item.op === 'fadeLayer') { const p = this.persistent.getContext('2d'); p.globalCompositeOperation = 'destination-in'; p.fillStyle = `rgba(0,0,0,${args.retention ?? .85})`; p.fillRect(0, 0, this.canvas.width, this.canvas.height); p.globalCompositeOperation = 'source-over'; }
      else if (item.op === 'strokePath' || item.op === 'dryBrushPath') {
        const positive = (args.pigment || '').includes('positive'), negative = (args.pigment || '').includes('negative');
        this.ctx.strokeStyle = positive ? `rgba(148,66,56,${args.opacity || .2})` : negative ? `rgba(75,93,132,${args.opacity || .2})` : `rgba(42,70,72,${args.opacity || .15})`;
        this.ctx.lineWidth = Math.max(1, (args.width || .003) * this.canvas.height); this.ctx.beginPath(); args.points.forEach((point, i) => { const q = this.point(point); i ? this.ctx.lineTo(...q) : this.ctx.moveTo(...q); }); this.ctx.stroke();
      } else if (item.op === 'dab' || item.op === 'poolPigment') {
        const p = this.persistent.getContext('2d'), q = this.point(args.center), radius = (args.radius || .01) * this.canvas.height;
        const gradient = p.createRadialGradient(q[0], q[1], 0, q[0], q[1], radius * 2.2), color = (args.pigment || '').includes('negative') ? '75,93,132' : '148,66,56';
        gradient.addColorStop(0, `rgba(${color},${args.opacity || .2})`); gradient.addColorStop(.6, `rgba(${color},${(args.opacity || .2) * .45})`); gradient.addColorStop(1, `rgba(${color},0)`); p.fillStyle = gradient; p.fillRect(q[0] - radius * 2.2, q[1] - radius * 2.2, radius * 4.4, radius * 4.4);
      }
    }
    this.ctx.drawImage(this.persistent, 0, 0); this.ctx.strokeStyle = 'rgba(55,49,42,.18)'; this.ctx.lineWidth = 1;
    const separatrix = frame.geometry.separatrix_face_u * this.canvas.width; this.ctx.beginPath(); this.ctx.moveTo(separatrix, 0); this.ctx.lineTo(separatrix, this.canvas.height); this.ctx.stroke();
  }
}
