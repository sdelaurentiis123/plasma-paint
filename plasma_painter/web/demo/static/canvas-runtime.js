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
  color(value) {
    const ramps = {
      density: [[24,35,111],[34,91,172],[12,157,160],[113,190,110],[241,196,62],[224,81,47]],
      density_fluctuation: [[39,46,132],[70,136,193],[186,221,208],[247,225,164],[225,116,62],[150,35,73]],
      potential: [[54,24,107],[117,49,145],[199,68,117],[238,128,76],[249,204,94]],
      electron_temperature: [[24,48,105],[29,125,151],[104,188,140],[237,197,70],[199,58,44]]
    };
    const ramp = ramps[this.field] || ramps.density;
    const position = Math.max(0,Math.min(1,value))*(ramp.length-1), i = Math.min(ramp.length-2,Math.floor(position)), t = position-i;
    return ramp[i].map((v,k)=>Math.round(v*(1-t)+ramp[i+1][k]*t));
  }
  drawRaster(frame, opacity, bleed) {
    const r = this.raster(frame.rasters[this.field] || frame.rasters.density), tiny = document.createElement('canvas'); tiny.width = r.nx; tiny.height = r.nz;
    const context = tiny.getContext('2d'), image = context.createImageData(r.nx, r.nz);
    for (let x = 0; x < r.nx; x += 1) for (let z = 0; z < r.nz; z += 1) { const value = r.bytes[x * r.nz + z] / 255, i = (z * r.nx + x) * 4; image.data[i] = 35 + 179 * value; image.data[i + 1] = 58 + 112 * value; image.data[i + 2] = 77 + 21 * value; image.data[i + 3] = (45 + 210 * value) * opacity; }
    if (this.style.vibrant) for (let x=0;x<r.nx;x++) for(let z=0;z<r.nz;z++) { const value=r.bytes[x*r.nz+z]/255,i=(z*r.nx+x)*4; image.data.set(this.color(value),i); image.data[i+3]=255*Math.min(.85,this.style.washStrength ?? .68); }
    context.putImageData(image, 0, 0); this.ctx.save(); this.ctx.filter = `blur(${1 + bleed * 4}px)`; this.ctx.imageSmoothingEnabled = true; this.ctx.drawImage(tiny, 0, 0, this.canvas.width, this.canvas.height); this.ctx.restore();
  }
  scientificFrame(frame) {
    const r = this.raster(frame.rasters[this.field] || frame.rasters.density), tiny = document.createElement('canvas'); tiny.width = r.nx; tiny.height = r.nz;
    const context = tiny.getContext('2d'), image = context.createImageData(r.nx, r.nz);
    for (let x = 0; x < r.nx; x += 1) for (let z = 0; z < r.nz; z += 1) { const value = r.bytes[x * r.nz + z], i = (z * r.nx + x) * 4; image.data[i] = image.data[i + 1] = image.data[i + 2] = value; image.data[i + 3] = 255; }
    if(this.style.vibrant) for(let x=0;x<r.nx;x++) for(let z=0;z<r.nz;z++) { const i=(z*r.nx+x)*4; image.data.set(this.color(r.bytes[x*r.nz+z]/255),i); }
    context.putImageData(image, 0, 0); this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); this.ctx.drawImage(tiny, 0, 0, this.canvas.width, this.canvas.height);
    this.ctx.strokeStyle = '#c94d36'; this.ctx.lineWidth = 2; const x = frame.geometry.separatrix_face_u * this.canvas.width; this.ctx.beginPath(); this.ctx.moveTo(x, 0); this.ctx.lineTo(x, this.canvas.height); this.ctx.stroke();
  }
  render(frame, operations) {
    if (this.scientific) { this.scientificFrame(frame); return; }
    if (this.style.sketch) { this.sketchFrame(frame); return; }
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
  sketchFrame(frame) {
    const ctx=this.ctx,w=this.canvas.width,h=this.canvas.height;
    ctx.fillStyle='#f5f1e7';ctx.fillRect(0,0,w,h);
    ctx.lineCap='round';ctx.lineJoin='round';
    for(const contour of frame.contours){
      ctx.strokeStyle=contour.sign>0?'rgba(44,43,46,.64)':'rgba(78,91,116,.46)';
      ctx.lineWidth=contour.sign>0?1.25:.8;
      ctx.setLineDash(contour.sign>0?[]:[5,2]);
      for(let pass=0;pass<2;pass++){
        ctx.beginPath();contour.points.forEach((p,i)=>{const q=this.point(p),d=pass*.45;i?ctx.lineTo(q[0]+d,q[1]+d):ctx.moveTo(q[0]+d,q[1]+d);});ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    for(const v of frame.vectors.density_gradient){
      ctx.strokeStyle=`rgba(50,52,57,${.08+.22*v.magnitude})`;ctx.lineWidth=.65;
      const q=this.point([v.x,v.z]),length=3+v.magnitude*11;
      ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(q[0]+v.dx*length,q[1]+v.dz*length);ctx.stroke();
    }
    for(const f of frame.filaments){const q=this.point(f.centroid),r=Math.max(3,Math.sqrt(f.area_fraction)*h*.65);
      ctx.strokeStyle=f.sign>0?'rgba(49,48,48,.28)':'rgba(62,84,124,.25)';ctx.lineWidth=.6;
      for(let k=-r;k<=r;k+=3){const half=Math.sqrt(Math.max(0,r*r-k*k));ctx.beginPath();ctx.moveTo(q[0]-half,q[1]+k);ctx.lineTo(q[0]+half,q[1]+k);ctx.stroke();}
    }
    ctx.strokeStyle='rgba(130,88,66,.35)';ctx.lineWidth=.8;const x=frame.geometry.separatrix_face_u*w;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
  }
}
