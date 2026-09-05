export function createPainter(api, styleConfig) {
  return {
    reset(seed) { api.reset(seed); },
    renderFrame(frameFeatures, time, persistentState) {
      api.createPaper({color:'#f5efe1',grain:0});
      for (const s of frameFeatures.stroke_samples) {
        const pencil=styleConfig.medium==='graphite';
        const colors=pencil?['#334d70','#66728b','#b8ad98','#665246','#342723']:['#19355c','#307d96','#8ba98e','#d6ac51','#bc4937'];
        const color=colors[Math.min(4,Math.floor(s.value*5))];
        for(let pass=0;pass<2;pass+=1) {
          const angle=pencil?(pass===1?-.65:.65):Math.atan2(s.tz,s.tx)+pass*.18;
          const length=.043+.012*Math.abs(s.value-.5)*2;
          const dx=Math.cos(angle)*length/2,dz=Math.sin(angle)*length/2;
          const offset=(pass-.5)*.012;
          api.mark({sample_id:s.id,
            points:[[Math.max(0,Math.min(1,s.x-dx)),Math.max(0,Math.min(1,s.z-dz+offset))],[Math.max(0,Math.min(1,s.x+dx)),Math.max(0,Math.min(1,s.z+dz+offset))]],
            width:pencil?.0022:.018,opacity:.35+.4*Math.abs(s.value-.5)*2,
            pressure:.5+.5*s.value,texture:pencil?.55:.22,
            medium:pencil?'graphite':'bristle',color:color});
        }
      }
    }
  };
}
