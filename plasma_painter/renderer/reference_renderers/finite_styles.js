export function createPainter(api, styleConfig) {
  return {
    reset(seed) { api.reset(seed); },
    renderFrame(frameFeatures, time, persistentState) {
      api.createPaper({color:'#f7f0df',grain:0});
      const mode=styleConfig.study;
      for(const s of frameFeatures.stroke_samples) {
        const v=Math.max(0,Math.min(1,.5+(s.value-.5)*2.2));
        const strength=Math.abs(v-.5)*2;
        const palette=mode==='pencil'?['#34455e','#657283','#c3b7a0','#746254','#342a28']:
          mode==='screenprint'?['#202659','#e04482','#f6c843','#ec6236','#24243f']:
          mode==='cubist'?['#234955','#618586','#d5bd78','#ae743e','#643c36']:
          mode==='tonal'?['#20374d','#5d797b','#c3b696','#b98857','#583b35']:
          ['#213a75','#2c8b99','#b7bb67','#edb541','#d05a35'];
        const color=palette[Math.min(4,Math.floor(v*5))];
        for(let pass=0;pass<2;pass+=1) {
          let angle=Math.atan2(s.tz,s.tx),length=.044,width=.016,medium='bristle',texture=.18;
          let offset=(pass-.5)*.014;
          if(mode==='pointillist'){length=.0025;width=.009+.006*strength;medium='ink';texture=0;angle=0;}
          if(mode==='cubist'){angle=Math.round(angle/(Math.PI/3))*Math.PI/3;length=.036;width=.013;medium='pastel';texture=.08;}
          if(mode==='tonal'){angle=pass===0?.12:-.12;length=.026+.02*strength;width=.018;texture=.06;}
          if(mode==='pencil'){angle=pass===0?.65:-.65;length=.05;width=.001+.002*strength;medium='graphite';texture=.35;}
          if(mode==='screenprint'){angle=0;length=.036;width=.012;medium='ink';texture=0;offset=(pass-.5)*.016;}
          const dx=Math.cos(angle)*length/2,dz=Math.sin(angle)*length/2;
          const shift=mode==='pointillist'?(pass-.5)*.02:0;
          const points=[[s.x+shift-dx,s.z-dz+offset]];
          if(mode==='directional')points.push([s.x-Math.sin(angle)*.006,s.z+Math.cos(angle)*.006+offset]);
          if(mode==='cubist')points.push([s.x+dx,s.z-dz+offset]);
          points.push([s.x+shift+dx,s.z+dz+offset]);
          const bounded=[];
          for(const p of points)bounded.push([Math.max(0,Math.min(1,p[0])),Math.max(0,Math.min(1,p[1]))]);
          api.mark({sample_id:s.id,points:bounded,width:width,opacity:mode==='screenprint'?.8:.5+.3*strength,
            pressure:.75+.25*strength,texture:texture,medium:medium,color:color});
        }
      }
    }
  };
}
