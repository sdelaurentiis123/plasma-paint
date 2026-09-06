"""Single-frame, JSON-action painting episodes. No model-written code is executed."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from plasma_painter.config import stable_hash
from plasma_painter.features.pipeline import decode_unit_raster
from plasma_painter.renderer.dsl import validate_operation, STROKE_MEDIA
from plasma_painter.renderer.canvas_runtime import CanvasRuntime
from plasma_painter.rewards.fidelity import pigment_density, _safe_spearman


TOOLS = {
    'paint': 'strokes: 1..16 objects with points (2..64 normalized XY pairs), medium, #rrggbb color, width .0005...08, opacity 0...8, optional pressure .05..1 and texture 0..1',
    'undo': 'Remove the most recent accepted paint batch.',
    'finish': 'End the episode.',
    'media': list(STROKE_MEDIA),
}


class PaintingGym:
    """reset/step API; RGB observations can feed a local vision policy or a human.

    Reward is explicitly diagnostic, NOT yet the scientific RL training gate.
    Each action costs one turn, including rejected actions and undo.
    """
    def __init__(self, frame, *, synthetic=False, seed=1701, size=(384,256),
                 max_turns=24, max_strokes=256):
        if not synthetic:
            src=frame.get('source',{})
            if str(src.get('shot'))!='85604' or not 0<=src.get('frame_index',-1)<288:
                raise ValueError('Gym currently permits only 85604 art_train frames [0,288)')
        if frame.get('masks',{}).get('invalid_cells',0):
            raise ValueError('Masked frame requires a cell-aware reward implementation')
        if not (32<=size[0]<=512 and 32<=size[1]<=512 and 1<=max_turns<=64 and 1<=max_strokes<=512):
            raise ValueError('Episode resource limits exceeded')
        self.frame=copy.deepcopy(frame);self.synthetic=synthetic;self.seed=seed
        self.size=size;self.max_turns=max_turns;self.max_strokes=max_strokes
        self.field=decode_unit_raster(frame['rasters']['density_fluctuation']).T
        self.paper={'op':'createPaper','args':{'color':'#f7f0df','grain':0}}
        self.reset()

    def reset(self):
        self.batches=[];self.events=[];self.turn=0;self.done=False
        return self.observe(),{'synthetic':self.synthetic,'reward_scope':'diagnostic_only'}

    @property
    def operations(self):
        return [copy.deepcopy(self.paper)]+[copy.deepcopy(op) for batch in self.batches for op in batch]

    def observe(self):
        canvas=CanvasRuntime(*self.size,{},self.seed).render_frame(self.frame,self.operations)
        # Fixed field view: signed fluctuation blue/neutral/red, not a painting target.
        v=self.field; rgb=np.stack([v,1-abs(2*v-1),1-v],axis=-1)
        science=Image.fromarray(np.uint8(rgb*255)).resize(self.size)
        ink=pigment_density(canvas,self.field.shape,self.paper['args']['color'])
        smooth=lambda a:gaussian_filter(a,2,mode=('wrap','nearest'))
        coarse=_safe_spearman(smooth(ink),smooth(abs(2*self.field-1)))
        return {'canvas':canvas,'scientific':science,'tools':copy.deepcopy(TOOLS),
                'turns_left':self.max_turns-self.turn,'strokes_left':self.max_strokes-len(self.operations)+1,
                'source':copy.deepcopy(self.frame.get('source',{})),
                'geometry':copy.deepcopy(self.frame.get('geometry',{})),
                'coarse_correspondence':coarse,'nonempty':bool(np.max(ink)>.01),
                'synthetic':self.synthetic}

    def step(self, action):
        if self.done:raise RuntimeError('Episode finished; reset before another action')
        before=self.observe()['coarse_correspondence'];self.turn+=1
        error=None;accepted=False
        try:
            if not isinstance(action,dict):raise ValueError('Action must be a JSON object')
            if len(json.dumps(action,allow_nan=False))>64000:raise ValueError('Action exceeds 64KB')
            kind=action.get('action')
            if kind=='paint':
                if set(action)!={'action','strokes'}:raise ValueError('paint accepts only action and strokes')
                strokes=action['strokes']
                if not isinstance(strokes,list) or not 1<=len(strokes)<=16:raise ValueError('Supply 1..16 strokes')
                if len(self.operations)-1+len(strokes)>self.max_strokes:raise ValueError('Stroke budget exceeded')
                batch=[]
                for stroke in strokes:
                    if not isinstance(stroke,dict) or set(stroke)-{'points','medium','color','width','opacity','pressure','texture'}:
                        raise ValueError('Unknown stroke properties; actions are data, never JavaScript')
                    op={'op':'paintStroke','args':copy.deepcopy(stroke)}
                    op['args']['stroke_id']=len(self.operations)+len(batch)
                    validate_operation(op,max_path_points=64);batch.append(op)
                self.batches.append(batch)
            elif kind=='undo' and set(action)=={'action'}:
                if not self.batches:raise ValueError('Nothing to undo')
                self.batches.pop()
            elif kind=='finish' and set(action)=={'action'}:self.done=True
            else:raise ValueError('Use paint, undo, or finish')
            accepted=True
        except (ValueError,TypeError) as exc:error=str(exc)
        truncated=self.turn>=self.max_turns and not self.done
        self.done=self.done or truncated
        observation=self.observe()
        reward=observation['coarse_correspondence']-before if accepted else -1.
        if self.done and not observation['nonempty']:reward=-1.
        info={'accepted':accepted,'error':error,'reward_scope':'diagnostic_only',
              'scientific_training_eligible':False,'full_fidelity_evaluated':False}
        self.events.append({'turn':self.turn,'action':copy.deepcopy(action),'reward':reward,**info,
                            'coarse_correspondence':observation['coarse_correspondence']})
        return observation,reward,self.done and not truncated,truncated,info

    def save(self, directory):
        out=Path(directory);out.mkdir(parents=True,exist_ok=False)
        observation=self.observe();observation['canvas'].save(out/'painting.png')
        observation['scientific'].save(out/'scientific.png')
        program={'version':'single-frame-gym-v1','seed':self.seed,'size':self.size,
                 'frame_hash':stable_hash(self.frame),'source':self.frame.get('source'),
                 'synthetic':self.synthetic,'operations':self.operations}
        digest=stable_hash(program)
        (out/'program.json').write_text(json.dumps(program,indent=2))
        (out/'trajectory.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
        (out/'manifest.json').write_text(json.dumps({'program_hash':digest,'done':self.done,
            'turns':self.turn,'stroke_count':len(self.operations)-1,'synthetic':self.synthetic,
            'coarse_correspondence':observation['coarse_correspondence'],
            'origin':'gym_episode','trained_policy':False,'full_fidelity_evaluated':False},indent=2))
        return out
