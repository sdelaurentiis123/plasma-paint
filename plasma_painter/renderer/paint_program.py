"""Model-authored finite-mark programs compiled to the existing sandboxed JS API.

No artist presets. The compiler supplies lifecycle, field access and bounded tool
semantics; every layer, palette, path and medium is supplied by the program.
"""
import json
import math
from .dsl import STROKE_MEDIA


COEFFICIENTS = ('base','value','strength','signed')
PARAMETERS = {
    'length':(.002,.06), 'width':(.0005,.018), 'opacity':(0,.8),
    'pressure':(.05,1), 'texture':(0,1), 'angle_offset':(-math.pi,math.pi),
}


def number(value,low,high):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError(f'Expected finite number within [{low},{high}], got {value!r}')


def color(value):
    if not isinstance(value,str) or len(value)!=7 or value[0]!='#':raise ValueError('Colors must be #rrggbb')
    try:int(value[1:],16)
    except ValueError:raise ValueError('Colors must be #rrggbb') from None


def validate_paint_program(program):
    if not isinstance(program,dict) or set(program)!={'version','paper','layers'} or program['version']!=1:
        raise ValueError('Program requires exactly version:1, paper and layers')
    paper=program['paper']
    if not isinstance(paper,dict) or set(paper)!={'color','grain'}:raise ValueError('paper needs color and grain')
    color(paper['color']);number(paper['grain'],0,.02)
    layers=program['layers']
    if not isinstance(layers,list) or not 1<=len(layers)<=8:raise ValueError('Use 1..8 layers')
    budget=0
    fields={'tool','palette','color_field','path','direction','stride','phase','select'}|set(PARAMETERS)
    for layer in layers:
        if not isinstance(layer,dict) or set(layer)!=fields:raise ValueError('Layer requires exactly '+', '.join(sorted(fields)))
        if layer['tool'] not in STROKE_MEDIA:raise ValueError('Unknown painting tool')
        if layer['color_field'] not in ('value','strength'):raise ValueError('color_field is value or strength')
        if layer['direction'] not in ('tangent','fixed'):raise ValueError('direction is tangent (proxy) or fixed')
        if not isinstance(layer['palette'],list) or not 2<=len(layer['palette'])<=12:raise ValueError('Use 2..12 palette colors')
        for c in layer['palette']:color(c)
        stride=layer['stride'];phase=layer['phase']
        if type(stride)!=int or not 1<=stride<=16:raise ValueError('stride is integer 1..16')
        if type(phase)!=int or not 0<=phase<stride:raise ValueError('phase must be integer 0..stride-1')
        budget+=math.ceil(550/stride)
        selection=layer['select']
        if not isinstance(selection,list) or len(selection)!=2:raise ValueError('select is [minimum value,maximum value]')
        for v in selection:number(v,0,1)
        if selection[0]>=selection[1]:raise ValueError('select must be increasing')
        path=layer['path']
        if not isinstance(path,list) or not 2<=len(path)<=8:raise ValueError('path needs 2..8 local [x,z] points')
        for p in path:
            if not isinstance(p,list) or len(p)!=2:raise ValueError('path points must be pairs')
            for v in p:number(v,-1,1)
        arc=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
        if arc<.01:raise ValueError('path cannot be zero or near-zero length')
        for name in PARAMETERS:
            expr=layer[name]
            if isinstance(expr,dict):
                if not expr or set(expr)-set(COEFFICIENTS):raise ValueError('Expression supports only base,value,strength,signed')
                for v in expr.values():number(v,-10,10)
            else:number(expr,*PARAMETERS[name])
    if budget>1100:raise ValueError('Layer stride budget exceeds 1100 marks for a 550-sample frame; increase strides')
    return program


def expression(value,name):
    if not isinstance(value,dict):return repr(value)
    terms=[f'({value.get(k,0)})*{var}' for k,var in zip(COEFFICIENTS,('1','v','a','f'))]
    lo,hi=PARAMETERS[name]
    return f'Math.max({lo},Math.min({hi},'+ '+'.join(terms)+'))'


def compile_paint_program(program):
    validate_paint_program(program)
    blocks=[]
    for layer in program['layers']:
        path=layer['path'];arc=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
        center=[(max(p[i] for p in path)+min(p[i] for p in path))/2 for i in (0,1)]
        points=[[(p[i]-center[i])/arc for i in (0,1)] for p in path]
        palette=[[int(c[i:i+2],16) for i in (1,3,5)] for c in layer['palette']]
        angle=('Math.atan2(s.tz,s.tx)+' if layer['direction']=='tangent' else '')+expression(layer['angle_offset'],'angle_offset')
        colorvar='v' if layer['color_field']=='value' else 'a'
        attrs=','.join(f'{k}:{expression(layer[k],k)}' for k in ('width','opacity','pressure','texture'))
        blocks.append(f'''{{
          const shape={json.dumps(points)}, colors={json.dumps(palette)};
          for(const s of frameFeatures.stroke_samples) {{
            if(s.id%{layer['stride']}!=={layer['phase']} || s.value<{layer['select'][0]} || s.value>{layer['select'][1]}) continue;
            const v=s.value, f=2*v-1, a=Math.abs(f);
            const length={expression(layer['length'],'length')}, angle={angle};
            const c=Math.cos(angle), d=Math.sin(angle);
            const points=shape.map(p=>[Math.max(0,Math.min(1,s.x+length*(p[0]*c-p[1]*d))),Math.max(0,Math.min(1,s.z+length*(p[0]*d+p[1]*c)))]);
            const t=Math.max(0,Math.min(1,{colorvar}))*(colors.length-1), i=Math.min(colors.length-2,Math.floor(t)), u=t-i;
            const color='#'+colors[i].map((v,k)=>Math.round(v*(1-u)+colors[i+1][k]*u).toString(16).padStart(2,'0')).join('');
            api.mark({{sample_id:s.id,points,color,medium:{json.dumps(layer['tool'])},{attrs}}});
          }}
        }}''')
    return '''export function createPainter(api, styleConfig) {
      return { reset(seed) { api.reset(seed); }, renderFrame(frameFeatures,time,persistentState) {
        api.createPaper('''+json.dumps(program['paper'])+');\n'+'\n'.join(blocks)+'''\n      }};
    }\n'''


PROGRAM_CONTRACT = '''Return only a JSON painting program. No JavaScript and no prose.
The compiler provides reset, loops and the bounded tool API, but NO artist preset.
You choose every layer's tool, color, geometry and use of the field.
Top-level keys: version (1), paper ({color:'#rrggbb', grain:0..0.02}), layers (1..8 objects).
Every layer requires ALL these keys:
tool: watercolor | bristle | graphite | charcoal | ink | pastel
palette: 2..12 actual #rrggbb colors you choose (smooth interpolation, low to high)
color_field: value | strength
path: 2..8 local [x,z] points, coordinates in [-1,1], not all equal.
The compiler centers this local shape, scales its TOTAL ARC LENGTH to length,
rotates it, and anchors it at a real plasma sample. You design the stroke shape.
direction: tangent | fixed. Tangent is the scalar contour tangent, a display proxy, NOT flow.
stride: integer 1..16; phase: integer 0..stride-1. Selects sample IDs modulo stride.
Sum of ceil(550/stride) across layers must be <=1100. Budget layers intentionally.
select: [minimum value,maximum value], increasing within [0,1].
length: .002.. .06; width: .0005.. .018; opacity:0.. .8; pressure:.05..1;
texture:0..1; angle_offset: -3.14159..3.14159 radians.
Each of those six parameters can be a constant OR an expression object with any
of base,value,strength,signed coefficients (each -10..10).
Expression = base + value*v + strength*abs(2*v-1) + signed*(2*v-1).
The tool clamps expression results to its documented range. All path points are
clamped to the normalized domain and must still pass the existing physical-anchor
and finite-path validation. Invalid programs are rejected, not painted silently.
v is normalized signed density fluctuation, [0,1], physical zero at .5.
Negative and positive structures must stay distinguishable; stronger structures
should receive intentional pigment emphasis. Every mark stays tied to real data.
Use data-responsive parameter expressions; do not paint an unchanging pattern.
Avoid arbitrary noise. Tools have different physical textures; select tools and
layer order for the observed reference style, not a generic rainbow palette.
There is no provided painting to copy. Output only the complete JSON object.
'''
