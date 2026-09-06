"""Free placement, generic tools, and no supplied painting algorithm."""

CONTRACT = '''Write one JavaScript function renderFrame(frameFeatures,time,persistentState).
Do not write export, createPainter or reset: the trusted wrapper supplies those.
Do not return an image description. Write a reusable painting algorithm.

You are free to choose stroke positions and paths ANYWHERE in normalized canvas XY.
There is no sample grid, mandatory anchor, artist preset, supplied palette, or stroke recipe.
Canvas X maps to radial x; canvas Y maps to periodic field-aligned z. These are NOT
Cartesian R-Z coordinates. The same program must work on different field-aligned
cross-sections and times; never memorize the depicted frame or paint its objects.

Generic tools:
api.createPaper({color:'#rrggbb',grain:0..0.02}): exactly once per frame.
api.paintStroke({points:[[x,y],...],medium:TOOL,color:'#rrggbb',width:NUMBER,
  opacity:NUMBER,pressure:NUMBER,texture:NUMBER,stroke_id:INTEGER}): one finite stroke.
TOOL: watercolor, bristle, graphite, charcoal, ink, pastel. Choose tools to fit the
reference style; combine tools and layers if appropriate. No default artist brush.
Choose 2..64 points, all in [0,1]; total polyline length .0005..2 domain units.
Width .0005.. .08; opacity 0.. .8; pressure .05..1; texture 0..1.
The medium choice and color are required. Generic pressure/texture defaults are .7/.4
if omitted; you may override them to suit the chosen tool and reference style.
At most 1100 strokes per frame. stroke_id is an optional stable integer 0..9999999
for texture continuity, NOT a data/sample ID and imposes NO placement restriction.

Read the real field wherever YOU choose:
api.sample('density_fluctuation',x,y) returns normalized scalar [0,1]. Neutral=.5,
negative below .5, positive above .5. Density, potential, electron_temperature are
also available by those names. api.sample('density',x,y) gives normalized density.
api.gradient(field,x,y) returns {dx,dz} in normalized display coordinates; contour
tangent angle is atan2(dx,-dz). This is a visualization proxy, NOT physical flow.
Sampling is bilinear; radial boundaries use one-sided gradients; z wraps periodically
for derivative neighbors. Coordinates are within [0,1]. Max 40000 query samples/frame;
each gradient costs four query samples, and each sample call costs one.

frameFeatures.contours is an array of {points:[[x,y],...],sign:-1|1,level:number}.
frameFeatures.filaments is an array of {centroid:[x,y],area_fraction,peak,sign,track_id}.
These are optional evidence, not instructions to stroke all contours or plot all centers.
frameFeatures.geometry describes the cross-section. There are no stroke_samples,
value_range, or canvas properties. Query data with the read-only functions above.
The style images are seen by YOU, not available as files or image objects to the code.

Use finite, bounded for loops, ordinary arrays, Math and local helper functions.
Seeded Math.random is available; the wrapper resets it to the same seed each frame
so paper/brush choices need not flicker. Do not reseed it yourself. Random noise is
not a substitute for plasma structures. Every major structure must correspond to
the field: use scalar queries to place, shape and weight marks, not just palette.
No imports, network, DOM, canvas access, timers, while/do loops, async, classes,
eval, Function, host globals, prototype access or dynamic property-name access.
Return the complete renderFrame function. Preserve sign, relative intensity and geometry in
the finished image; fidelity will be measured from the painting, not grid anchors.
'''


def prepare_response(raw):
    from plasma_painter.generation.sample_programs import _strip_fence
    from plasma_painter.renderer.javascript import parse_source
    parsed = parse_source(_strip_fence(raw), 'body')
    return wrap_body(parsed['normalized']), parsed['format']


def wrap_body(body):
    return '''export function createPainter(api, styleConfig) {
      let painterSeed=0;
      return {reset(seed){painterSeed=seed;api.reset(seed);},
        renderFrame(frameFeatures,time,persistentState){
          api.reset(painterSeed);
    '''+body+'''\n      }};
    }\n'''
