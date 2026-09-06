"""Artist-neutral interface examples, not a painting/composition template."""

INTERFACE_GUIDANCE = '''
Execution contract (mandatory): createPainter only constructs and returns the painter.
reset(seed) only resets state/randomness. All drawing calls, including createPaper,
belong INSIDE synchronous renderFrame. No async functions or promises.
The reference images are visible to YOU during inference, not to the generated code.
Never load, require, decode or process image files in JavaScript. Infer color and
mark-making relationships visually now and express them as constants/algorithms.
The renderer receives only api, styleConfig and frameFeatures, not image objects.

API syntax example ONLY, not a prescribed composition or style:
Inside renderFrame, paper is emitted with api.createPaper({color:'#f7f0df',grain:0});
Given an actual sample s from frameFeatures.stroke_samples whose x is in [.01,.99],
one legal mark call is:
api.mark({points:[[s.x-.003,s.z],[s.x+.003,s.z]],width:.002,
  opacity:.4,pressure:.6,texture:.3,medium:'graphite',color:'#345678',sample_id:s.id});
Each call takes ONE object, not an array of marks, and returns undefined.
points contains [x,z] PAIRS, never {x,z} objects or [x,z,r,g,b].
Use both s.x and s.z, and retain s.id; never make up sample IDs.
The encoded fluctuation s.value is in [0,1]: signed fluctuation is 2*s.value-1,
neutral is .5. Do not test s.value<0 to detect negative fluctuation.
Choose your own palette, media, paths, layers, selection and density of marks.
Keep every point in [0,1] and within .04 of its original sample anchor.
Check all calls against the contract before returning complete JavaScript.
'''


def repair_feedback(error):
    return (
        'The preceding program failed. Do not repeat it unchanged. Fix ALL contract '
        'violations, including factory/reset drawing, image access, batched marks, '
        'point shapes, sample IDs and bounds. Validator evidence: ' + str(error) +
        '\nReturn the complete corrected synchronous JavaScript program only.'
    )
