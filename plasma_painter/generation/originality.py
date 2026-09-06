"""Detect copied mark construction; not an aesthetic or scientific fidelity score."""
from plasma_painter.config import stable_hash
from plasma_painter.features.stroke_samples import with_stroke_samples
from plasma_painter.renderer.sandbox import run_program


def construction_signature(operations):
    # Deliberately ignore color/paper: palette substitution alone is not new mark-making.
    return stable_hash([[{k:v for k,v in op['args'].items() if k != 'color'}
                         for op in frame if op['op']=='mark'] for frame in operations])


def check_construction(code, reference, frames, seed=1701):
    frames=[with_stroke_samples(f) for f in frames]
    candidate=run_program(code,frames,seed=seed,profile='stroke_only')
    control=run_program(reference,frames,seed=seed,profile='stroke_only')
    if not control.valid: raise RuntimeError('Originality control failed: '+str(control.error))
    if not candidate.valid: return {'different_construction':False,'error':candidate.error}
    candidate_hash=construction_signature(candidate.operations_by_frame)
    control_hash=construction_signature(control.operations_by_frame)
    return {'different_construction':candidate_hash!=control_hash,
            'candidate_construction_hash':candidate_hash,'control_construction_hash':control_hash,
            'scope':'exact construction-copy detection only; not artistic quality, fidelity, or sufficient diversity'}
