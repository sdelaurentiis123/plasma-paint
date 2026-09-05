"""Compact local scalar samples, not screenshots or invented flow vectors."""
import numpy as np
from .pipeline import decode_unit_raster


def with_stroke_samples(frame):
    if frame.get('masks',{}).get('invalid_cells',0):
        raise ValueError('finite-mark samples require an explicit cell mask when invalid cells exist')
    values=decode_unit_raster(frame['rasters']['density_fluctuation'])
    nx,nz=values.shape
    gx=np.gradient(values,axis=0)*(nx-1)
    gz=(np.roll(values,-1,axis=1)-np.roll(values,1,axis=1))*(nz-1)/2
    samples=[]
    for x in range(2,nx-2,3):
        for z in range(2,nz-2,3):
            dx,dz=float(gx[x,z]),float(gz[x,z]);norm=max(1e-9,float(np.hypot(dx,dz)))
            samples.append({'id':len(samples),'x':x/(nx-1),'z':z/(nz-1),
                            'value':float(values[x,z]),'tx':-dz/norm,'tz':dx/norm})
    if len(samples)>550: raise ValueError('sample cap exceeded; resample with a documented larger stride')
    return {**frame,'stroke_samples':samples,'stroke_sample_semantics':'signed density fluctuation encoded [0,1], zero at .5, using existing train statistics; tangent in normalized display coordinates is a visualization proxy, not flow'}
