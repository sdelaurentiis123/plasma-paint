"""Field queries for free painting, without exposing a mandatory mark lattice."""
from .pipeline import decode_unit_raster


def query_fields(frame):
    if frame.get('masks',{}).get('invalid_cells',0):
        raise ValueError('Free field queries require a cell mask when invalid cells exist')
    fields={}
    for name,record in frame['rasters'].items():
        values=decode_unit_raster(record)
        if values.ndim!=2 or min(values.shape)<2 or values.size>200000:
            raise ValueError('Field query grid must be 2D, at least 2x2 and <=200000 cells')
        fields[name]={'nx':values.shape[0],'nz':values.shape[1],'values':values.ravel().tolist()}
    return fields
