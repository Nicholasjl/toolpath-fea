"""Content-addressed, opt-in cache of validated volume meshes (no pickle)."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import uuid
import numpy as np


def signature(case):
    keys=('pitch','mesh_mode','snap_tolerance','surface_remesh_mm','tetra_backend','surface_tolerance_mm','tetwild_mode','tetgen_radius_edge','max_elements','tetwild_stop_energy','tetwild_max_iterations','tetwild_quality_policy')
    values={k:case[k] for k in keys}
    # JSON's 2 and 2.0 describe the same mesh request.
    for k in ('pitch','snap_tolerance','surface_remesh_mm','surface_tolerance_mm','tetgen_radius_edge','tetwild_stop_energy'):
        if values[k] is not None:values[k]=float(values[k])
    digest=hashlib.sha256(json.dumps(values,sort_keys=True).encode())
    for path in [Path(case['stl'])]+[Path(__file__).with_name(x) for x in ('meshing.py','robust_meshing.py','solid.py','mesh_cache.py')]:
        digest.update(path.read_bytes())
    optional=(('pytetwild',) if case['tetwild_mode']=='fast' else ('wildmeshing',)) if case['tetra_backend']=='tetwild' else ()
    for package in ('numpy','trimesh','meshpy','pymeshlab')+optional:
        digest.update((package+importlib.metadata.version(package)).encode())
    return digest.hexdigest()


def checksum(nodes,cells,info):
    h=hashlib.sha256(nodes.tobytes());h.update(cells.tobytes());h.update(info.encode());return h.hexdigest()


def cached_mesh(case,generate):
    folder=case.get('mesh_cache_dir')
    if not folder or not case['stl'] or case['mesh_mode']!='tetra':return generate()
    key=signature(case);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/(key+'.npz')
    if path.exists():
        try:
            with np.load(path,allow_pickle=False) as data:
                n=data['nodes'];c=data['cells'];info=str(data['info'])
                if str(data['key'])!=key or str(data['checksum'])!=checksum(n,c,info):raise ValueError('mesh cache checksum mismatch')
                if n.ndim!=2 or n.shape[1]!=3 or c.ndim!=2 or c.shape[1]!=4 or not np.isfinite(n).all() or c.min()<0 or c.max()>=len(n):raise ValueError('invalid mesh cache arrays')
                return n,c,json.loads(info)
        except (ValueError,KeyError,OSError):
            # Invalid/incomplete caches are recomputed, never accepted as geometry.
            pass
    n,c,meta=generate();info=json.dumps(meta,sort_keys=True,allow_nan=False)
    temporary=folder/(key+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('wb') as stream:np.savez_compressed(stream,nodes=n,cells=c,info=info,key=key,checksum=checksum(n,c,info))
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return n,c,meta
