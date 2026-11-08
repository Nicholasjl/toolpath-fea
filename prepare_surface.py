"""Explicit boolean union of overlapping solid components, for bonded-part FEA.

This changes interface assumptions: overlapping parts become perfectly bonded.
Never use for movable assemblies or cavity shells. Source files stay untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh


def union_solids(surface):
    parts=surface.split(only_watertight=False,repair=False)
    if not parts or any(not p.is_watertight or not p.is_winding_consistent or p.volume<=0 for p in parts):
        raise ValueError('Union requires closed, consistently oriented positive-volume solids; no automatic hole filling or cavity conversion')
    result=trimesh.boolean.union(parts,engine='manifold',check_volume=True)
    if not result.is_watertight or not result.is_winding_consistent or result.volume<=0:
        raise ValueError('Union did not produce a valid solid')
    if len(result.split(only_watertight=False,repair=False))!=1:
        raise ValueError('Union remains disconnected; cannot glue a gap or model assembly contact')
    if not np.allclose(result.bounds,surface.bounds,rtol=0,atol=1e-4):raise ValueError('Union changed exterior bounds')
    total=sum(float(p.volume) for p in parts)
    if result.volume>total*(1+1e-6) or result.volume<max(p.volume for p in parts)*(1-1e-6):
        raise ValueError('Union volume is inconsistent with input solids')
    return result,dict(method='explicit_manifold_boolean_union',interface_assumption='perfectly_bonded_overlap',
        source_components=len(parts),output_components=1,source_summed_volume_mm3=total,
        union_volume_mm3=float(result.volume),overlap_volume_removed_mm3=total-float(result.volume),
        bounds_mm=result.bounds.tolist(),source_faces=len(surface.faces),output_faces=len(result.faces))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.source.resolve()==args.output.resolve():raise ValueError('Cannot overwrite source')
    if args.output.suffix.lower()!='.stl':raise ValueError('Output must be an STL')
    before=hashlib.sha256(args.source.read_bytes()).hexdigest()
    result,audit=union_solids(trimesh.load_mesh(args.source))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.export(args.output,file_type='stl_ascii')
    audit.update(source=str(args.source.resolve()),source_sha256=before,
        output=str(args.output.resolve()),output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    if hashlib.sha256(args.source.read_bytes()).hexdigest()!=before:raise RuntimeError('Source changed during preparation')
    args.output.with_suffix('.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
