"""Opt-in TetWild volume meshing with independent geometry/quality audits.

This backend changes the analysis surface only inside an explicit envelope.
It never repairs an open input silently and never relaxes volume/topology checks.
"""
import importlib.metadata
import numpy as np


def oriented_boundary(nodes,cells):
    from solid import boundary_faces
    faces,owners=boundary_faces(cells)
    xyz=nodes[faces];normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    inward=nodes[cells[owners]].mean(axis=1)-xyz.mean(axis=1)
    flip=np.einsum('ij,ij->i',normal,inward)>0
    faces[flip]=faces[flip][:,[0,2,1]]
    return faces


def surface_audit(source,nodes,cells,tolerance,samples=8192):
    import trimesh
    faces=oriented_boundary(nodes,cells)
    output=trimesh.Trimesh(nodes,faces,process=False);output.remove_unreferenced_vertices()
    if not output.is_watertight or not output.is_winding_consistent:
        raise ValueError('稳健体网格边界未闭合或朝向不一致')
    before=source.split(only_watertight=False,repair=False);after=output.split(only_watertight=False,repair=False)
    if sorted(x.euler_number for x in before)!=sorted(x.euler_number for x in after) or sum(x.volume<0 for x in before)!=sum(x.volume<0 for x in after):
        raise ValueError('稳健体网格改变孔道、内腔或分量拓扑')
    distances=[]
    for a,b in [(source,output),(output,source)]:
        rng=np.random.default_rng(20261116)
        ids=rng.choice(len(a.faces),samples,p=a.area_faces/a.area)
        uv=rng.random((samples,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
        tri=a.triangles[ids];points=tri[:,0]+uv[:,:1]*(tri[:,1]-tri[:,0])+uv[:,1:]*(tri[:,2]-tri[:,0])
        # Bound proximity-query memory for large, finely faceted plates.
        worst=0.
        for chunk in np.array_split(points,max(1,int(np.ceil(samples/256)))):
            _,d,_=trimesh.proximity.closest_point(b,chunk)
            if not np.isfinite(d).all():raise ValueError('表面距离校核产生非有限值')
            worst=max(worst,float(d.max()))
        distances.append(worst)
    if max(distances)>tolerance*1.01+1e-8:raise ValueError(f'稳健体网格表面偏差超限：{distances} > {tolerance} mm')
    return dict(topology_preserved=True,bidirectional_sample_max_mm=distances,samples_per_direction=samples,
                tolerance_mm=tolerance,hausdorff_verified=False,euler_number=int(output.euler_number),cavities=int(sum(x.volume<0 for x in before)))


def tetra_mesh_robust(path,pitch,snap_tolerance=.0001,surface_tolerance_mm=.005,max_elements=600000,mode='modern',stop_energy=None,max_iterations=80,quality_policy='target'):
    import trimesh
    if quality_policy not in ('target','measured'):raise ValueError('quality_policy must be target/measured')
    if quality_policy=='measured' and mode!='fast':raise ValueError('measured quality policy currently requires fast backend')
    target=(100 if mode=='legacy' else 10) if stop_energy is None else stop_energy
    if isinstance(target,bool) or not np.isfinite(target) or not 10<=target<=100:raise ValueError('TetWild stop_energy must be in [10,100]')
    if isinstance(max_iterations,bool) or not isinstance(max_iterations,int) or not 1<=max_iterations<=200:raise ValueError('TetWild max_iterations must be in [1,200]')
    if mode not in ('modern','legacy','fast'):raise ValueError('TetWild mode must be modern/legacy/fast')
    if not np.isfinite(pitch) or pitch<=0:raise ValueError('pitch 必须为正有限数')
    if not np.isfinite(surface_tolerance_mm) or not 0<surface_tolerance_mm<=.015:raise ValueError('表面包络范围为 (0,0.015] mm')
    if not np.isfinite(snap_tolerance) or not 0<=snap_tolerance<=.01:raise ValueError('坐标量化范围为 0–0.01 mm')
    source=trimesh.load_mesh(path)
    if not isinstance(source,trimesh.Trimesh) or not np.isfinite(source.vertices).all():raise ValueError('输入必须为有限三角网格')
    source.merge_vertices();source.update_faces(source.unique_faces());source.update_faces(source.nondegenerate_faces());source.remove_unreferenced_vertices()
    source_closed=bool(source.is_watertight and source.is_winding_consistent)
    original_volume=float(source.volume)
    if original_volume<=0:raise ValueError('输入总体积必须为正，且保留内腔朝向')
    surface=source.copy()
    if snap_tolerance:
        surface.vertices=np.round(surface.vertices/snap_tolerance)*snap_tolerance
        surface.merge_vertices();surface.update_faces(surface.unique_faces());surface.update_faces(surface.nondegenerate_faces());surface.remove_unreferenced_vertices()
    if not surface.is_watertight or not surface.is_winding_consistent:raise ValueError('稳健网格仍要求闭合且朝向一致的输入；未自动修补')
    snap_displacement_bound=float(np.sqrt(3)*snap_tolerance/2)
    envelope=surface_tolerance_mm-snap_displacement_bound
    if envelope<=0:raise ValueError('表面容差必须大于坐标量化的最大位移')
    snapped_error=abs(surface.volume/original_volume-1)
    if snapped_error>.001:raise ValueError('坐标清理体积变化超过 0.1%')
    parts=surface.split(only_watertight=False,repair=False)
    if any(not x.is_watertight for x in parts) or sum(x.volume>0 for x in parts)!=1:raise ValueError('仅支持单个连通实体及其内部空腔')
    import pymeshlab as ml
    check=ml.MeshSet();check.add_mesh(ml.Mesh(vertex_matrix=surface.vertices,face_matrix=surface.faces.astype(np.int32)))
    check.compute_selection_by_self_intersections_per_face()
    if check.current_mesh().face_selection_array().any():raise ValueError('输入表面相交，拒绝静默修复')
    diagonal=float(np.linalg.norm(surface.extents))
    opts=dict(input=['surface'],filter='tracked',eps_rel=envelope/diagonal,length_rel=pitch/diagonal,
              stop_energy=target,num_threads=4,preserve_topology=True,max_iterations=max_iterations,throw_on_fail=True,write_vtu=False)
    if mode=='legacy':
        # The legacy tracked-surface writeback loses cavity-component orientation.
        # Use input winding for negative shells and audit the resulting boundary.
        opts.update(skip_simplify=True,use_sample_envelope=True,use_legacy_code=True,
                    filter='input' if any(p.volume<0 for p in parts) else 'tracked')
    if mode=='fast':
        import pytetwild
        opts=dict(edge_length_abs=pitch,epsilon=envelope/diagonal,optimize=True,simplify=False,
                  coarsen=False,stop_energy=target,num_opt_iter=max_iterations,num_threads=4,loglevel=2,quiet=False)
        try:nodes,cells=pytetwild.tetrahedralize(np.asarray(surface.vertices),np.asarray(surface.faces,dtype=np.int32),**opts)
        except RuntimeError as exc:raise ValueError(f'fTetWild 生成失败：{exc}') from exc
        nodes=np.asarray(nodes,dtype=float);cells=np.asarray(cells,dtype=np.int32);report={}
    else:
        import wildmeshing as wm
        mesh={'V':np.asarray(surface.vertices),'cells':{'triangle':{'connectivity':np.asarray(surface.faces,dtype=np.int64)}}}
        try:outputs,report=wm.tetwild(opts,{'surface':mesh})
        except RuntimeError as exc:raise ValueError(f'TetWild 生成失败：{exc}') from exc
        result=outputs['mesh'];nodes=np.asarray(result['V'],dtype=float);cells=np.asarray(result['cells']['tetrahedron']['connectivity'],dtype=np.int32)
    if not len(cells) or len(cells)>max_elements:raise ValueError(f'贴体单元数量 {len(cells)} 超出限制 {max_elements}')
    used,inverse=np.unique(cells,return_inverse=True);nodes=nodes[used];cells=inverse.reshape(-1,4).astype(np.int32)
    xyz=nodes[cells];volumes=np.abs(np.linalg.det(xyz[:,1:]-xyz[:,:1]))/6
    if not np.isfinite(nodes).all() or not np.isfinite(volumes).all() or np.any(volumes<=1e-16):raise ValueError('稳健网格含非有限或退化单元')
    error=abs(volumes.sum()/original_volume-1)
    if error>.002:raise ValueError(f'稳健体网格体积误差 {error:.6g} 超过 0.2%')
    lengths=sum(np.sum((xyz[:,a]-xyz[:,b])**2,axis=1) for a in range(4) for b in range(a))
    quality=12*(3*volumes)**(2/3)/lengths
    if mode=='fast':report=dict(max_energy=float(3/quality.min()),measurement='independent_3_over_mean_ratio',all_rounded=True)
    target_met=bool(report.get('all_rounded') and report.get('max_energy',float('inf'))<=opts['stop_energy']*1.00001)
    if quality_policy=='target' and not target_met:raise ValueError('TetWild 未达到质量优化目标')
    if quality_policy=='measured' and (quality.min()<.001 or np.mean(quality<.01)>.001):
        raise ValueError('TetWild 实测质量未达标：minimum >= 0.001, fraction below 0.01 <= 0.001')
    report.update(optimization_target_met=target_met,quality_policy=quality_policy)
    audit=surface_audit(source if source_closed else surface,nodes,cells,surface_tolerance_mm)
    audit['reference']='original_cleaned_surface' if source_closed else 'snapped_closed_surface'
    audit['snap_displacement_bound_mm']=snap_displacement_bound
    audit['envelope_method']='sampled' if mode in ('legacy','fast') else 'exact'
    ordered=np.sort(volumes);p95=float(ordered[np.searchsorted(np.cumsum(ordered),.95*ordered.sum())])
    return nodes,cells,dict(mode='tetra',backend='ftetwild' if mode=='fast' else 'tetwild',backend_version=importlib.metadata.version('pytetwild' if mode=='fast' else 'wildmeshing'),
        source_volume_mm3=original_volume,volume_relative_error=float(error),snap_tolerance_mm=snap_tolerance,
        snapped_volume_relative_change=float(snapped_error),source_closed_before_snap=source_closed,surface_faces=len(surface.faces),
        surface_conditioning=audit,backend_options=opts,backend_report=report,max_volume_target_mm3=pitch**3/6,
        max_volume_actual_mm3=float(volumes.max()),volume_p95_mm3=float(np.percentile(volumes,95)),volume_weighted_p95_mm3=p95,
        quality_refinement=True,boundary_refinement=True,quality_min=float(quality.min()),quality_p01=float(np.percentile(quality,1)),
        low_quality_fraction=float(np.mean(quality<.01)),cavities=audit['cavities'],numeric_flat_tets_removed=0,numeric_flat_volume_mm3=0.)
