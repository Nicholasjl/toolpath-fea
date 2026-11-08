"""Conforming PLC tetrahedra via TetGen, with explicitly audited coordinate snapping."""
import numpy as np


def tetra_mesh(path,pitch,snap_tolerance=1e-4,max_elements=600000,surface_remesh_mm=0):
    import trimesh
    from meshpy.tet import MeshInfo,Options,build
    if not np.isfinite(pitch) or pitch<=0:raise ValueError('pitch 必须为正有限数')
    if not np.isfinite(snap_tolerance) or not 0<=snap_tolerance<=.01:raise ValueError('snap_tolerance 范围为 0–0.01 mm')
    def clean(mesh):
        # Never call fix_normals(multibody=True): it would flip cavity shells outward.
        mesh.merge_vertices();mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces());mesh.remove_unreferenced_vertices()
    source=trimesh.load_mesh(path);clean(source)
    source_closed=bool(source.is_watertight and source.is_winding_consistent)
    if not source_closed and not snap_tolerance:
        raise ValueError('贴体网格需要闭合且面朝向一致的输入')
    original_volume=abs(float(source.volume));surface=source.copy()
    if snap_tolerance:
        surface.vertices=np.round(surface.vertices/snap_tolerance)*snap_tolerance
        clean(surface)
    if not surface.is_watertight or not surface.is_winding_consistent:
        raise ValueError('坐标清理改变了闭合拓扑，请减小 snap_tolerance，未自动填补孔洞')
    if surface.volume<0:surface.invert()
    relative_change=abs(surface.volume-original_volume)/max(original_volume,1e-20)
    if relative_change>.001:raise ValueError('坐标清理体积变化超过 0.1%，请减小 snap_tolerance')
    remesh_info={}
    preconditioned=surface
    if not np.isfinite(surface_remesh_mm) or not 0<=surface_remesh_mm<=5:
        raise ValueError('surface_remesh_mm 范围为 0–5 mm')
    if surface_remesh_mm:
        surface,remesh_info=condition_surface(surface,surface_remesh_mm)
        if abs(surface.volume-original_volume)/max(original_volume,1e-20)>.001:
            raise ValueError('表面预处理累计体积变化超过 0.1%')
    def build_domain(candidate):
        parts=candidate.split(only_watertight=False,repair=False)
        if any(not p.is_watertight for p in parts):raise ValueError('输入表面存在未闭合分量')
        if sum(p.volume>0 for p in parts)!=1:raise ValueError('仅支持单个连通实体及其内部封闭空腔')
        seeds=[]
        for part in parts:
            if part.volume>=0:continue
            step=max(min(part.extents)*1e-5,1e-7)
            points=part.triangles_center[:100]+part.face_normals[:100]*step
            inside=part.contains(points)
            if not inside.any():raise ValueError('无法可靠确定内腔种子点')
            seeds.append(points[np.flatnonzero(inside)[0]].tolist())
        info=MeshInfo();info.set_points(candidate.vertices.tolist());info.set_facets(candidate.faces.tolist())
        if seeds:info.set_holes(seeds)
        return info,seeds

    attempts=[]
    info,holes=build_domain(surface)
    # Conditioning is opt-in. Retry a rejected PLC once, on a separately audited
    # smoothed copy of the ORIGINAL snapped surface, never on an accumulated edit.
    try:
        mesh=build(info,options=Options('pYQ'),max_volume=pitch**3/6)
        attempts.append(dict(smoothing=False,status='completed'))
    except RuntimeError as exc:
        attempts.append(dict(smoothing=False,status='rejected',error=str(exc)))
        if not surface_remesh_mm or not any(f'error code {code}' in str(exc) for code in (3,4,5)):
            raise ValueError(f'四面体生成失败（{exc}）；检查相交或近退化表面，可显式配置 surface_remesh_mm，不能据此判断结构强度') from exc
        surface,remesh_info=condition_surface(preconditioned,surface_remesh_mm,smoothing=True)
        if abs(surface.volume-original_volume)/max(original_volume,1e-20)>.001:
            raise ValueError('备用表面预处理累计体积变化超过 0.1%')
        info,holes=build_domain(surface)
        try:
            mesh=build(info,options=Options('pYQ'),max_volume=pitch**3/6)
            attempts.append(dict(smoothing=True,status='completed'))
        except RuntimeError as retry:
            raise ValueError(f'两种经校验的表面策略均被 TetGen 拒绝：{exc}；{retry}') from retry
    nodes=np.asarray(mesh.points,dtype=float);cells=np.asarray(mesh.elements,dtype=np.int32)
    if not len(cells) or len(cells)>max_elements:raise ValueError(f'贴体单元数量 {len(cells)} 超出限制 {max_elements}')
    used,inverse=np.unique(cells,return_inverse=True);nodes=nodes[used];cells=inverse.reshape(-1,4).astype(np.int32)
    p=nodes[cells];edges=p[:,1:]-p[:,:1];volumes=np.abs(np.linalg.det(edges))/6
    # TetGen may emit numerically coplanar tets. Remove only round-off-scale volume;
    # ordinary thin elements remain, and the solver rechecks face connectivity.
    flat=volumes<original_volume*1e-15
    removed_count=int(flat.sum());removed_volume=float(volumes[flat].sum())
    if removed_volume>original_volume*1e-10:raise ValueError('数值退化单元总体积过大，拒绝自动清理')
    if removed_count:
        cells=cells[~flat];volumes=volumes[~flat]
        used,inverse=np.unique(cells,return_inverse=True);nodes=nodes[used];cells=inverse.reshape(-1,4).astype(np.int32)
        p=nodes[cells]
    volume_error=abs(volumes.sum()/original_volume-1)
    if volume_error>.002:raise ValueError('四面体体积与原始 STL 偏差超过 0.2%，可能存在内腔或拓扑问题')
    # Mean-ratio quality: 1 for equilateral, approaches zero for degenerate tetrahedra.
    lengths=sum(np.sum((p[:,a]-p[:,b])**2,axis=1) for a in range(4) for b in range(a))
    quality=12*(3*volumes)**(2/3)/lengths
    return nodes,cells,dict(mode='tetra',source_volume_mm3=original_volume,volume_relative_error=float(volume_error),
        snap_tolerance_mm=snap_tolerance,snapped_volume_relative_change=float(relative_change),surface_faces=len(surface.faces),
        source_closed_before_snap=source_closed,surface_conditioning=remesh_info,tetgen_attempts=attempts,
        max_volume_target_mm3=pitch**3/6,max_volume_actual_mm3=float(volumes.max()),
        boundary_refinement=False,quality_min=float(quality.min()),quality_p01=float(np.percentile(quality,1)),
        low_quality_fraction=float(np.mean(quality<.01)),cavities=len(holes),
        numeric_flat_tets_removed=removed_count,numeric_flat_volume_mm3=removed_volume)


def condition_surface(surface,target_mm,smoothing=False):
    """Explicit analysis-only sliver conditioning; no filling or topology repair.

    Collapse/swap with constrained reprojection; no smoothing or edge splitting.
    The distance audit is deterministic sampling, not a Hausdorff proof.
    """
    import trimesh
    import pymeshlab as ml
    tolerance=.015
    before_volume=float(surface.volume)
    before_parts=len(surface.split(only_watertight=False,repair=False))
    ms=ml.MeshSet();ms.add_mesh(ml.Mesh(vertex_matrix=np.asarray(surface.vertices),face_matrix=np.asarray(surface.faces,dtype=np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    input_intersections=int(ms.current_mesh().face_selection_array().sum())
    if input_intersections:raise ValueError(f'输入表面存在 {input_intersections} 个相交面，拒绝自动重网格')
    ms.meshing_isotropic_explicit_remeshing(iterations=3,targetlen=ml.PureValue(target_mm),featuredeg=30,
        checksurfdist=True,maxsurfdist=ml.PureValue(tolerance/3),splitflag=False,smoothflag=smoothing)
    ms.compute_selection_by_self_intersections_per_face()
    intersections=int(ms.current_mesh().face_selection_array().sum())
    if intersections:raise ValueError(f'表面重网格产生 {intersections} 个相交面，拒绝使用')
    current=ms.current_mesh();result=trimesh.Trimesh(current.vertex_matrix(),current.face_matrix(),process=False)
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError('表面重网格改变闭合性或朝向，拒绝使用')
    if result.euler_number!=surface.euler_number or len(result.split(only_watertight=False,repair=False))!=before_parts:
        raise ValueError('表面重网格改变孔道或分量拓扑，拒绝使用')
    change=abs(result.volume-before_volume)/max(abs(before_volume),1e-20)
    if change>.001:raise ValueError('表面重网格体积变化超过 0.1%')
    distances=[]
    for src,dst in [(surface,result),(result,surface)]:
        rng=np.random.default_rng(20261107)
        ids=rng.choice(len(src.faces),8192,p=src.area_faces/src.area)
        uv=rng.random((len(ids),2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
        tri=src.triangles[ids];points=tri[:,0]+uv[:,:1]*(tri[:,1]-tri[:,0])+uv[:,1:]*(tri[:,2]-tri[:,0])
        _,distance,_=trimesh.proximity.closest_point(dst,points)
        distances.append(float(distance.max()))
    if max(distances)>tolerance*1.01:raise ValueError(f'表面重网格双向抽样距离超限：{distances}')
    return result,dict(method='constrained_sliver_conditioning',target_edge_mm=target_mm,tolerance_mm=tolerance,
        before_faces=len(surface.faces),after_faces=len(result.faces),volume_relative_change=float(change),
        bidirectional_sample_max_mm=distances,samples_per_direction=8192,euler_number=int(result.euler_number),
        topology_preserved=True,hausdorff_verified=False,smoothing=bool(smoothing),
        input_intersecting_faces=input_intersections,output_intersecting_faces=intersections)
