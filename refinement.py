"""Conforming eight-child tetrahedral refinement with a shared edge midpoint map."""
import numpy as np


def measures(nodes,cells):
    p=nodes[cells];v=np.abs(np.linalg.det(p[:,1:]-p[:,:1]))/6
    lengths=sum(np.sum((p[:,a]-p[:,b])**2,axis=-1) for a in range(4) for b in range(a))
    return v,12*(3*v)**(2/3)/lengths


def refine_tetra(nodes,cells,info,pitch,max_elements=600000):
    """Preserve the parent piecewise-linear boundary exactly; no surface smoothing."""
    if len(cells)*8>max_elements:raise ValueError(f'细化单元数量 {len(cells)*8} 超出限制 {max_elements}')
    pairs=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    edges=np.sort(cells[:,pairs].reshape(-1,2),axis=1)
    unique,inv=np.unique(edges,axis=0,return_inverse=True)
    n=np.vstack([nodes,nodes[unique].mean(axis=1)])
    mid=inv.reshape(-1,6)+len(nodes);del edges,unique,inv
    a,b,c,d=cells.T;ab,ac,ad,bc,bd,cd=mid.T
    corners=np.stack([np.stack(x,axis=1) for x in [(a,ab,ac,ad),(b,ab,bc,bd),(c,ac,bc,cd),(d,ad,bd,cd)]],axis=1)
    best=np.empty((len(cells),4,4),dtype=np.int32);best_q=np.full(len(cells),-1.)
    # The three octahedron diagonals affect only each parent's interior.
    for left,right,ring in [(ab,cd,[ac,ad,bd,bc]),(ac,bd,[ab,ad,cd,bc]),(ad,bc,[ab,ac,cd,bd])]:
        candidate=np.stack([np.stack([left,right,ring[i],ring[(i+1)%4]],axis=1) for i in range(4)],axis=1)
        _,q=measures(n,candidate.reshape(-1,4));q=q.reshape(-1,4).min(axis=1)
        better=q>best_q;best[better]=candidate[better];best_q[better]=q[better]
    refined=np.concatenate([corners,best],axis=1).reshape(-1,4).astype(np.int32)
    del corners,best,mid
    volumes,quality=measures(n,refined);parent_volume,_=measures(nodes,cells)
    if not np.isfinite(volumes).all() or np.any(volumes<=1e-16):raise ValueError('细化产生退化单元')
    if not np.allclose(volumes.reshape(-1,8).sum(axis=1),parent_volume,rtol=1e-9,atol=1e-14):raise ValueError('细化未保持父单元体积')
    if quality.min()<.001 or np.mean(quality<.01)>.001:raise ValueError('细化后的实测单元质量未达标')
    from robust_meshing import oriented_boundary
    import trimesh
    parent=trimesh.Trimesh(nodes,oriented_boundary(nodes,cells),process=False);parent.remove_unreferenced_vertices()
    child=trimesh.Trimesh(n,oriented_boundary(n,refined),process=False);child.remove_unreferenced_vertices()
    if not parent.is_watertight or not child.is_watertight or not child.is_winding_consistent or parent.euler_number!=child.euler_number:
        raise ValueError('细化边界拓扑校核失败')
    ordered=np.sort(volumes);p95=float(ordered[np.searchsorted(np.cumsum(ordered),.95*ordered.sum())])
    meta={**info,'refinement':dict(method='conforming_midpoint_8',parent_nodes=len(nodes),parent_elements=len(cells),
        boundary_preservation='exact_linear_subdivision',diagonal='maximize_minimum_child_mean_ratio',euler_number=int(child.euler_number)),
        'max_volume_target_mm3':pitch**3/6,'max_volume_actual_mm3':float(volumes.max()),'volume_p95_mm3':float(np.percentile(volumes,95)),
        'volume_weighted_p95_mm3':p95,'quality_min':float(quality.min()),'quality_p01':float(np.percentile(quality,1)),
        'low_quality_fraction':float(np.mean(quality<.01)),'quality_refinement':True,'boundary_refinement':True}
    meta['parent_backend']=info.get('backend','tetgen');meta['backend']='uniform_refinement'
    if 'backend_report' in info:
        meta['backend_report']={**info['backend_report'],'max_energy':float(3/quality.min()),
            'measurement':'independent_refined_3_over_mean_ratio',
            'optimization_target_met':bool(3/quality.min()<=info['backend_options']['stop_energy']*1.00001)}
    return n,refined,meta
