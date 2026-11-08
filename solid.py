"""General 3D voxel-to-tetrahedron elasticity. Units: mm, N, MPa."""
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from solver import map_angles


@dataclass
class Material:
    e1: float = 2400
    e2: float = 1500
    e3: float = 1000
    g12: float = 600
    g23: float = 400
    g13: float = 450
    nu12: float = .3
    nu23: float = .3
    nu13: float = .3


def local_stiffness(m):
    v=np.array(list(vars(m).values()))
    if not np.isfinite(v).all() or min(v[:6])<=0:
        raise ValueError('材料参数必须有限，弹性模量与剪切模量必须为正')
    S=np.diag([1/m.e1,1/m.e2,1/m.e3,1/m.g12,1/m.g23,1/m.g13])
    S[0,1]=S[1,0]=-m.nu12/m.e1
    S[1,2]=S[2,1]=-m.nu23/m.e2
    S[0,2]=S[2,0]=-m.nu13/m.e1
    if np.linalg.eigvalsh(S).min()<=0: raise ValueError('材料柔度非正定，请检查泊松比')
    return np.linalg.inv(S)


def transform(angle, build_axis='Z'):
    # Columns: path direction, transverse direction, build normal.
    frames={'Z':np.eye(3),'X':np.eye(3)[:,[1,2,0]],'Y':np.eye(3)[:,[2,0,1]]}
    c,s=np.cos(angle),np.sin(angle)
    R=frames[build_axis]@np.array([[c,-s,0],[s,c,0],[0,0,1]])
    T=np.empty((6,6))
    for j in range(6):
        E=np.zeros((3,3))
        if j<3: E[j,j]=1
        else:
            a,b=[(0,1),(1,2),(0,2)][j-3]; E[a,b]=E[b,a]=.5
        L=R.T@E@R
        T[:,j]=[L[0,0],L[1,1],L[2,2],2*L[0,1],2*L[1,2],2*L[0,2]]
    return T


def voxel_mesh(indices, pitch, origin):
    indices=np.asarray(indices,dtype=int)
    if not len(indices): raise ValueError('网格为空；请减小网格尺寸')
    if len(indices)>16000: raise ValueError('超过 16000 个体素，请增大网格尺寸')
    corners=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    vertices=(indices[:,None,:]+corners).reshape(-1,3)
    unique,inverse=np.unique(vertices,axis=0,return_inverse=True)
    cubes=inverse.reshape(-1,8)
    pattern=np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]])
    cells=cubes[:,pattern].reshape(-1,4)
    return unique*pitch+origin,cells


def box_mesh(size=(40,12,8), pitch=2):
    size=np.array(size,float)
    if not np.isfinite(size).all() or np.any(size<=0) or not np.isfinite(pitch) or pitch<=0:
        raise ValueError('尺寸和网格间距须为正有限值')
    count=np.ceil(size/pitch).astype(int)
    if np.prod(count)>16000: raise ValueError('网格过密，请增大间距')
    indices=np.indices(count).reshape(3,-1).T
    return voxel_mesh(indices,size/count,np.zeros(3))


def stl_mesh(path,pitch):
    import trimesh
    if not np.isfinite(pitch) or pitch<=0: raise ValueError('网格间距须为正有限值')
    surface=trimesh.load_mesh(path)
    if not isinstance(surface,trimesh.Trimesh) or not surface.is_watertight:
        raise ValueError('STL 必须为闭合三角网格，请先修复孔洞')
    if not np.isfinite(surface.vertices).all(): raise ValueError('STL 顶点含无效数值')
    origin=surface.bounds[0]
    count=np.ceil(surface.extents/pitch).astype(int)
    if np.any(count<1) or np.prod(count)>150000:
        raise ValueError('包围盒采样数过大或尺寸无效；请增大网格间距')
    ids=np.indices(count).reshape(3,-1).T
    # Fit cells to the STL bounding box, as box_mesh does. A scalar pitch can
    # otherwise truncate or extend a thin part by an entire partial layer.
    cell_pitch=surface.extents/count
    inside=[]
    for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/2000)))):
        inside.append(surface.contains(origin+(chunk+.5)*cell_pitch))
    occupied=ids[np.concatenate(inside)]
    return voxel_mesh(occupied,cell_pitch,origin)


def boundary_faces(cells):
    faces=cells[:,[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]].reshape(-1,3)
    _,first,count=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    idx=first[count==1]
    return faces[idx],idx//4


def select_faces(nodes,faces,region):
    centers=nodes[faces].mean(axis=1)
    if isinstance(region,str):
        axis='XYZ'.index(region[0]); edge=nodes[:,axis].min() if region[1]=='-' else nodes[:,axis].max()
        return np.all(np.isclose(nodes[faces,axis],edge,atol=1e-7,rtol=0),axis=1)
    region=np.asarray(region,float)
    if region.shape!=(2,3) or not np.isfinite(region).all() or np.any(region[1]<region[0]):
        raise ValueError('选区格式：xmin,ymin,zmin,xmax,ymax,zmax；最大值不能小于最小值')
    return np.all((centers>=region[0])&(centers<=region[1]),axis=1)


def path_field(centers,segments,offset):
    # Preserve absolute machine coordinates. User gives explicit XYZ translation.
    p=np.array(segments,dtype=float,copy=True); offset=np.asarray(offset,float)
    p[:,:2]+=offset[:2]; p[:,2:4]+=offset[:2]; p[:,4]+=offset[2]
    layers=np.unique(np.round(p[:,4],5))
    layer_id=np.abs(centers[:,2,None]-layers[None,:]).argmin(axis=1) if len(layers)<100 else None
    if layer_id is None:
        from scipy.spatial import cKDTree
        layer_id=cKDTree(layers[:,None]).query(centers[:,2,None])[1]
    angles=np.empty(len(centers)); distance=np.empty(len(centers))
    for i in np.unique(layer_id):
        mask=layer_id==i; selected=p[np.isclose(p[:,4],layers[i],atol=1e-5,rtol=0)]
        if len(selected)>50000 or len(selected)*mask.sum()>2000000:
            from numerics import nearest_segments
            angles[mask],d=nearest_segments(centers[mask,:2],selected)
        else:angles[mask],d=map_angles(centers[mask,:2],selected)
        distance[mask]=np.hypot(d,centers[mask,2]-layers[i])
    return angles,distance


def surface_load(nodes, faces, force, load_point=None):
    """Area weighted force, with an equivalent couple for a remote force point (mm)."""
    force=np.asarray(force,float)
    if force.shape!=(3,) or not np.isfinite(force).all():raise ValueError('force must be a finite 3-vector')
    tri=nodes[faces]
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if areas.sum()<=0:raise ValueError('Load surface has zero area')
    weights=np.zeros(len(nodes))
    for j in range(3):np.add.at(weights,faces[:,j],areas/(3*areas.sum()))
    centroid=weights@nodes
    f=weights[:,None]*force
    if load_point is not None:
        point=np.asarray(load_point,float)
        if point.shape!=(3,) or not np.isfinite(point).all():raise ValueError('load_point must be a finite 3-vector')
        r=nodes-centroid
        inertia=np.eye(3)*np.sum(weights*np.sum(r*r,axis=1))-r.T@(weights[:,None]*r)
        moment=np.cross(point-centroid,force)
        if np.linalg.matrix_rank(inertia)<3:raise ValueError('Load surface cannot transmit a general moment')
        omega=np.linalg.solve(inertia,moment)
        f+=weights[:,None]*np.cross(omega,r)
    return f


def solve_solid(nodes,cells,material=None,fixed='X-',loaded='X+',force=(0,0,-10),
                angle=0,build_axis='Z',segments=None,offset=(0,0,0),progress=None,
                solver='auto',rtol=1e-9,maxiter=2000,chunk_size=4096,load_point=None):
    progress=progress or (lambda stage: None)
    progress('boundary_validation')
    material=material or Material(); D=local_stiffness(material)
    force=np.asarray(force,float)
    if not np.isfinite(force).all() or not np.isfinite(angle) or not np.isfinite(offset).all():
        raise ValueError('载荷、角度、偏移必须有限')
    faces,owners=boundary_faces(cells)
    fixed_mask=select_faces(nodes,faces,fixed); load_mask=select_faces(nodes,faces,loaded)
    if not fixed_mask.any() or not load_mask.any(): raise ValueError('固定或加载选区没有选中表面三角形')
    fixed_nodes=np.unique(faces[fixed_mask]); loaded_nodes=np.unique(faces[load_mask])
    if np.intersect1d(fixed_nodes,loaded_nodes).size: raise ValueError('固定区域与加载区域共享节点，请分开选区')
    # Require a face-connected body: edge/point-only voxel contacts are not valid joints.
    all_faces=np.sort(cells[:,[[0,1,2],[0,1,3],[0,2,3],[1,2,3]]].reshape(-1,3),axis=1)
    _,inv,counts=np.unique(all_faces,axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inv); starts=np.r_[0,np.cumsum(counts)[:-1]]; shared=starts[counts==2]
    a=order[shared]//4; b=order[shared+1]//4
    graph=coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(len(cells),len(cells))).tocsr()
    if connected_components(graph,directed=False,return_labels=False)!=1:
        raise ValueError('网格有多个不连通实体或仅边角接触；请逐件分析或减小网格间距')
    centers=nodes[cells].mean(axis=1)
    angles=np.full(len(cells),np.deg2rad(angle)); distances=np.zeros(len(cells))
    if segments is not None:
        progress('path_mapping')
        if build_axis!='Z': raise ValueError('G-code 模式使用世界 Z 为层法线；请在切片前确定打印姿态')
        angles,distances=path_field(centers,segments,offset)
    progress('assembly')
    from numerics import assemble,solve_linear,recover
    K,volume,assembly_seconds=assemble(nodes,cells,angles,build_axis,D,chunk_size,progress)
    f=surface_load(nodes,faces[load_mask],force,load_point)
    fixed_dofs=(fixed_nodes[:,None]*3+np.arange(3)).ravel()
    free=np.setdiff1d(np.arange(nodes.size),fixed_dofs); u=np.zeros(nodes.size)
    progress('linear_solve')
    u[free],solver_info=solve_linear(K,f.ravel(),nodes,free,solver,rtol,maxiter,progress)
    solver_info['assembly_seconds']=assembly_seconds
    residual=K@u-f.ravel()
    relative=np.linalg.norm(residual[free])/max(np.linalg.norm(f),1e-20)
    if not np.isfinite(u).all() or relative>1e-6: raise ValueError('求解未通过残差检查；请检查约束及网格')
    progress('postprocess')
    local,stress=recover(nodes,cells,angles,build_axis,D,u,chunk_size)
    sx,sy,sz,txy,tyz,txz=stress.T
    vm=np.sqrt(.5*((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)+3*(txy*txy+tyz*tyz+txz*txz))
    return dict(nodes=nodes,cells=cells,faces=faces,owners=owners,displacement=u.reshape(-1,3),
                stress=stress,local_stress=local,von_mises=vm,angles=np.rad2deg(angles),path_distance=distances,
                fixed_nodes=fixed_nodes,loaded_nodes=loaded_nodes,forces=f,reactions=residual.reshape(-1,3)[fixed_nodes],
                relative_residual=relative,energy=float(.5*u@f.ravel()),volume=volume,solver_info=solver_info)


def write_vtk(path,result):
    r=result
    with open(path,'w',encoding='ascii') as f:
        f.write('# vtk DataFile Version 3.0\nPathFEA solid\nASCII\nDATASET UNSTRUCTURED_GRID\n')
        f.write(f"POINTS {len(r['nodes'])} double\n")
        np.savetxt(f,r['nodes'])
        f.write(f"CELLS {len(r['cells'])} {len(r['cells'])*5}\n")
        np.savetxt(f,np.column_stack([np.full(len(r['cells']),4),r['cells']]),fmt='%d')
        f.write(f"CELL_TYPES {len(r['cells'])}\n"); np.savetxt(f,np.full(len(r['cells']),10),fmt='%d')
        f.write(f"POINT_DATA {len(r['nodes'])}\nVECTORS displacement double\n"); np.savetxt(f,r['displacement'])
        f.write(f"CELL_DATA {len(r['cells'])}\n")
        for key,data in [('von_mises',r['von_mises']),('path_angle',r['angles']),('path_distance',r['path_distance']),('sigma_layer',r['local_stress'][:,2])]:
            f.write(f'SCALARS {key} double 1\nLOOKUP_TABLE default\n'); np.savetxt(f,data)
