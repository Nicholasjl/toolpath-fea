"""Memory-bounded tetrahedron assembly and elasticity-aware iterative solution."""
import time
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import cg,spsolve,MatrixRankWarning
import warnings


def strain_transform(angles,axis='Z'):
    from solid import transform
    angles=np.asarray(angles);c=np.cos(angles);s=np.sin(angles)
    T=np.zeros((len(angles),6,6))
    T[:,0,0]=T[:,1,1]=c*c;T[:,0,1]=T[:,1,0]=s*s
    T[:,0,3]=c*s;T[:,1,3]=-c*s;T[:,2,2]=1
    T[:,3,0]=-2*c*s;T[:,3,1]=2*c*s;T[:,3,3]=c*c-s*s
    T[:,4,4]=c;T[:,4,5]=-s;T[:,5,4]=s;T[:,5,5]=c
    return T if axis=='Z' else T@transform(0,axis)


def element_data(nodes,cells,angles,axis):
    xyz=nodes[cells];edges=xyz[:,1:]-xyz[:,:1]
    det=np.linalg.det(edges);volumes=np.abs(det)/6
    if not np.isfinite(volumes).all() or np.any(volumes<=1e-16):raise ValueError('存在退化体单元')
    grads=np.empty((len(cells),3,4));grads[:,:,1:]=np.linalg.inv(edges);grads[:,:,0]=-grads[:,:,1:].sum(axis=2)
    B=np.zeros((len(cells),6,12))
    B[:,0,0::3]=grads[:,0];B[:,1,1::3]=grads[:,1];B[:,2,2::3]=grads[:,2]
    B[:,3,0::3]=grads[:,1];B[:,3,1::3]=grads[:,0]
    B[:,4,1::3]=grads[:,2];B[:,4,2::3]=grads[:,1]
    B[:,5,0::3]=grads[:,2];B[:,5,2::3]=grads[:,0]
    T=strain_transform(angles,axis)
    dofs=(cells[:,:,None]*3+np.arange(3)).reshape(-1,12)
    return T@B,T,volumes,dofs


def assemble(nodes,cells,angles,axis,D,chunk_size=4096,progress=None):
    """Balanced CSR reduction avoids retaining global element matrices or COO arrays."""
    levels=[];total_volume=0.;started=time.perf_counter()
    for start in range(0,len(cells),chunk_size):
        stop=start+chunk_size
        TB,_,v,dofs=element_data(nodes,cells[start:stop],angles[start:stop],axis)
        ke=(TB.transpose(0,2,1)@D@TB)*v[:,None,None]
        total_volume+=v.sum()
        block=coo_matrix((ke.ravel(),(np.repeat(dofs,12,axis=1).ravel(),np.tile(dofs,(1,12)).ravel())),shape=(nodes.size,nodes.size)).tocsr()
        level=0
        while level<len(levels) and levels[level] is not None:
            block=levels[level]+block;levels[level]=None;level+=1
        if level==len(levels):levels.append(block)
        else:levels[level]=block
        if progress and start%(chunk_size*16)==0:progress(f'assembly_{min(stop,len(cells))}_of_{len(cells)}')
    K=None
    for block in levels:
        if block is not None:K=block if K is None else K+block
    K.eliminate_zeros()
    return K,float(total_volume),time.perf_counter()-started


def rigid_modes(nodes):
    p=nodes-nodes.mean(axis=0);p/=max(np.ptp(p,axis=0).max(),1e-15)
    R=np.zeros((len(nodes),3,6));R[:,:,:3]=np.eye(3)
    for axis in range(3):R[:,:,axis+3]=np.cross(np.eye(3)[axis],p)
    return R.reshape(-1,6)


def solve_linear(K,f,nodes,free,method='auto',rtol=1e-9,maxiter=2000,progress=None):
    if method not in ('auto','direct','amg'):raise ValueError('solver 必须为 auto/direct/amg')
    if not 0<rtol<=1e-4 or maxiter<1:raise ValueError('迭代容差或次数无效')
    started=time.perf_counter();A=K[free][:,free].tocsr();rhs=f[free]
    if np.any(A.diagonal()<=0):raise ValueError('自由自由度存在非正刚度，请检查网格或约束')
    method=('direct' if len(free)<30000 else 'amg') if method=='auto' else method
    stats={'method':method,'iterations':0,'free_dofs':len(free),'matrix_nnz':int(A.nnz)}
    if np.linalg.norm(rhs)==0:return np.zeros(len(free)),{**stats,'seconds':time.perf_counter()-started,'relative_residual':0.}
    # Equilibrate both direct and iterative solves to improve thin-section conditioning.
    scale=1/np.sqrt(A.diagonal());S=diags(scale);scaled=(S@A@S).tocsr();b=scale*rhs
    if method=='direct':
        with warnings.catch_warnings():
            warnings.simplefilter('error',MatrixRankWarning);y=spsolve(scaled,b)
    else:
        import pyamg
        candidates=rigid_modes(nodes)[free]/scale[:,None]
        if progress:progress('amg_setup')
        # All boundary conditions clamp complete nodes, preserving three-DOF blocks.
        block=scaled.tobsr(blocksize=(3,3))
        hierarchy=pyamg.smoothed_aggregation_solver(block,B=candidates,symmetry='symmetric',
                    smooth='energy',max_coarse=120)
        stats['amg_levels']=len(hierarchy.levels);stats['operator_complexity']=float(hierarchy.operator_complexity())
        def callback(_):
            stats['iterations']+=1
            if progress and stats['iterations']%25==0:progress(f'cg_iteration_{stats["iterations"]}')
        y,info=cg(scaled,b,M=hierarchy.aspreconditioner(),rtol=rtol,atol=0,maxiter=maxiter,callback=callback)
        if info!=0:raise ValueError(f'AMG/CG 未收敛，info={info}，iterations={stats["iterations"]}；未自动转入高内存直接求解')
    x=scale*y;residual=float(np.linalg.norm(A@x-rhs)/np.linalg.norm(rhs))
    if not np.isfinite(x).all() or residual>1e-6:raise ValueError(f'原始系统残差检查失败：{residual:.3g}')
    stats.update(seconds=time.perf_counter()-started,relative_residual=residual)
    return x,stats


def recover(nodes,cells,angles,axis,D,u,chunk_size=4096):
    local=np.empty((len(cells),6));stress=np.empty_like(local)
    for start in range(0,len(cells),chunk_size):
        stop=start+chunk_size;TB,T,_,dofs=element_data(nodes,cells[start:stop],angles[start:stop],axis)
        local[start:stop]=np.einsum('eij,ej->ei',TB,u[dofs])@D.T
        stress[start:stop]=np.einsum('eji,ej->ei',T,local[start:stop])
    return local,stress


def nearest_segments(centers,segments,max_length=1.):
    """Exact nearest-line distance using short subsegments and a KD-tree bound."""
    from scipy.spatial import cKDTree
    a=segments[:,:2];d=segments[:,2:4]-a;length=np.linalg.norm(d,axis=1)
    if np.any(length<=0):raise ValueError('路径含零长度线段')
    counts=np.maximum(1,np.ceil(length/max_length).astype(int))
    if counts.sum()>3000000:raise ValueError('路径空间索引超过 300 万子段，请裁剪路径')
    owner=np.repeat(np.arange(len(segments)),counts)
    starts=np.repeat(np.cumsum(counts)-counts,counts)
    fraction=(np.arange(len(owner))-starts)/counts[owner]
    aa=a[owner]+fraction[:,None]*d[owner];dd=d[owner]/counts[owner,None]
    mid=aa+.5*dd;half=np.linalg.norm(dd,axis=1).max()*.5;tree=cKDTree(mid)
    angles=np.empty(len(centers));distances=np.empty(len(centers))
    for start in range(0,len(centers),2048):
        p=centers[start:start+2048];nearest=tree.query(p)[1]
        diff=p-aa[nearest];t=np.clip(np.sum(diff*dd[nearest],axis=1)/np.sum(dd[nearest]**2,axis=1),0,1)
        upper=np.linalg.norm(diff-t[:,None]*dd[nearest],axis=1)
        candidates=tree.query_ball_point(p,upper+half+1e-10)
        sizes=np.array([len(c) for c in candidates]);idx=np.concatenate(candidates).astype(int);point=np.repeat(np.arange(len(p)),sizes)
        diff=p[point]-aa[idx];t=np.clip(np.sum(diff*dd[idx],axis=1)/np.sum(dd[idx]**2,axis=1),0,1)
        squared=np.sum((diff-t[:,None]*dd[idx])**2,axis=1)
        minimum=np.full(len(p),np.inf);np.minimum.at(minimum,point,squared)
        # Original segment index is a stable tie break (same as brute-force argmin).
        chosen=np.full(len(p),len(segments),dtype=int);tied=np.abs(squared-minimum[point])<1e-18
        np.minimum.at(chosen,point[tied],owner[idx[tied]])
        angles[start:start+len(p)]=np.arctan2(d[chosen,1],d[chosen,0]);distances[start:start+len(p)]=np.sqrt(minimum)
    return angles,distances
