"""Spawned numerical worker. No GUI imports; never write to stdout."""
import hashlib
import html
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from solid import Material,local_stiffness,box_mesh,stl_mesh,boundary_faces,select_faces,solve_solid,write_vtk
from solver import read_gcode
from defaults import DEFAULT_MESH_MODE, DEFAULT_PITCH_MM, REFINED_VOLUME_RATIO
from meshing import volume_resolution


class JobError(Exception):
    def __init__(self,code,message):
        super().__init__(message);self.code=code


def validate_case(cfg,base):
    if not isinstance(cfg,dict):raise JobError('INVALID_CASE','工况必须为 JSON 对象')
    allowed={'stl','size','pitch','gcode','offset','angle','build_axis','material','fixed','loaded','force',
             'mesh_mode','snap_tolerance','solver','rtol','maxiter','chunk_size','load_point','surface_remesh_mm'}
    if set(cfg)-allowed:raise JobError('INVALID_CASE',f'未知工况字段：{sorted(set(cfg)-allowed)}')
    cfg={'stl':'','size':[40,12,8],'pitch':DEFAULT_PITCH_MM,'gcode':'','offset':[0,0,0],'angle':0,
         'build_axis':'Z','material':{},'fixed':'X-','loaded':'X+','force':[0,0,-10],
         'mesh_mode':DEFAULT_MESH_MODE,'snap_tolerance':.0001,'surface_remesh_mm':0,'solver':'auto','rtol':1e-9,'maxiter':2000,'chunk_size':4096,**cfg}
    def number(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    for key in ('pitch','angle'):
        if not number(cfg[key]):raise JobError('INVALID_CASE',f'{key} 必须为有限数值')
    if cfg['pitch']<=0:raise JobError('INVALID_CASE','pitch 必须大于零')
    if cfg['mesh_mode'] not in ('voxel','tetra'):raise JobError('INVALID_CASE','mesh_mode 必须为 voxel/tetra')
    if cfg['solver'] not in ('auto','direct','amg'):raise JobError('INVALID_CASE','solver 必须为 auto/direct/amg')
    if not number(cfg['snap_tolerance']) or not 0<=cfg['snap_tolerance']<=.01:raise JobError('INVALID_CASE','snap_tolerance 范围为 0–0.01 mm')
    if not number(cfg['surface_remesh_mm']) or not 0<=cfg['surface_remesh_mm']<=5:raise JobError('INVALID_CASE','surface_remesh_mm 范围为 0–5 mm')
    if cfg['surface_remesh_mm'] and (cfg['mesh_mode']!='tetra' or not cfg['stl']):raise JobError('INVALID_CASE','表面重网格需要 STL 与 tetra 模式')
    if not number(cfg['rtol']) or not 0<cfg['rtol']<=1e-4:raise JobError('INVALID_CASE','rtol 范围为 (0,1e-4]')
    for key,lo,hi in [('maxiter',1,10000),('chunk_size',128,16384)]:
        if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or not lo<=cfg[key]<=hi:raise JobError('INVALID_CASE',f'{key} 范围 {lo}–{hi}')
    for key in ('size','offset','force'):
        if not isinstance(cfg[key],list) or len(cfg[key])!=3 or not all(map(number,cfg[key])):
            raise JobError('INVALID_CASE',f'{key} 必须为三个有限数值')
    if 'load_point' in cfg and (not isinstance(cfg['load_point'],list) or len(cfg['load_point'])!=3 or not all(map(number,cfg['load_point']))):
        raise JobError('INVALID_CASE','load_point 必须为三个有限数值，单位 mm')
    if min(cfg['size'])<=0:raise JobError('INVALID_CASE','size 必须大于零')
    if cfg['build_axis'] not in ('X','Y','Z'):raise JobError('INVALID_CASE','build_axis 必须为 X/Y/Z')
    if not isinstance(cfg['material'],dict) or not all(map(number,cfg['material'].values())):
        raise JobError('INVALID_CASE','material 必须为数值参数对象')
    try:local_stiffness(Material(**cfg['material']))
    except (ValueError,TypeError) as exc:raise JobError('INVALID_MATERIAL',str(exc)) from exc
    for key in ('fixed','loaded'):
        value=cfg[key]
        if isinstance(value,str):
            if value not in ('X-','X+','Y-','Y+','Z-','Z+'):raise JobError('INVALID_CASE',f'{key} 面名称无效')
        else:
            try:
                if not isinstance(value,list) or len(value)!=2 or any(not isinstance(row,list) or len(row)!=3 or not all(map(number,row)) for row in value):raise ValueError()
                if np.any(np.array(value[0])>value[1]):raise ValueError()
            except (ValueError,TypeError):raise JobError('INVALID_CASE',f'{key} 需为 [[xmin,ymin,zmin],[xmax,ymax,zmax]]')
    for key in ('stl','gcode'):
        if not isinstance(cfg[key],str):raise JobError('INVALID_CASE',f'{key} 必须为文件路径字符串')
        if cfg[key]:
            path=Path(cfg[key]);path=path if path.is_absolute() else Path(base)/path
            if not path.is_file():raise JobError('INPUT_NOT_FOUND',f'{key}: {path}')
            cfg[key]=str(path.resolve())
    if cfg['gcode'] and cfg['build_axis']!='Z':raise JobError('INVALID_CASE','G-code 模式要求 build_axis=Z')
    return cfg


def check(code,status,message,**metrics):
    return dict(code=code,status=status,message=message,metrics=metrics)


def diagnostics(cfg,r,segments):
    checks=[];nodes=r['nodes'];cells=r['cells'];pitch=cfg['pitch']
    reaction=r['reactions'].sum(axis=0);force=r['forces'].sum(axis=0)
    rel_force=float(np.linalg.norm(reaction+force)/max(np.linalg.norm(force),1))
    moment=np.cross(nodes-nodes.mean(axis=0),r['forces']).sum(axis=0)+np.cross(nodes[r['fixed_nodes']]-nodes.mean(axis=0),r['reactions']).sum(axis=0)
    rel_moment=float(np.linalg.norm(moment)/max(np.linalg.norm(force)*np.ptp(nodes,axis=0).max(),1))
    checks.append(check('EQUILIBRIUM','pass' if max(rel_force,rel_moment)<1e-6 else 'warn','力及力矩平衡',force_relative_error=rel_force,moment_relative_error=rel_moment,residual=r['relative_residual']))
    if np.linalg.norm(force)==0:checks.append(check('ZERO_LOAD','warn','载荷为零，无法评估网格收敛或实际受力响应'))
    faces=r['faces']
    for key in ('fixed','loaded'):
        selected=faces[select_faces(nodes,faces,cfg[key])]
        tri=nodes[selected];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        checks.append(check('REGION_'+key.upper(),'warn' if len(selected)<4 else 'pass',
                            '区域按表面中心选择；面积仅用于核对，不能自动判断是否符合真实安装条件',faces=len(selected),area_mm2=float(areas.sum()),bounds=[tri.min(axis=(0,1)).tolist(),tri.max(axis=(0,1)).tolist()]))
    extent=np.ptp(nodes,axis=0)
    checks.append(check('MESH_RESOLUTION','warn' if extent.min()/pitch<4 else 'pass','包围盒最短方向网格数检查；不替代局部薄壁检测',min_extent_over_pitch=float(extent.min()/pitch)))
    if cfg['stl'] and r.get('mesh_info',{}).get('mode')=='tetra':
        info=r['mesh_info']
        checks.append(check('GEOMETRY_APPROXIMATION','pass' if info['volume_relative_error']<.002 else 'warn','贴体四面体保留清理后的表面；数值清理容差与体积偏差已记录',**info))
        checks.append(check('ELEMENT_QUALITY','warn' if info['low_quality_fraction']>.001 or info['quality_min']<1e-6 else 'pass','几何细节附近仍可能产生扁薄单元；需结合原始系统残差与收敛检查',quality_min=info['quality_min'],low_quality_fraction=info['low_quality_fraction']))
        sizing=volume_resolution(info)
        checks.append(check('MESH_VOLUME_SIZE',sizing.pop('status'),
                            '实际单元体积及按材料体积加权的p95与目标比较；表面节点数不能替代体积细化',**sizing))
    elif cfg['stl']:
        import trimesh
        surface=trimesh.load_mesh(cfg['stl']);source_volume=abs(float(surface.volume))
        error=abs(r['volume']-source_volume)/max(source_volume,1e-20)
        # Sample actual triangle surfaces; ask whether an occupied voxel covers each point.
        rng=np.random.default_rng(2026);ids=rng.choice(len(surface.faces),size=4096,p=surface.area_faces/surface.area)
        triangles=surface.triangles[ids];uv=rng.random((4096,2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
        points=triangles[:,0]+uv[:,0,None]*(triangles[:,1]-triangles[:,0])+uv[:,1,None]*(triangles[:,2]-triangles[:,0])
        occupied=nodes[cells.reshape(-1,6,4)].mean(axis=(1,2))
        distance=cKDTree(occupied).query(points,p=np.inf)[0]
        missing=float(np.mean(distance>pitch*.51))
        checks.append(check('GEOMETRY_APPROXIMATION','warn' if error>.05 or missing>.05 else 'pass',
                            '体积误差与确定性表面抽样筛查；可能漏掉小于采样/网格尺度的薄壁和孔洞',source_volume_mm3=source_volume,volume_relative_error=error,uncovered_surface_fraction=missing,samples=4096))
    displacement=float(np.linalg.norm(r['displacement'],axis=1).max())
    checks.append(check('SMALL_DISPLACEMENT','warn' if displacement/extent.max()>.05 else 'pass',
                        '最大位移/包围盒长度筛查，不替代局部应变和几何非线性判定',ratio=displacement/float(extent.max())))
    if segments is not None:
        d=r['path_distance'];threshold=max(pitch,0.1)
        checks.append(check('PATH_ALIGNMENT','warn' if np.mean(d>threshold)>.05 else 'pass',
                            '最近层/最近路径覆盖检查，不证明实际零件及路径完全一致',distance_p95_mm=float(np.percentile(d,95)),distance_max_mm=float(d.max()),fraction_over_pitch=float(np.mean(d>threshold))))
        layers=np.unique(segments[:,4]);spacing=np.diff(layers)
        if len(spacing):
            dz=float(np.median(spacing))
            checks.append(check('LAYER_RESOLUTION','warn' if pitch>dz else 'pass','体单元跨层时最近层映射可能欠采样交替方向',pitch_mm=pitch,median_layer_height_mm=dz))
    return checks


def convergence_check(base,fine,du,ds,force):
    # A new pitch may leave a boundary-constrained tetra mesh exactly unchanged.
    unchanged=(np.array_equal(base['nodes'],fine['nodes']) and np.array_equal(base['cells'],fine['cells']))
    status='not_checked' if unchanged else 'pass' if du<=.1 and ds<=.15 and np.linalg.norm(force)>0 else 'warn'
    message='网格未实际变化，不能据此认定收敛' if unchanged else '两级网格最大位移及全局应力 p95 比较；不构成完整收敛证明'
    refinement={}
    bi=base.get('mesh_info',{});fi=fine.get('mesh_info',{})
    if bi.get('mode')=='tetra' or fi.get('mode')=='tetra':
        bv=bi.get('volume_weighted_p95_mm3');fv=fi.get('volume_weighted_p95_mm3')
        ratio=float(fv/bv) if bv and fv is not None else None
        sizing_ok=all(volume_resolution(info)['status']=='pass' for info in (bi,fi))
        enabled=all(info.get('quality_refinement') and info.get('boundary_refinement') for info in (bi,fi))
        refinement=dict(volume_weighted_p95_refinement_ratio=ratio,refinement_ratio_limit=REFINED_VOLUME_RATIO,
                        sizing_checks_passed=sizing_ok,quality_and_boundary_refinement_enabled=bool(enabled))
        if not sizing_ok or not enabled or ratio is None or ratio>REFINED_VOLUME_RATIO:
            status='not_checked';message='未证明实际体积网格充分细化，位移接近不能认定收敛'
    return check('MESH_CONVERGENCE',status,message,mesh_unchanged=unchanged,
                 displacement_relative_change=du,stress_p95_relative_change=ds,displacement_tolerance=.1,stress_tolerance=.15,**refinement)


def metrics(r):
    return dict(nodes=len(r['nodes']),elements=len(r['cells']),
                max_displacement_mm=float(np.linalg.norm(r['displacement'],axis=1).max()),
                stress_p95_MPa=float(np.percentile(r['von_mises'],95)),max_von_mises_MPa=float(r['von_mises'].max()),
                reaction_N=r['reactions'].sum(axis=0).tolist(),relative_residual=float(r['relative_residual']),volume_mm3=r['volume'],solver=r.get('solver_info',{}),mesh=r.get('mesh_info',{}))


def write_npz(path,result):
    # Metadata stored as JSON Unicode arrays; loading never requires allow_pickle=True.
    np.savez_compressed(path,**{k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in result.items()})


def render(r,path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    fig=Figure(figsize=(15,6),dpi=130)
    fig.subplots_adjust(left=.02,right=.96,wspace=.38)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    p=r['nodes'];faces=r['faces'];owners=r['owners']
    values=[np.linalg.norm(r['displacement'],axis=1)[faces].mean(axis=1),r['local_stress'][owners,2]]
    for i,(data,title) in enumerate(zip(values,['Displacement / mm (undeformed)','Layer normal stress / MPa'])):
        ax=fig.add_subplot(1,2,i+1,projection='3d');norm=Normalize(float(data.min()),float(data.max())+1e-15)
        ax.add_collection3d(Poly3DCollection(p[faces],facecolors=matplotlib.colormaps['viridis'](norm(data)),linewidths=0))
        ax.auto_scale_xyz(p[:,0],p[:,1],p[:,2]);ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),1e-6))
        ax.set(xlabel='X / mm',ylabel='Y / mm',zlabel='Z / mm',title=title)
        ax.tick_params(labelsize=8)
        fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=ax,shrink=.55,pad=.2,fraction=.035)
    fig.savefig(path,bbox_inches='tight')


def write_report(path,summary):
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [x['name'],x['pitch'],x['angle'],x['metrics']['max_displacement_mm'],x['metrics']['stress_p95_MPa']])+'</tr>' for x in summary['runs'])
    checks=''.join(f"<li><b>{html.escape(c['code'])} [{c['status']}]</b> {html.escape(c['message'])}<pre>{html.escape(json.dumps(c['metrics'],ensure_ascii=False,indent=2))}</pre></li>" for c in summary['checks'])
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PathFEA analysis report</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#213547}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border:1px solid #ccc}}img{{width:100%}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:12px}}</style>
<h1>PathFEA 三维分析报告</h1><p>数值筛查状态：{summary['quality']}。实心等效、线弹性模型；材料仍需标定。本报告不证明承载安全，不预测断裂。</p>
<img src="preview.png" alt="位移与层间正应力"><h2>工况对比</h2><p>位移单位 mm，应力单位 MPa。应力为全体单元 p95；不替代目标区域应力收敛。</p>
<table><tr><th>工况</th><th>网格 mm</th><th>统一角度 °</th><th>最大位移</th><th>等效应力 p95</th></tr>{rows}</table>
<h2>自动校核（基础工况）</h2><p>角度对比工况仅报告求解指标，未逐一进行网格收敛检查。</p><ul>{checks}</ul><h2>输入工况</h2><pre>{html.escape(json.dumps(summary['settings'],ensure_ascii=False,indent=2))}</pre></html>'''
    Path(path).write_text(content,encoding='utf-8')


def worker_main(queue,cfg,base,directory,options):
    stage='validation'
    def emit(value):
        nonlocal stage
        stage=value;queue.put({'type':'progress','stage':value,'time':time.time()})
    try:
        emit('validation');cfg=validate_case(cfg,base);out=Path(directory)
        # Hash inputs before computation for reproducibility. Files must remain unchanged during a run.
        inputs={}
        for key in ('stl','gcode'):
            if cfg[key]:
                digest=hashlib.sha256()
                with open(cfg[key],'rb') as source:
                    for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
                inputs[key]={'path':cfg[key],'sha256':digest.hexdigest()}
        emit('gcode_read');segments=read_gcode(cfg['gcode']) if cfg['gcode'] else None
        mesh_cache={}
        def compute(case,label):
            emit(label+'.mesh')
            key=(case['stl'],case['pitch'],case['mesh_mode'],case['snap_tolerance'],case['surface_remesh_mm'])
            if key not in mesh_cache:
                if case['stl'] and case['mesh_mode']=='tetra':
                    from meshing import tetra_mesh
                    mesh_cache[key]=tetra_mesh(case['stl'],case['pitch'],case['snap_tolerance'],surface_remesh_mm=case['surface_remesh_mm'])
                else:
                    n,c=stl_mesh(case['stl'],case['pitch']) if case['stl'] else box_mesh(case['size'],case['pitch'])
                    mesh_cache[key]=(n,c,{'mode':'voxel'})
            n,c,info=mesh_cache[key]
            numerical={k:case[k] for k in ('fixed','loaded','force','angle','build_axis','offset','solver','rtol','maxiter','chunk_size')}
            if 'load_point' in case:numerical['load_point']=case['load_point']
            if numerical['solver']=='auto' and info.get('mode')=='tetra' and info.get('quality_min',1)<1e-5:
                numerical['solver']='direct'
            result=solve_solid(n,c,Material(**case['material']),**numerical,
                               segments=segments,progress=lambda s:emit(label+'.'+s))
            result['solver_info']['requested']=case['solver']
            if numerical['solver']!=case['solver']:
                result['solver_info']['selection_reason']='direct_for_low_quality_surface_tetrahedra'
            result['mesh_info']=info
            return result
        r=compute(cfg,'base');emit('checks');checks=diagnostics(cfg,r,segments)
        runs=[dict(name='base',pitch=cfg['pitch'],angle=cfg['angle'],metrics=metrics(r))]
        if options['convergence']:
            refined={**cfg,'pitch':cfg['pitch']/1.5}
            try:
                fine=compute(refined,'refined');fm=metrics(fine);bm=runs[0]['metrics']
                du=abs(fm['max_displacement_mm']-bm['max_displacement_mm'])/max(fm['max_displacement_mm'],1e-20)
                ds=abs(fm['stress_p95_MPa']-bm['stress_p95_MPa'])/max(fm['stress_p95_MPa'],1e-20)
                checks.append(convergence_check(r,fine,du,ds,cfg['force']))
                if r['mesh_info']['mode']=='tetra':
                    enabled=bool(r['mesh_info'].get('boundary_refinement') and fine['mesh_info'].get('boundary_refinement'))
                    checks.append(check('BOUNDARY_REFINEMENT','pass' if enabled else 'not_checked',
                                        '边界细分已启用；仅证明允许细分，不单独证明薄壁或局部应力收敛',
                                        base_elements=len(r['cells']),refined_elements=len(fine['cells'])))
                runs.append(dict(name='refined',pitch=refined['pitch'],angle=refined['angle'],metrics=fm))
                write_npz(out/'refined.npz',fine);del fine
            except ValueError as exc:
                checks.append(check('MESH_CONVERGENCE','not_checked','细化网格未完成，不能认定收敛',reason=str(exc)))
        else:checks.append(check('MESH_CONVERGENCE','not_checked','调用者关闭了网格收敛检查'))
        if options['angles']:
            if segments is not None:raise JobError('INVALID_CASE','角度对比仅用于统一走线；请移除 G-code 或单独运行路径工况')
            for i,angle in enumerate(options['angles']):
                case={**cfg,'angle':angle};result=compute(case,f'angle_{i}')
                runs.append(dict(name=f'angle_{i}',pitch=case['pitch'],angle=angle,metrics=metrics(result)))
                write_npz(out/f'angle_{i}.npz',result);del result
        emit('export');write_vtk(out/'result.vtk',r);write_npz(out/'result.npz',r)
        summary=dict(settings=cfg,inputs=inputs,units='mm,N,MPa',metrics=metrics(r),checks=checks,checks_scope='base_case_only',runs=runs,
                     quality='needs_review' if any(c['status'] in ('warn','not_checked') for c in checks) else 'numerical_checks_passed',scope='numerical_screening_only')
        emit('render');render(r,out/'preview.png');write_report(out/'report.html',summary)
        (out/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        queue.put({'type':'complete','summary':summary})
    except Exception as exc:
        if isinstance(exc,JobError):code=exc.code
        elif isinstance(exc,MemoryError):code='MEMORY_LIMIT'
        elif isinstance(exc,(ValueError,TypeError)):
            code='MESH_ERROR' if stage.endswith('.mesh') else ('GCODE_ERROR' if stage=='gcode_read' else 'SOLVER_ERROR')
        elif isinstance(exc,OSError):code='IO_ERROR'
        else:code='INTERNAL_ERROR'
        queue.put({'type':'error','error':{'code':code,'message':str(exc),'stage':stage}})
