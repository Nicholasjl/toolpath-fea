"""General-purpose local 3D toolpath FEA desktop interface."""
import json
import threading
from pathlib import Path
from dataclasses import asdict
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from solver import read_gcode
from solid import Material,box_mesh,stl_mesh,boundary_faces,select_faces,solve_solid,write_vtk


class SolidApp:
    def __init__(self,root):
        self.root=root; root.title('PathFEA 3D · 通用打印件分析'); root.geometry('1320x880')
        self.values={}; self.result=None; self.mesh=None; self.busy=False; self.output=None
        panel=ttk.Frame(root,padding=12,width=375); panel.pack(side='left',fill='y')
        ttk.Label(panel,text='PathFEA 3D',font=('Segoe UI',23,'bold')).pack(anchor='w')
        ttk.Label(panel,text='通用 STL · 三维各向异性 · 线弹性静力').pack(anchor='w',pady=(0,12))
        tabs=ttk.Notebook(panel); tabs.pack(fill='both',expand=True)
        geo=ttk.Frame(tabs,padding=10); mat=ttk.Frame(tabs,padding=10); bc=ttk.Frame(tabs,padding=10)
        tabs.add(geo,text='几何 / 走线'); tabs.add(mat,text='材料'); tabs.add(bc,text='约束 / 载荷')
        self.entry(geo,'stl','STL 路径（空=测试块）','')
        ttk.Button(geo,text='选择 STL',command=lambda:self.choose('stl','*.stl')).pack(fill='x')
        for k,label,value in [('size','测试块 XYZ / mm','40,12,8'),('pitch','网格间距 / mm','2'),('gcode','G-code 路径（可选）','')]:
            self.entry(geo,k,label,value)
        ttk.Button(geo,text='选择 G-code',command=lambda:self.choose('gcode','*.gcode *.gco')).pack(fill='x')
        self.entry(geo,'offset','路径 XYZ 平移 / mm','0,0,0')
        self.entry(geo,'angle','无 G-code 时走线角 / °','0')
        ttk.Label(geo,text='无 G-code 时打印层法线').pack(anchor='w',pady=(8,0))
        self.axis=tk.StringVar(value='Z'); ttk.Combobox(geo,textvariable=self.axis,values=['X','Y','Z'],state='readonly').pack(fill='x')
        ttk.Label(geo,text='网格：voxel 体素 / tetra 贴体四面体').pack(anchor='w')
        self.mesh_mode=tk.StringVar(value='voxel');ttk.Combobox(geo,textvariable=self.mesh_mode,values=['voxel','tetra'],state='readonly').pack(fill='x')
        ttk.Label(geo,text='STL 坐标与 G-code 坐标保留不变。\n用 XYZ 平移对齐；不会自动缩放或居中。\nG-code 模式的层法线为世界 Z。\n预览中黑线显示挤出路径。',wraplength=320).pack(anchor='w',pady=14)
        for k,label in [('e1','沿线 E₁'),('e2','层内横向 E₂'),('e3','层间 E₃'),('g12','剪切 G₁₂'),('g23','剪切 G₂₃'),('g13','剪切 G₁₃'),('nu12','泊松比 ν₁₂'),('nu23','泊松比 ν₂₃'),('nu13','泊松比 ν₁₃')]:
            self.entry(mat,k,label+(' / MPa' if not k.startswith('nu') else ''),str(getattr(Material(),k)))
        ttk.Label(mat,text='默认参数仅为演示。九项参数需按打印工艺标定；不能凭材料名称推算层间强度。',wraplength=310).pack(pady=12)
        ttk.Label(mat,text='求解器：auto / direct / amg').pack(anchor='w')
        self.solver=tk.StringVar(value='auto');ttk.Combobox(mat,textvariable=self.solver,values=['auto','direct','amg'],state='readonly').pack(fill='x')
        self.advanced={'surface_remesh_mm':0,'snap_tolerance':.0001,'rtol':1e-9,'maxiter':2000,'chunk_size':4096}
        for key,title,default in [('fixed','完全固定区域','X-'),('loaded','施加合力区域','X+')]:
            ttk.Label(bc,text=title).pack(anchor='w',pady=(10,4))
            self.values[key]=tk.StringVar(value=default)
            ttk.Combobox(bc,textvariable=self.values[key],values=['X-','X+','Y-','Y+','Z-','Z+','空间盒'],state='readonly').pack(fill='x')
            self.entry(bc,key+'_box','盒范围 xmin,ymin,zmin,xmax,ymax,zmax','0,0,0,1,12,8' if key=='fixed' else '39,0,0,40,12,8')
        self.entry(bc,'force','总合力 Fx,Fy,Fz / N','0,0,-10')
        ttk.Label(bc,text='X± 等选择全局包围盒极值平面上的表面。\n空间盒选择表面三角形中心位于盒内的区域。\n固定区：蓝；加载区：橙。\n力按选中表面面积分配。固定区三向位移为零。',wraplength=310).pack(anchor='w',pady=14)
        self.buttons=[]
        for text,cmd in [('预览网格 / 选区',lambda:self.start(False)),('运行三维分析',lambda:self.start(True)),('导出 JSON + VTK',self.export)]:
            btn=ttk.Button(panel,text=text,command=cmd); btn.pack(fill='x',pady=4); self.buttons.append(btn)
        row=ttk.Frame(panel);row.pack(fill='x')
        ttk.Button(row,text='保存工况',command=self.save_settings).pack(side='left',fill='x',expand=True)
        ttk.Button(row,text='载入工况',command=self.load_settings).pack(side='left',fill='x',expand=True)
        ttk.Label(panel,text='实心等效模型：不重建填充空隙，不含螺栓接触、预紧、塑性或开裂。',wraplength=340,foreground='#826134').pack(pady=8)
        main=ttk.Frame(root,padding=10); main.pack(side='right',fill='both',expand=True)
        bar=ttk.Frame(main); bar.pack(fill='x')
        self.field=tk.StringVar(value='位移 / mm')
        box=ttk.Combobox(bar,textvariable=self.field,state='readonly',width=22,values=['位移 / mm','等效应力 / MPa','沿线应力 / MPa','横向应力 / MPa','层间正应力 / MPa','路径距离 / mm','走线角度 / °'])
        box.pack(side='left'); box.bind('<<ComboboxSelected>>',lambda e:self.draw())
        self.deform=tk.BooleanVar(value=False); ttk.Checkbutton(bar,text='放大变形',variable=self.deform,command=self.draw).pack(side='left',padx=10)
        ttk.Label(bar,text='显示 Z 下部比例').pack(side='left')
        self.cut=tk.DoubleVar(value=1)
        ttk.Scale(bar,from_=.05,to=1,variable=self.cut,command=lambda _:self.draw(),length=100).pack(side='left')
        self.figure=Figure(figsize=(9,7),dpi=100,facecolor='#f5f7fa')
        self.canvas=FigureCanvasTkAgg(self.figure,master=main); self.canvas.get_tk_widget().pack(fill='both',expand=True)
        NavigationToolbar2Tk(self.canvas,main)
        self.status=tk.StringVar(value='先预览网格和固定/加载区域，再运行分析。鼠标可旋转三维视图。')
        ttk.Label(main,textvariable=self.status,wraplength=850,justify='left').pack(fill='x',pady=10)
        self.draw()

    def save_settings(self):
        try:
            cfg=self.settings()
            path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='case.json')
            if path:Path(path).write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        except Exception as exc:messagebox.showerror('保存失败',str(exc))

    def load_settings(self):
        if self.busy:return
        path=filedialog.askopenfilename(filetypes=[('工况 JSON','*.json')])
        if not path:return
        try:
            cfg=json.loads(Path(path).read_text(encoding='utf-8'));cfg=cfg.get('settings',cfg)
            for key in ('stl','gcode','pitch','angle'):self.values[key].set(str(cfg[key]))
            for key in ('size','offset','force'):self.values[key].set(','.join(map(str,cfg[key])))
            for key,value in cfg['material'].items():self.values[key].set(str(value))
            for key in ('fixed','loaded'):
                value=cfg[key]
                self.values[key].set(value if isinstance(value,str) else '空间盒')
                if not isinstance(value,str):self.values[key+'_box'].set(','.join(map(str,np.array(value).ravel())))
            self.axis.set(cfg['build_axis'])
            self.mesh_mode.set(cfg.get('mesh_mode','voxel'));self.solver.set(cfg.get('solver','auto'))
            self.advanced={key:cfg.get(key,default) for key,default in {'surface_remesh_mm':0,'snap_tolerance':.0001,'rtol':1e-9,'maxiter':2000,'chunk_size':4096}.items()}
            self.result=None;self.mesh=None;self.draw();self.status.set('工况已载入，请重新预览或求解。')
        except Exception as exc:messagebox.showerror('载入失败',str(exc))

    def entry(self,parent,key,label,value):
        ttk.Label(parent,text=label).pack(anchor='w',pady=(7,0))
        var=tk.StringVar(value=value); self.values[key]=var
        ttk.Entry(parent,textvariable=var,width=40).pack(fill='x')

    def choose(self,key,pattern):
        path=filedialog.askopenfilename(filetypes=[(key,pattern)])
        if path:self.values[key].set(path)

    @staticmethod
    def vector(text,n):
        result=[float(x.strip()) for x in text.replace('，',',').split(',')]
        if len(result)!=n or not np.isfinite(result).all():raise ValueError(f'需要 {n} 个逗号分隔的有限数字')
        return result

    def settings(self):
        v={k:x.get() for k,x in self.values.items()}
        regions={}
        for key in ('fixed','loaded'):
            regions[key]=np.array(self.vector(v[key+'_box'],6)).reshape(2,3).tolist() if v[key]=='空间盒' else v[key]
        return dict(stl=v['stl'],size=self.vector(v['size'],3),pitch=float(v['pitch']),gcode=v['gcode'],
                    material={k:float(v[k]) for k in asdict(Material())},angle=float(v['angle']),build_axis=self.axis.get(),
                    offset=self.vector(v['offset'],3),force=self.vector(v['force'],3),mesh_mode=self.mesh_mode.get(),solver=self.solver.get(),**self.advanced,**regions)

    def start(self,analysis):
        if self.busy:return
        try:cfg=self.settings()
        except Exception as exc:messagebox.showerror('参数错误',str(exc));return
        self.busy=True; self.output=None
        for b in self.buttons:b.state(['disabled'])
        self.status.set('正在生成体网格'+('、映射走线并求解…' if analysis else '并选择表面…'))
        def worker():
            try:
                if cfg['stl'] and cfg['mesh_mode']=='tetra':
                    from meshing import tetra_mesh
                    nodes,cells,mesh_info=tetra_mesh(cfg['stl'],cfg['pitch'],cfg['snap_tolerance'],surface_remesh_mm=cfg.get('surface_remesh_mm',0))
                else:
                    nodes,cells=stl_mesh(cfg['stl'],cfg['pitch']) if cfg['stl'] else box_mesh(cfg['size'],cfg['pitch'])
                    mesh_info={'mode':'voxel'}
                paths=read_gcode(cfg['gcode']) if cfg['gcode'] else None
                faces,owners=boundary_faces(cells)
                fixed=select_faces(nodes,faces,cfg['fixed']); loaded=select_faces(nodes,faces,cfg['loaded'])
                if not fixed.any() or not loaded.any():raise ValueError('选区没有选中表面，请调整空间盒范围')
                mesh=dict(nodes=nodes,cells=cells,faces=faces,owners=owners,fixed=fixed,loaded=loaded,paths=paths)
                chosen_solver=cfg['solver']
                if chosen_solver=='auto' and mesh_info.get('mode')=='tetra' and mesh_info.get('quality_min',1)<1e-5:chosen_solver='direct'
                result=solve_solid(nodes,cells,Material(**cfg['material']),fixed=cfg['fixed'],loaded=cfg['loaded'],force=cfg['force'],angle=cfg['angle'],build_axis=cfg['build_axis'],segments=paths,offset=cfg['offset'],solver=chosen_solver,**{k:cfg[k] for k in ('rtol','maxiter','chunk_size')}) if analysis else None
                if result is not None:
                    result['mesh_info']=mesh_info;result['solver_info']['requested']=cfg['solver']
                self.output=('ok',(cfg,mesh,result))
            except Exception as exc:self.output=('error',str(exc))
        threading.Thread(target=worker,daemon=True).start();self.root.after(100,self.poll)

    def poll(self):
        if self.output is None:self.root.after(100,self.poll);return
        self.busy=False
        for b in self.buttons:b.state(['!disabled'])
        kind,data=self.output
        if kind=='error':self.status.set('操作失败；保留上次成功结果。');messagebox.showerror('分析失败',data);return
        self.cfg,self.mesh,self.result=data
        m=self.mesh;r=self.result
        summary=f"{len(m['nodes'])} 节点 / {len(m['cells'])} 四面体；固定 {m['fixed'].sum()} 面，加载 {m['loaded'].sum()} 面"
        if r is not None:
            summary+=f"\n最大位移 {np.linalg.norm(r['displacement'],axis=1).max():.5g} mm；反力 {r['reactions'].sum(axis=0).round(5)} N；残差 {r['relative_residual']:.2e}"
            if m['paths'] is not None:summary+=f"；路径距离最大 {r['path_distance'].max():.3g} mm"
        self.status.set(summary);self.draw()

    def draw(self):
        self.figure.clear();ax=self.figure.add_subplot(111,projection='3d')
        ax.set_xlabel('X / mm');ax.set_ylabel('Y / mm');ax.set_zlabel('Z / mm')
        if self.mesh is None:
            ax.set_title('Import STL or preview the verification block');self.canvas.draw_idle();return
        m=self.mesh;r=self.result;p=m['nodes'].copy();faces=m['faces'];owners=m['owners']
        if self.cut.get()<.999:
            centers=m['nodes'][m['cells']].mean(axis=1)
            threshold=p[:,2].min()+np.ptp(p[:,2])*self.cut.get()
            selected=np.flatnonzero(centers[:,2]<=threshold)
            if len(selected):
                faces,local_owners=boundary_faces(m['cells'][selected]);owners=selected[local_owners]
        scale=0
        if r is not None and self.deform.get():
            maximum=np.linalg.norm(r['displacement'],axis=1).max()
            scale=.06*np.ptp(p,axis=0).max()/maximum if maximum else 0;p+=r['displacement']*scale
        title='Mesh / BC preview (blue=fixed, orange=loaded)'
        if r is None:
            colors=np.tile([.69,.76,.81,1.],(len(faces),1))
            fixed=select_faces(m['nodes'],faces,self.cfg['fixed']);loaded=select_faces(m['nodes'],faces,self.cfg['loaded'])
            colors[fixed]=[.1,.4,.9,1];colors[loaded]=[1,.45,.05,1]
        else:
            key=self.field.get()
            if key=='位移 / mm':values=np.linalg.norm(r['displacement'],axis=1)[faces].mean(axis=1);title='Displacement / mm'
            else:
                values,title={'等效应力 / MPa':(r['von_mises'],'von Mises / MPa (not failure)'),
                    '沿线应力 / MPa':(r['local_stress'][:,0],'Along path stress / MPa'),
                    '横向应力 / MPa':(r['local_stress'][:,1],'Transverse stress / MPa'),
                    '层间正应力 / MPa':(r['local_stress'][:,2],'Layer normal stress / MPa'),
                    '路径距离 / mm':(r['path_distance'],'Path distance / mm'),
                    '走线角度 / °':(r['angles'],'Path angle / degree')}[key]
                values=values[owners]
            norm=Normalize(float(values.min()),float(values.max())+1e-15)
            colors=matplotlib.colormaps['viridis'](norm(values))
            self.figure.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=ax,shrink=.65,pad=.12)
        ax.add_collection3d(Poly3DCollection(p[faces],facecolors=colors,edgecolors=(0,0,0,.12),linewidths=.15))
        if r is None and m['paths'] is not None:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            seg=m['paths'][::max(1,len(m['paths'])//10000)]
            lines=np.stack([seg[:,[0,1,4]],seg[:,[2,3,4]]],axis=1)+np.array(self.cfg['offset'])
            ax.add_collection3d(Line3DCollection(lines,colors='black',linewidths=.4,alpha=.3))
        ax.auto_scale_xyz(p[:,0],p[:,1],p[:,2]);ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),1e-6))
        ax.set_title(title+(f' | deformation x{scale:.2g}' if scale else ''))
        self.figure.tight_layout();self.canvas.draw_idle()

    def export(self):
        if self.result is None:messagebox.showinfo('导出','请先运行三维分析');return
        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='solid_result.json')
        if not path:return
        try:
            payload={'settings':self.cfg,'units':'mm,N,MPa','model':'solid linear tetrahedral orthotropic elasticity',
                     'result':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in self.result.items()}}
            Path(path).write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
            write_vtk(Path(path).with_suffix('.vtk'),self.result)
            self.status.set(f'已导出 {path} 和同名 VTK，可在 ParaView 查看完整体网格。')
        except Exception as exc:messagebox.showerror('导出失败',str(exc))


if __name__=='__main__':
    root=tk.Tk();SolidApp(root);root.mainloop()
