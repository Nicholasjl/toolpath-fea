"""Local desktop frontend; run python app.py."""
import csv
import json
import threading
from dataclasses import asdict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from solver import Settings, solve, read_gcode


class App:
    def __init__(self, root):
        self.root = root
        root.title('PathFEA · 打印走线有限元分析')
        root.geometry('1250x850')
        self.result = None
        self.segments = None
        self.busy = False
        self.fields = {}
        side = ttk.Frame(root, padding=16); side.pack(side='left', fill='y')
        ttk.Label(side, text='PathFEA', font=('Segoe UI',24,'bold')).pack(anchor='w')
        ttk.Label(side, text='二维薄板 · 正交各向异性 · 线弹性').pack(anchor='w', pady=(0,12))
        labels = [('length','板长 mm'),('height','板高 mm'),('thickness','厚度 mm'),
                  ('nx','X 网格数'),('ny','Y 网格数'),('e1','沿线 E₁ / MPa'),('e2','横线 E₂ / MPa'),
                  ('g12','剪切 G₁₂ / MPa'),('nu12','泊松比 ν₁₂'),('angle','统一走线角 °'),
                  ('fx','右边总力 Fx / N'),('fy','右边总力 Fy / N')]
        defaults = asdict(Settings())
        for name,label in labels:
            row = ttk.Frame(side); row.pack(fill='x', pady=3)
            ttk.Label(row,text=label,width=20).pack(side='left')
            value = tk.StringVar(value=str(defaults[name])); self.fields[name]=value
            ttk.Entry(row,textvariable=value,width=12).pack(side='right')
        ttk.Separator(side).pack(fill='x',pady=10)
        ttk.Button(side,text='导入 G-code 路径',command=self.import_path).pack(fill='x')
        self.path_label = tk.StringVar(value='未导入：使用统一走线角')
        ttk.Label(side,textvariable=self.path_label,wraplength=280).pack(anchor='w',pady=5)
        self.layer = tk.StringVar()
        self.layer_box = ttk.Combobox(side,textvariable=self.layer,state='readonly',width=28)
        self.layer_box.pack(fill='x')
        ttk.Label(side,text='层 Z / mm；XY 原点移至该层包围盒左下角',wraplength=280).pack(anchor='w')
        ttk.Button(side,text='清除路径',command=self.clear_path).pack(fill='x',pady=4)
        self.run_button = ttk.Button(side,text='运行有限元分析',command=self.run)
        self.run_button.pack(fill='x',pady=(12,4))
        self.compare_button = ttk.Button(side,text='对比 0° / 45° / 90° 统一走线',command=lambda:self.run(True))
        self.compare_button.pack(fill='x')
        ttk.Button(side,text='导出结果 JSON + CSV',command=self.export).pack(fill='x',pady=4)
        ttk.Label(side,text='边界：左边全固定；右边均布合力。\n材料默认值仅用于演示，需要实测标定。\n连续实心矩形；不模拟填充孔隙、层间失效\n或三维弯曲。等效应力不作为各向异性\n材料的强度失效判据。',wraplength=290,foreground='#78603b').pack(anchor='w',pady=12)
        main = ttk.Frame(root,padding=12); main.pack(side='right',fill='both',expand=True)
        top = ttk.Frame(main); top.pack(fill='x')
        self.plot_type = tk.StringVar(value='位移幅值 / mm')
        combo = ttk.Combobox(top,textvariable=self.plot_type,state='readonly',width=26,
            values=['位移幅值 / mm','等效应力 / MPa','沿线应力 σ₁ / MPa','横线应力 σ₂ / MPa','走线角度 / °','路径距离 / mm'])
        combo.pack(side='left'); combo.bind('<<ComboboxSelected>>',lambda e:self.draw())
        self.deform = tk.BooleanVar(value=False)
        ttk.Checkbutton(top,text='显示放大变形',variable=self.deform,command=self.draw).pack(side='left',padx=12)
        self.figure = Figure(figsize=(9,6),dpi=100,facecolor='#f5f7fa')
        self.canvas = FigureCanvasTkAgg(self.figure,master=main)
        self.canvas.get_tk_widget().pack(fill='both',expand=True)
        NavigationToolbar2Tk(self.canvas,main)
        self.status = tk.StringVar(value='准备就绪。填写参数后运行；单位：mm、N、MPa。')
        ttk.Label(main,textvariable=self.status,wraplength=850,justify='left').pack(fill='x',pady=8)
        self.draw()

    def clear_path(self):
        self.segments=None; self.path_label.set('未导入：使用统一走线角')
        self.layer_box['values']=[]; self.layer.set('')

    def import_path(self):
        if self.busy: return
        path=filedialog.askopenfilename(filetypes=[('G-code','*.gcode *.gco'),('All files','*.*')])
        if not path: return
        try:
            segments=read_gcode(path)
            layers=np.unique(np.round(segments[:,4],5))
            self.segments=segments
            self.layer_box['values']=[str(z) for z in layers]
            self.layer.set(str(layers[len(layers)//2]))
            self.path_label.set(f'{Path(path).name}\n{len(layers)} 层 / {len(segments)} 条挤出段')
        except Exception as exc: messagebox.showerror('路径导入失败',str(exc))

    def run(self, compare=False):
        if self.busy: return
        try:
            cfg=Settings(**{k:(int(v.get()) if k in ('nx','ny') else float(v.get())) for k,v in self.fields.items()})
            paths=None
            if self.segments is not None and not compare:
                z=float(self.layer.get())
                paths=self.segments[np.isclose(self.segments[:,4],z,atol=1e-5,rtol=0)].copy()
                if not len(paths): raise ValueError('所选层没有有效路径')
                origin=np.minimum(paths[:,:2].min(axis=0),paths[:,2:4].min(axis=0))
                paths[:,:2]-=origin; paths[:,2:4]-=origin
                if len(paths)*cfg.nx*cfg.ny*2 > 100_000_000:
                    raise ValueError('路径映射计算量过大，请降低网格数或裁剪单层路径')
        except Exception as exc:
            messagebox.showerror('参数错误',str(exc)); return
        self.busy=True
        self.run_button.state(['disabled']); self.compare_button.state(['disabled'])
        self.status.set('正在装配稀疏刚度矩阵并求解…')
        self.job_output=None
        def work():
            try:
                if compare:
                    results=[]
                    for angle in (0,45,90):
                        test=Settings(**asdict(cfg)); test.angle=angle
                        r=solve(test)
                        results.append((angle,float(np.linalg.norm(r['displacement'],axis=1).max())))
                    self.job_output=('compare',results)
                else: self.job_output=('result',(solve(cfg,paths),cfg,paths))
            except Exception as exc: self.job_output=('error',str(exc))
        threading.Thread(target=work,daemon=True).start()
        self.root.after(100,self.poll)

    def poll(self):
        if self.job_output is None:
            self.root.after(100,self.poll); return
        kind,value=self.job_output
        self.busy=False
        self.run_button.state(['!disabled']); self.compare_button.state(['!disabled'])
        if kind=='error':
            self.status.set('计算失败'); messagebox.showerror('计算失败',value)
        elif kind=='compare':
            self.status.set('统一走线对比，最大位移：'+'；'.join(f'{a}° = {u:.5g} mm' for a,u in value)+'。当前云图保持原分析结果。')
        else:
            self.result,self.solved_cfg,self.solved_paths=value
            r=self.result
            self.status.set(f"最大位移 {np.linalg.norm(r['displacement'],axis=1).max():.5g} mm | "
                f"最大等效应力 {r['von_mises'].max():.5g} MPa | 反力 {r['reaction'].round(4)} N\n"
                f"自由自由度相对残差 {r['relative_residual']:.2e} | {len(r['cells'])} 单元 | "
                +(f"最近路径最大距离 {r['path_distance'].max():.3g} mm（距离大表示覆盖不足）" if self.solved_paths is not None else '使用统一走线方向'))
            self.draw()

    def draw(self):
        self.figure.clear(); ax=self.figure.add_subplot(111)
        ax.set_xlabel('X / mm'); ax.set_ylabel('Y / mm'); ax.set_aspect('equal')
        if self.result is None:
            ax.text(.5,.5,'PathFEA\nRun an analysis to view results',ha='center',va='center',transform=ax.transAxes,color='#536879')
            self.canvas.draw(); return
        r=self.result; p=r['nodes'].copy(); key=self.plot_type.get()
        scale=0
        if self.deform.get():
            max_u=np.linalg.norm(r['displacement'],axis=1).max()
            scale=.08*max(self.solved_cfg.length,self.solved_cfg.height)/max_u if max_u else 0
            p+=scale*r['displacement']
        if key.startswith('位移'):
            artist=ax.tripcolor(p[:,0],p[:,1],r['cells'],np.linalg.norm(r['displacement'],axis=1),shading='gouraud',cmap='viridis')
            title='Displacement magnitude / mm'
        else:
            data,title={'等效应力 / MPa':(r['von_mises'],'von Mises / MPa (diagnostic only)'),
                '沿线应力 σ₁ / MPa':(r['local_stress'][:,0],'Along-path stress / MPa'),
                '横线应力 σ₂ / MPa':(r['local_stress'][:,1],'Transverse stress / MPa'),
                '走线角度 / °':(r['angles'],'Path angle / deg'),
                '路径距离 / mm':(r['path_distance'],'Nearest path distance / mm')}[key]
            artist=ax.tripcolor(p[:,0],p[:,1],r['cells'],facecolors=data,cmap='viridis')
        self.figure.colorbar(artist,ax=ax,shrink=.65)
        if self.solved_paths is not None and not self.deform.get():
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(self.solved_paths[:,:4].reshape(-1,2,2),colors='black',linewidths=.3,alpha=.25))
        ax.set_title(title+(f' | deformation x{scale:.2g}' if scale else ''))
        self.figure.tight_layout(); self.canvas.draw()

    def export(self):
        if self.result is None: return messagebox.showinfo('导出','请先运行分析')
        path=filedialog.asksaveasfilename(defaultextension='.json',initialfile='fea_result.json')
        if not path: return
        try:
            r=self.result
            payload={'settings':asdict(self.solved_cfg),'units':'mm, N, MPa',
                     'model':'2D plane stress; solid rectangle; nearest-segment orthotropy',
                     'mapped_segments':self.solved_paths.tolist() if self.solved_paths is not None else None,
                     'result':{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}}
            Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            with Path(path).with_suffix('.csv').open('w',newline='',encoding='utf-8-sig') as file:
                writer=csv.writer(file); writer.writerow(['element','angle_deg','s11_MPa','s22_MPa','t12_MPa','von_mises_MPa','path_distance_mm'])
                for i in range(len(r['cells'])):
                    writer.writerow([i,r['angles'][i],*r['local_stress'][i],r['von_mises'][i],r['path_distance'][i]])
            self.status.set(f'已导出 {path} 及同名 CSV（保存的是上次成功求解的参数和结果）。')
        except Exception as exc: messagebox.showerror('导出失败',str(exc))


if __name__=='__main__':
    root=tk.Tk(); App(root); root.mainloop()
