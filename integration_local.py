"""Run a manifest of local engineering cases through the public supervised CLI.

python integration_local.py cases.json --output output/integration
Manifest: [{"name":"part", "case":{...}, "convergence":false,
            "expected_error":"GCODE_ERROR" (optional)}]
Input paths use the manifest directory. Source geometry is never modified.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--memory-mb',type=float,default=2048)
    args=parser.parse_args()
    entries=json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for entry in entries:
        name=entry['name']
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):raise ValueError('Invalid case name')
        cfg=dict(entry['case'])
        before={}
        for key in ('stl','gcode'):
            if cfg.get(key):
                p=Path(cfg[key]);p=p if p.is_absolute() else args.manifest.resolve().parent/p
                cfg[key]=str(p.resolve());before[key]=digest(p)
        case_path=out/f'{name}.case.json';case_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        command=[sys.executable,str(Path(__file__).with_name('run3d.py')),str(case_path),
                 '--output',str(out/name),'--timeout',str(args.timeout),'--memory-mb',str(args.memory_mb)]
        if not entry.get('convergence',False):command.append('--no-convergence')
        print(f'START {name}',flush=True)
        started=time.monotonic();issues=[];result={}
        with (out/f'{name}.stderr.jsonl').open('w',encoding='utf-8') as err:
            # The CLI owns its worker timeout and kills the process tree on expiry.
            p=subprocess.run(command,stdout=subprocess.PIPE,stderr=err,encoding='utf-8',env={**os.environ,'PYTHONUTF8':'1'})
        (out/f'{name}.stdout.json').write_text(p.stdout,encoding='utf-8')
        try:
            result=json.loads(p.stdout)
            if result['exit_code']!=p.returncode:issues.append('Process and manifest exit codes differ')
            saved=json.loads((out/f'{name}.json').read_text(encoding='utf-8'))
            if saved['run_id']!=result['run_id']:issues.append('Manifest run_id mismatch')
            for key,sha in before.items():
                if digest(cfg[key])!=sha:issues.append(f'Source modified: {key}')
            if result['status']=='completed':
                artifacts=result['artifacts']
                for filename in ('result.npz','result.vtk','analysis.json','preview.png','report.html'):
                    target=Path(artifacts[filename])
                    if not target.is_relative_to(out) or not target.is_file() or target.stat().st_size==0:issues.append(f'Invalid artifact: {filename}')
                with np.load(artifacts['result.npz'],allow_pickle=False) as data:
                    for field in ('displacement','stress','forces','reactions'):
                        if not np.isfinite(data[field]).all():issues.append(f'Nonfinite {field}')
                    force=np.asarray(cfg.get('force',[0,0,-10]),float)
                    if not np.allclose(data['forces'].sum(axis=0),force,rtol=1e-7,atol=1e-8):issues.append('Applied force mismatch')
                    if float(data['relative_residual'])>1e-6:issues.append('Raw residual failed')
                analysis=result['analysis']
                if next(c for c in analysis['checks'] if c['code']=='EQUILIBRIUM')['status']!='pass':issues.append('Equilibrium failed')
                for key,sha in before.items():
                    if analysis['inputs'][key]['sha256']!=sha:issues.append(f'Input hash mismatch: {key}')
                if entry.get('expected_error'):issues.append('Unexpected acceptance of rejection case')
            elif not result.get('error',{}).get('code'):issues.append('Failure lacks error code')
        except Exception as exc:issues.append(f'{type(exc).__name__}: {exc}')
        code=(result.get('error') or {}).get('code')
        if issues:outcome='contract_failure'
        elif result.get('status')=='completed':outcome='solved'
        elif code==entry.get('expected_error'):outcome='expected_rejection'
        else:outcome='rejected_or_limited'
        row=dict(name=name,outcome=outcome,issues=issues,error=result.get('error'),
                 elapsed_seconds=time.monotonic()-started,resources=result.get('resources'),
                 quality=result.get('analysis',{}).get('quality'),metrics=result.get('analysis',{}).get('metrics'),
                 checks=result.get('analysis',{}).get('checks'),source_hashes=before,
                 manifest=str(out/f'{name}.json'))
        rows.append(row)
        (out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({k:row[k] for k in ('name','outcome','error','elapsed_seconds','issues')},ensure_ascii=False),flush=True)
    table=''
    for row in rows:
        metrics=row['metrics'] or {};resource=row['resources'] or {}
        values=[row['name'],row['outcome'],row['quality'],f"{row['elapsed_seconds']:.1f}",
                resource.get('peak_worker_rss_mib'),metrics.get('elements'),metrics.get('max_displacement_mm'),row['error'] or row['issues']]
        table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    report='''<!doctype html><meta charset="utf-8"><title>Local engineering integration tests</title>
<style>body{font:15px system-ui;margin:30px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}td{max-width:420px;overflow-wrap:anywhere}</style>
<h1>本地工程集成测试</h1><p>solved 表示端到端流程、工件、输入哈希、数值有限性、残差与平衡通过；不等于网格收敛或实际结构安全。expected_rejection 是预期不支持输入被正确拒绝；其他拒绝/资源限制单独记录。无走线输入的工况使用统一材料方向及演示材料。</p>
<table><tr><th>工况</th><th>结果</th><th>数值质量</th><th>秒</th><th>峰值 MiB</th><th>单元</th><th>最大位移 mm</th><th>错误/问题</th></tr>'''+table+'</table>'
    (out/'report.html').write_text(report,encoding='utf-8')
    # Unexpected model rejection is an integration failure too: do not report
    # a green process exit just because the error envelope was well formed.
    return int(any(r['outcome'] in ('contract_failure','rejected_or_limited') for r in rows))


if __name__=='__main__':raise SystemExit(main())
