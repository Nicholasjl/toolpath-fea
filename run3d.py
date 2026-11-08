"""Agent CLI: one JSON result on stdout, JSONL progress on stderr; isolated worker."""
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import uuid

EXIT_CODES={'INVALID_CASE':2,'INVALID_JSON':2,'INPUT_NOT_FOUND':2,'INVALID_MATERIAL':2,
            'MESH_ERROR':3,'GCODE_ERROR':3,'SOLVER_ERROR':3,'TIMEOUT':4,'MEMORY_LIMIT':5,
            'CANCELLED':6,'IO_ERROR':7,'INTERNAL_ERROR':7,'WORKER_CRASH':7,'DEPENDENCY_MISSING':7,'CHECKS_FAILED':8}


class CliError(Exception):pass


class Parser(argparse.ArgumentParser):
    def error(self,message):raise CliError(message)


def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def worker_entry(*args):
    # Heavy numerical imports belong to the monitored process, not the supervisor.
    # Native mesh libraries can print outside Python: isolate fd 1 from protocol stdout.
    native_log=open(Path(args[3])/'native.log','w',encoding='utf-8')
    os.dup2(native_log.fileno(),1)
    try:
        from agent_worker import worker_main
    except Exception as exc:
        args[0].put({'type':'error','error':{'code':'DEPENDENCY_MISSING' if isinstance(exc,ModuleNotFoundError) else 'INTERNAL_ERROR','stage':'imports','message':str(exc)}})
        return
    try:worker_main(*args)
    finally:native_log.close()


def supervise(cfg,base,output,options):
    """Return envelope; numerical work runs only in spawned child process."""
    import psutil
    started=time.monotonic();run_id=uuid.uuid4().hex
    output=Path(output).resolve()
    directory=output.parent/(output.stem+'_runs')/run_id
    directory.mkdir(parents=True,exist_ok=False)
    manifest=output.with_suffix('.json')
    envelope={'schema_version':'1.0','run_id':run_id,'status':'running','artifacts':{},'error':None}
    atomic_json(manifest,envelope)
    log=directory/'progress.jsonl';peak=0;stage='starting';summary=None;error=None;process=None
    def progress(event):
        event={'run_id':run_id,**event}
        line=json.dumps(event,ensure_ascii=False,allow_nan=False)
        with log.open('a',encoding='utf-8') as stream:stream.write(line+'\n')
        print(line,file=sys.stderr,flush=True)
    try:
        progress({'type':'progress','stage':stage,'time':time.time()})
        cancel=options.get('cancel_file')
        if cancel and Path(cancel).exists():
            error={'code':'CANCELLED','message':'取消文件已存在，任务未启动','stage':stage}
        else:
            context=mp.get_context('spawn');events=context.Queue()
            process=context.Process(target=worker_entry,args=(events,cfg,base,str(directory),options))
            process.start();monitored=psutil.Process(process.pid)
            while True:
                if cancel and Path(cancel).exists():error={'code':'CANCELLED','message':'收到取消文件','stage':stage};break
                if time.monotonic()-started>options['timeout']:
                    error={'code':'TIMEOUT','message':f"超过 {options['timeout']} 秒时限",'stage':stage};break
                try:
                    rss=monitored.memory_info().rss
                    for child in monitored.children(recursive=True):
                        try:rss+=child.memory_info().rss
                        except psutil.NoSuchProcess:pass
                    peak=max(peak,rss)
                    if rss>options['memory_mb']*1024**2:
                        error={'code':'MEMORY_LIMIT','message':f"进程 RSS 超过 {options['memory_mb']} MiB",'stage':stage};break
                except psutil.NoSuchProcess:pass
                try:event=events.get(timeout=.1)
                except queue.Empty:
                    if not process.is_alive():
                        error={'code':'WORKER_CRASH','message':f'工作进程退出，exitcode={process.exitcode}','stage':stage};break
                    continue
                if event['type']=='progress':stage=event['stage'];progress(event)
                elif event['type']=='complete':summary=event['summary'];break
                elif event['type']=='error':error=event['error'];break
    except KeyboardInterrupt:error={'code':'CANCELLED','message':'收到中断信号','stage':stage}
    except Exception as exc:error={'code':'INTERNAL_ERROR','message':str(exc),'stage':stage}
    finally:
        if process is not None:
            if process.is_alive():
                if error:process.terminate()
                process.join(timeout=3)
                if process.is_alive():process.kill();process.join(timeout=3)
            else:process.join()
    if summary is not None and error is None:
        envelope['analysis']=summary
        envelope['artifacts']={p.name:str(p.resolve()) for p in directory.iterdir() if p.is_file()}
        if options['require_checks'] and summary['quality']!='numerical_checks_passed':
            error={'code':'CHECKS_FAILED','message':'求解完成，但存在警告或未完成的校核；结果仅供复核','stage':'checks'}
    envelope['status']='failed' if error else 'completed';envelope['error']=error
    envelope['resources']={'elapsed_seconds':time.monotonic()-started,'peak_worker_rss_mib':peak/1024**2,
                           'memory_monitor':'sampled_RSS_100ms_not_OS_hard_limit'}
    envelope['progress_log']=str(log)
    envelope['exit_code']=EXIT_CODES.get(error['code'],7) if error else 0
    progress({'type':'finished','status':envelope['status'],'exit_code':envelope['exit_code'],'time':time.time()})
    atomic_json(manifest,envelope)
    return envelope


def main(argv=None):
    parser=Parser(description=__doc__)
    parser.add_argument('case',help='JSON 工况；输入相对路径以此文件所在目录解析')
    parser.add_argument('--output',default='output/result',help='结果清单前缀；工件放入独立 run_id 子目录')
    parser.add_argument('--timeout',type=float,default=300,help='整个作业时限，秒')
    parser.add_argument('--memory-mb',type=float,default=2048,help='工作进程 RSS 阈值，MiB，100ms 采样')
    parser.add_argument('--cancel-file',help='创建此文件请求取消；调用前已有该文件则不启动')
    parser.add_argument('--no-convergence',action='store_true',help='关闭默认两档网格对比；标记为 not_checked')
    parser.add_argument('--angles',help='对比统一走线角，如 0,45,90；不能与 G-code 同用')
    parser.add_argument('--require-checks',action='store_true',help='存在警告或未校核项时返回退出码 8')
    args=None
    try:
        args=parser.parse_args(argv)
        if Path(args.output).with_suffix('.json').resolve()==Path(args.case).resolve():
            raise CliError('输出清单不得覆盖输入工况文件')
        if not math.isfinite(args.timeout) or args.timeout<=0 or not math.isfinite(args.memory_mb) or args.memory_mb<=0:
            raise CliError('timeout 和 memory-mb 必须为正有限数')
        angles=[]
        if args.angles:
            try:angles=[float(x) for x in args.angles.split(',')]
            except ValueError:raise CliError('angles 必须为逗号分隔角度')
            if len(angles)>12 or not all(map(math.isfinite,angles)):raise CliError('angles 最多 12 个有限角度')
        try:cfg=json.loads(Path(args.case).read_text(encoding='utf-8-sig'),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(f'非法常数 {v}')))
        except (ValueError,UnicodeError) as exc:raise CliError(f'INVALID_JSON: {exc}') from exc
        if isinstance(cfg,dict) and cfg.get('gcode') and angles:raise CliError('G-code 与 angles 不能同时使用')
        options=dict(timeout=args.timeout,memory_mb=args.memory_mb,cancel_file=str(Path(args.cancel_file).resolve()) if args.cancel_file else None,
                     convergence=not args.no_convergence,angles=angles,require_checks=args.require_checks)
        result=supervise(cfg,str(Path(args.case).resolve().parent),args.output,options)
    except Exception as exc:
        if isinstance(exc,CliError):code='INVALID_JSON' if str(exc).startswith('INVALID_JSON:') else 'INVALID_CASE'
        elif isinstance(exc,FileNotFoundError):code='INPUT_NOT_FOUND'
        elif isinstance(exc,ModuleNotFoundError):code='DEPENDENCY_MISSING'
        elif isinstance(exc,OSError):code='IO_ERROR'
        else:code='INTERNAL_ERROR'
        result={'schema_version':'1.0','status':'failed','error':{'code':code,'message':str(exc),'stage':'startup'},'artifacts':{},'exit_code':EXIT_CODES[code]}
        if args is not None and Path(args.output).with_suffix('.json').resolve()!=Path(args.case).resolve():
            try:
                path=Path(args.output).with_suffix('.json');path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,result)
            except OSError:pass
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result['exit_code']


if __name__=='__main__':
    mp.freeze_support()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
