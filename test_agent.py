import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from agent_worker import validate_case,JobError,diagnostics
from solid import box_mesh,solve_solid

ROOT=Path(__file__).resolve().parent


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.case=self.folder/'case.json'
        self.case.write_text(json.dumps({'size':[12,8,8],'pitch':2}),encoding='utf-8')

    def invoke(self,*args):
        p=subprocess.run([sys.executable,str(ROOT/'run3d.py'),str(self.case),'--output',str(self.folder/'result'),*args],capture_output=True,encoding='utf-8',timeout=45)
        value=json.loads(p.stdout)
        self.assertEqual(p.returncode,value['exit_code'])
        self.assertEqual(value,json.loads((self.folder/'result.json').read_text(encoding='utf-8')))
        return value,p

    def test_complete_report_convergence_and_angles(self):
        value,p=self.invoke('--angles','0,90')
        self.assertEqual(value['status'],'completed',p.stderr)
        self.assertEqual(len(value['analysis']['runs']),4)
        self.assertTrue(any(c['code']=='MESH_CONVERGENCE' for c in value['analysis']['checks']))
        for name in ['preview.png','report.html','result.npz','result.vtk','analysis.json']:
            self.assertGreater(Path(value['artifacts'][name]).stat().st_size,0)
        self.assertTrue(all(json.loads(line)['run_id']==value['run_id'] for line in p.stderr.splitlines()))

    def test_invalid_json_and_stale_manifest(self):
        (self.folder/'result.json').write_text('{"status":"completed"}')
        self.case.write_text('{bad')
        value,_=self.invoke();self.assertEqual(value['error']['code'],'INVALID_JSON');self.assertEqual(value['artifacts'],{})

    def test_invalid_material(self):
        self.case.write_text('{"material":{"e1":-1}}')
        value,_=self.invoke();self.assertEqual(value['error']['code'],'INVALID_MATERIAL')

    def test_timeout(self):
        value,_=self.invoke('--timeout','0.001');self.assertEqual(value['error']['code'],'TIMEOUT')
        self.assertEqual(value['artifacts'],{})

    def test_memory_limit(self):
        value,_=self.invoke('--memory-mb','1');self.assertEqual(value['error']['code'],'MEMORY_LIMIT')

    def test_cancel_live_worker(self):
        cancel=self.folder/'cancel'
        p=subprocess.Popen([sys.executable,str(ROOT/'run3d.py'),str(self.case),'--output',str(self.folder/'result'),'--cancel-file',str(cancel)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
        try:
            while True:
                event=json.loads(p.stderr.readline())
                if event.get('stage')=='validation':break
            cancel.touch();out,_=p.communicate(timeout=15)
            self.assertEqual(json.loads(out)['error']['code'],'CANCELLED');self.assertEqual(p.returncode,6)
        finally:
            if p.poll() is None:p.kill();p.communicate()

    def test_strict_checks(self):
        value,_=self.invoke('--no-convergence','--require-checks')
        self.assertEqual(value['error']['code'],'CHECKS_FAILED');self.assertIn('report.html',value['artifacts'])

    def test_relative_paths_and_bad_fields(self):
        (self.folder/'part.stl').write_text('dummy')
        cfg=validate_case({'stl':'part.stl'},self.folder)
        self.assertEqual(cfg['stl'],str((self.folder/'part.stl').resolve()))
        for bad in [{'pitch':True},{'force':[1,2]},{'unknown':2},{'fixed':'X?'}]:
            with self.assertRaises(JobError):validate_case(bad,self.folder)

    def test_misaligned_paths_warn(self):
        cfg=validate_case({'size':[12,8,8],'pitch':2},self.folder)
        n,c=box_mesh(cfg['size'],2);segments=np.array([[100,100,120,100,1]],float)
        r=solve_solid(n,c,segments=segments)
        checks=diagnostics(cfg,r,segments)
        self.assertEqual(next(x for x in checks if x['code']=='PATH_ALIGNMENT')['status'],'warn')

    def test_stl_geometry_loss_warning(self):
        import trimesh
        path=self.folder/'part.stl'
        # A subvoxel protrusion is omitted while the main cube stays connected.
        main=trimesh.creation.box(extents=[8,8,8]);main.apply_translation([4,4,4])
        thin=trimesh.creation.box(extents=[6,.2,4]);thin.apply_translation([11,4,4])
        trimesh.util.concatenate([main,thin]).export(path)
        cfg=validate_case({'stl':'part.stl','pitch':2},self.folder)
        n,c=box_mesh((8,8,8),2);r=solve_solid(n,c)
        checks=diagnostics(cfg,r,None)
        self.assertEqual(next(x for x in checks if x['code']=='GEOMETRY_APPROXIMATION')['status'],'warn')


if __name__=='__main__':unittest.main(verbosity=2)
