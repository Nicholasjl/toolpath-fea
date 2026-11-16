"""Exercise robust meshing, JSON metadata export and cache reuse end-to-end."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import trimesh


@unittest.skipUnless(importlib.util.find_spec('pytetwild') and importlib.util.find_spec('pyvista'),'optional robust dependencies missing')
class RobustCliTests(unittest.TestCase):
    def test_fast_backend_export_and_load_independent_cache(self):
        repo=Path(__file__).parent
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);mesh=trimesh.creation.box(extents=[8,4,2]);mesh.apply_translation([4,2,1]);mesh.export(root/'part.stl')
            cfg=dict(stl='part.stl',pitch=1,tetra_backend='tetwild',tetwild_mode='fast',mesh_cache_dir='cache',
                     fixed=[[-.02,-.02,-.02],[.02,4.02,2.02]],loaded=[[7.98,-.02,-.02],[8.02,4.02,2.02]],force=[0,0,-1])
            values=[]
            for amplitude in (1,2):
                cfg['force']=[0,0,-amplitude];case=root/'case.json';case.write_text(json.dumps(cfg))
                result=subprocess.run([sys.executable,str(repo/'run3d.py'),str(case),'--output',str(root/f'run{amplitude}'),'--timeout','60','--no-convergence'],capture_output=True,text=True,encoding='utf-8',timeout=75)
                data=json.loads(result.stdout);self.assertEqual(data['status'],'completed',data.get('error'))
                values.append(data['analysis']['metrics']['max_displacement_mm'])
                files=list((root/'cache').glob('*.npz'));self.assertEqual(len(files),1)
                if amplitude==1:stamp=files[0].stat().st_mtime_ns
                else:self.assertEqual(files[0].stat().st_mtime_ns,stamp)
            self.assertAlmostEqual(values[1]/values[0],2,places=8)
