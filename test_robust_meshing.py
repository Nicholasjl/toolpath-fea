import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import trimesh
from agent_worker import validate_case,JobError
from robust_meshing import tetra_mesh_robust,surface_audit
from solid import box_mesh,solve_solid,Material
from meshing import volume_resolution


class RobustSettingsTests(unittest.TestCase):
    def test_opt_in_and_invalid_settings(self):
        self.assertEqual(validate_case({},'.')['tetra_backend'],'tetgen')
        for cfg in [{'tetra_backend':'bad'},{'tetra_backend':'tetwild'},{'surface_tolerance_mm':0},{'surface_tolerance_mm':True},{'surface_tolerance_mm':.02}]:
            with self.assertRaises(JobError):validate_case(cfg,'.')

    def test_surface_audit_rejects_filled_through_hole(self):
        source=trimesh.creation.annulus(r_min=1,r_max=3,height=1)
        n,c=box_mesh((6,6,1),1);n-=np.array([3,3,.5])
        with self.assertRaisesRegex(ValueError,'拓扑'):surface_audit(source,n,c,.005,256)


@unittest.skipUnless(importlib.util.find_spec('wildmeshing'),'optional requirements-robust.txt not installed')
class RobustMeshingTests(unittest.TestCase):
    def test_legacy_backend_audits_cavity(self):
        outer=trimesh.creation.box(extents=[8,8,1]);inner=trimesh.creation.box(extents=[2,2,.4]);inner.invert()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'cavity.stl';trimesh.util.concatenate([outer,inner]).export(p)
            n,c,info=tetra_mesh_robust(p,2,mode='legacy')
        self.assertEqual(info['cavities'],1)
        self.assertEqual(info['surface_conditioning']['envelope_method'],'sampled')
        self.assertTrue(info['surface_conditioning']['topology_preserved'])
        self.assertLessEqual(info['backend_report']['max_energy'],100.001)

    def test_thin_cavity_topology_and_volume(self):
        outer=trimesh.creation.box(extents=[8,8,1]);inner=trimesh.creation.box(extents=[2,2,.4]);inner.invert()
        source=trimesh.util.concatenate([outer,inner])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'cavity.stl';source.export(p,file_type='stl_ascii')
            n,c,info=tetra_mesh_robust(p,2)
        self.assertEqual(info['cavities'],1)
        self.assertLess(info['volume_relative_error'],.002)
        self.assertLess(max(info['surface_conditioning']['bidirectional_sample_max_mm']),.0051)
        self.assertTrue(info['quality_refinement'] and info['boundary_refinement'])
        self.assertTrue(np.isfinite(n).all())

    def test_open_surface_not_silently_repaired(self):
        m=trimesh.creation.box();m.update_faces(np.arange(len(m.faces))!=0)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'open.stl';m.export(p)
            with self.assertRaisesRegex(ValueError,'闭合'):tetra_mesh_robust(p,2)

    def test_refinement_improves_cantilever_bending(self):
        self.check_cantilever_bending('modern')

    @unittest.skipUnless(importlib.util.find_spec('pytetwild') and importlib.util.find_spec('pyvista'),'optional fast backend missing')
    def test_fast_refinement_improves_cantilever_bending(self):
        self.check_cantilever_bending('fast')

    def check_cantilever_bending(self,mode):
        E=2000.;nu=.3;G=E/(2*(1+nu));material=Material(E,E,E,G,G,G,nu,nu,nu)
        exact=24**3/(3*E*(6*3**3/12));runs=[]
        source=trimesh.creation.box(extents=[24,6,3]);source.apply_translation([12,3,1.5])
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'beam.stl';source.export(p)
            for pitch in [2,.7]:
                n,c,info=tetra_mesh_robust(p,pitch,mode=mode)
                result=solve_solid(n,c,material,fixed=[[-.02,-.02,-.02],[.02,6.02,3.02]],loaded=[[23.98,-.02,-.02],[24.02,6.02,3.02]],force=[0,0,-1],solver='direct')
                # Work-conjugate displacement includes area weighting on the load patch.
                compliance=2*result['energy']
                runs.append((compliance,info))
        self.assertLess(abs(runs[1][0]/exact-1),.15)
        self.assertLess(abs(runs[1][0]/exact-1),abs(runs[0][0]/exact-1))
        self.assertLess(runs[1][1]['volume_weighted_p95_mm3']/runs[0][1]['volume_weighted_p95_mm3'],.85)
        self.assertEqual(volume_resolution(runs[1][1])['status'],'pass')


if __name__=='__main__':unittest.main()
