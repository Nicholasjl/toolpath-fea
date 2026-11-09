"""Regress the surface-only false-convergence failure against real solid bending."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import trimesh
from agent_worker import validate_case, convergence_check
from defaults import DEFAULT_MESH_MODE, DEFAULT_PITCH_MM, TETGEN_SWITCHES
from meshing import tetra_mesh, volume_resolution
from solid import Material, solve_solid


class MeshDefaultTests(unittest.TestCase):
    def test_default_stl_mode_and_legacy_opt_in(self):
        cfg=validate_case({},Path('.'))
        self.assertEqual(cfg['mesh_mode'],DEFAULT_MESH_MODE)
        self.assertEqual(cfg['mesh_mode'],'tetra')
        self.assertEqual(cfg['pitch'],DEFAULT_PITCH_MM)
        self.assertEqual(validate_case({'mesh_mode':'voxel'},Path('.'))['mesh_mode'],'voxel')
        self.assertIn('q',TETGEN_SWITCHES)
        self.assertNotIn('Y',TETGEN_SWITCHES)

    def test_actual_tetra_refinement_and_cantilever_reference(self):
        E=2000.;nu=.3;G=E/(2*(1+nu));m=Material(E,E,E,G,G,G,nu,nu,nu)
        exact=40**3/(3*E*(8*4**3/12));runs=[]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'beam.stl';mesh=trimesh.creation.box(extents=[40,8,4])
            mesh.apply_translation([20,4,2]);mesh.export(p)
            for pitch in [2,1]:
                n,c,info=tetra_mesh(p,pitch)
                self.assertEqual(volume_resolution(info)['status'],'pass')
                self.assertTrue(info['quality_refinement'] and info['boundary_refinement'])
                r=solve_solid(n,c,m,force=[0,0,-1],solver='auto');r['mesh_info']=info
                displacement=abs(r['displacement'][np.isclose(n[:,0],40),2].mean())
                runs.append((r,displacement,info))
        self.assertGreater(len(runs[1][0]['cells']),len(runs[0][0]['cells']))
        self.assertLess(runs[1][2]['volume_weighted_p95_mm3']/runs[0][2]['volume_weighted_p95_mm3'],.85)
        self.assertLess(abs(runs[1][1]/exact-1),abs(runs[0][1]/exact-1))
        self.assertLess(abs(runs[1][1]/exact-1),.1)

    def test_surface_only_change_cannot_pass_convergence(self):
        def info(p95):
            return dict(mode='tetra',max_volume_target_mm3=1,max_volume_actual_mm3=1,
                        volume_weighted_p95_mm3=p95,quality_refinement=True,boundary_refinement=True)
        base=dict(nodes=np.zeros((4,3)),cells=np.array([[0,1,2,3]]),mesh_info=info(.8))
        fine=dict(nodes=np.ones((5,3)),cells=np.array([[0,1,2,3],[1,2,3,4]]),mesh_info=info(.8))
        self.assertEqual(convergence_check(base,fine,.001,.001,[0,-1,0])['status'],'not_checked')
        fine['mesh_info']=info(.4)
        self.assertEqual(convergence_check(base,fine,.001,.001,[0,-1,0])['status'],'pass')
        fine['mesh_info']['max_volume_actual_mm3']=75
        self.assertEqual(convergence_check(base,fine,.001,.001,[0,-1,0])['status'],'not_checked')
        fine['mesh_info'].pop('volume_weighted_p95_mm3')
        self.assertEqual(convergence_check(base,fine,.001,.001,[0,-1,0])['status'],'not_checked')


if __name__=='__main__':unittest.main(verbosity=2)
