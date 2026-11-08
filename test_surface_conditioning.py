import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
from meshing import condition_surface,tetra_mesh
from agent_worker import validate_case,JobError,convergence_check

class SurfaceConditioningTests(unittest.TestCase):
    def test_thin_plate_with_cavity_preserves_topology(self):
        outer=trimesh.creation.box(extents=[8,8,.65])
        inner=trimesh.creation.box(extents=[2,2,.25]);inner.invert()
        surface=trimesh.util.concatenate([outer,inner])
        result,info=condition_surface(surface,.7)
        self.assertTrue(result.is_watertight)
        self.assertEqual(result.euler_number,surface.euler_number)
        self.assertEqual(len(result.split()),2)
        self.assertLess(info['volume_relative_change'],.001)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'thin.stl';surface.export(p,file_type='stl_ascii')
            _,_,audit=tetra_mesh(p,2,surface_remesh_mm=.7)
            self.assertEqual(audit['cavities'],1)
            self.assertLess(audit['volume_relative_error'],.002)

    def test_tetgen_failure_is_mesh_error(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'box.stl';trimesh.creation.box().export(p)
            with patch('meshpy.tet.build',side_effect=RuntimeError('TetGen runtime error code 3')):
                with self.assertRaisesRegex(ValueError,'四面体生成失败'):tetra_mesh(p,2)

    def test_identical_mesh_is_not_convergence(self):
        base=dict(nodes=np.zeros((4,3)),cells=np.array([[0,1,2,3]]))
        self.assertEqual(convergence_check(base,base,0,0,[0,-1,0])['status'],'not_checked')
        fine=dict(nodes=np.ones((5,3)),cells=np.array([[0,1,2,3],[1,2,3,4]]))
        self.assertEqual(convergence_check(base,fine,.01,.01,[0,-1,0])['status'],'pass')

    def test_intersecting_closed_surfaces_are_rejected(self):
        a=trimesh.creation.box(extents=[2,2,2]);b=a.copy();b.apply_translation([1,.5,.5])
        surface=trimesh.util.concatenate([a,b])
        self.assertTrue(surface.is_watertight)
        with self.assertRaisesRegex(ValueError,'相交面'):condition_surface(surface,.7)

    def test_bounded_fallback_records_attempts(self):
        import meshpy.tet
        original_build=meshpy.tet.build
        calls=[]
        def fail_once(*args,**kwargs):
            calls.append(1)
            if len(calls)==1:raise RuntimeError('TetGen runtime error code 3')
            return original_build(*args,**kwargs)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'box.stl';trimesh.creation.box(extents=[8,8,2]).export(p)
            with patch('meshpy.tet.build',side_effect=fail_once):
                _,_,info=tetra_mesh(p,2,surface_remesh_mm=.7)
            self.assertEqual([x['status'] for x in info['tetgen_attempts']],['rejected','completed'])
            self.assertTrue(info['surface_conditioning']['smoothing'])
            self.assertEqual(info['surface_conditioning']['output_intersecting_faces'],0)

    def test_failed_fallback_audit_never_reaches_tetgen(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'box.stl';m=trimesh.creation.box();m.export(p)
            with patch('meshing.condition_surface',side_effect=[(m,{}),ValueError('相交面')]):
                with patch('meshpy.tet.build',side_effect=RuntimeError('TetGen runtime error code 3')) as build:
                    with self.assertRaisesRegex(ValueError,'相交面'):tetra_mesh(p,2,surface_remesh_mm=.7)
                    self.assertEqual(build.call_count,1)

    def test_s1b1h_pad_sliver_regression(self):
        data=np.load(Path(__file__).parent/'testdata/s1b1h_surface.npz',allow_pickle=False)
        source=trimesh.Trimesh(data['vertices'],data['faces'],process=False)
        result,info=condition_surface(source,.7)
        self.assertFalse(info['smoothing'])
        self.assertEqual(info['input_intersecting_faces'],0)
        self.assertEqual(info['output_intersecting_faces'],0)
        self.assertTrue(result.is_watertight)
        self.assertEqual(result.euler_number,source.euler_number)
        self.assertLess(info['volume_relative_change'],.001)

    def test_invalid_conditioning_settings(self):
        for value in [-1,6,True,float('nan')]:
            with self.assertRaises(JobError):validate_case({'surface_remesh_mm':value},Path('.'))
        with self.assertRaises(JobError):validate_case({'surface_remesh_mm':.7,'mesh_mode':'voxel'},Path('.'))

if __name__=='__main__':unittest.main(verbosity=2)
