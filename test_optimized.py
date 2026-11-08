import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from solid import box_mesh,solve_solid,transform
from solver import map_angles
from numerics import strain_transform,nearest_segments
from agent_worker import write_npz
from meshing import tetra_mesh


class OptimizedTests(unittest.TestCase):
    def test_batch_rotation_matches_reference(self):
        angles=np.linspace(-3,3,23)
        for axis in 'XYZ':np.testing.assert_allclose(strain_transform(angles,axis),np.array([transform(a,axis) for a in angles]),atol=1e-14)

    def test_spatial_search_exact(self):
        rng=np.random.default_rng(7)
        segments=rng.uniform(-10,10,(100,5));points=rng.uniform(-12,12,(300,2))
        a,d=map_angles(points,segments);b,e=nearest_segments(points,segments)
        np.testing.assert_allclose(d,e,atol=1e-12);np.testing.assert_allclose(a,b,atol=1e-12)

    def test_direct_amg_and_chunk_invariance(self):
        n,c=box_mesh((16,8,4),1)
        direct=solve_solid(n,c,solver='direct',chunk_size=128,angle=37)
        iterative=solve_solid(n,c,solver='amg',chunk_size=4096,angle=37,rtol=1e-11)
        np.testing.assert_allclose(iterative['displacement'],direct['displacement'],rtol=1e-6,atol=1e-9)
        np.testing.assert_allclose(iterative['stress'],direct['stress'],rtol=1e-5,atol=1e-7)

    def test_tetra_thin_body_and_cavity(self):
        import trimesh
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'thin.stl'
            trimesh.creation.box(extents=[12,.65,5]).export(path)
            n,c,info=tetra_mesh(path,2)
            r=solve_solid(n,c,solver='direct')
            self.assertAlmostEqual(r['volume'],12*.65*5,places=5)
            self.assertLess(info['volume_relative_error'],1e-6)
            outer=trimesh.creation.box(extents=[10,10,10]);inner=trimesh.creation.box(extents=[4,4,4]);inner.invert()
            trimesh.util.concatenate([outer,inner]).export(path)
            n,c,info=tetra_mesh(path,2)
            self.assertEqual(info['cavities'],1)
            self.assertLess(info['volume_relative_error'],1e-6)

    def test_metadata_npz_without_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'r.npz';write_npz(path,{'nodes':np.ones((2,3)),'solver_info':{'method':'amg'}})
            data=dict(np.load(path,allow_pickle=False))
            self.assertEqual(json.loads(str(data['solver_info']))['method'],'amg')


if __name__=='__main__':unittest.main(verbosity=2)
