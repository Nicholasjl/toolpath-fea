import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from solid import stl_mesh


class VoxelBoundsTests(unittest.TestCase):
    def test_partial_layer_fits_translated_thin_plate(self):
        surface=trimesh.creation.box(extents=[4.1,3.2,1.28])
        surface.apply_translation([11,-7,2])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'plate.stl';surface.export(path,file_type='stl_ascii')
            for pitch in [.5,.5/1.5]:
                nodes,cells=stl_mesh(path,pitch)
                np.testing.assert_allclose([nodes.min(axis=0),nodes.max(axis=0)],surface.bounds,atol=1e-12)
                t=nodes[cells]
                volume=np.abs(np.linalg.det(t[:,1:]-t[:,:1])).sum()/6
                self.assertAlmostEqual(volume,surface.volume,places=9)


if __name__=='__main__':unittest.main()
