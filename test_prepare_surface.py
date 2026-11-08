import unittest
import trimesh
from prepare_surface import union_solids


class UnionTests(unittest.TestCase):
    def test_overlap_preserves_union_and_bounds(self):
        a=trimesh.creation.box(extents=[2,2,2]);b=a.copy();b.apply_translation([1,0,0])
        result,audit=union_solids(trimesh.util.concatenate([a,b]))
        self.assertAlmostEqual(result.volume,12)
        self.assertAlmostEqual(audit['overlap_volume_removed_mm3'],4)
        self.assertEqual(audit['output_components'],1)

    def test_separate_components_are_not_glued(self):
        a=trimesh.creation.box();b=a.copy();b.apply_translation([3,0,0])
        with self.assertRaisesRegex(ValueError,'disconnected'):union_solids(trimesh.util.concatenate([a,b]))

    def test_cavity_and_open_surface_rejected(self):
        a=trimesh.creation.box(extents=[3,3,3]);b=trimesh.creation.box();b.invert()
        with self.assertRaises(ValueError):union_solids(trimesh.util.concatenate([a,b]))
        a.update_faces(range(len(a.faces)-1))
        with self.assertRaises(ValueError):union_solids(a)


if __name__=='__main__':unittest.main()
