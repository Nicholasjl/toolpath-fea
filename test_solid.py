import tempfile
import unittest
from pathlib import Path
import numpy as np
from solid import *


class SolidTests(unittest.TestCase):
    def test_rotation_isotropy(self):
        e,nu=2000,.3;g=e/(2*(1+nu))
        m=Material(e,e,e,g,g,g,nu,nu,nu);D=local_stiffness(m)
        for axis in 'XYZ':
            for a in [0,.3,1.2]:
                T=transform(a,axis);np.testing.assert_allclose(T.T@D@T,D,atol=1e-10)

    def test_uniform_tension(self):
        nodes,cells=box_mesh((20,8,6),2)
        m=Material(nu12=0,nu23=0,nu13=0)
        r=solve_solid(nodes,cells,m,force=(100,0,0))
        exact=100*nodes[:,0]/(8*6*m.e1)
        np.testing.assert_allclose(r['displacement'][:,0],exact,atol=1e-11)
        np.testing.assert_allclose(r['reactions'].sum(axis=0),[-100,0,0],atol=1e-9)
        self.assertAlmostEqual(r['volume'],20*8*6)
        r90=solve_solid(nodes,cells,m,force=(100,0,0),angle=90)
        np.testing.assert_allclose(r90['displacement'][:,0],exact*m.e1/m.e2,atol=1e-11)

    def test_bending_convergence_and_moment(self):
        e=2000;nu=.3;g=e/(2*(1+nu));m=Material(e,e,e,g,g,g,nu,nu,nu)
        tip=[]
        for pitch in (1,.5):
            nodes,cells=box_mesh((40,8,4),pitch)
            r=solve_solid(nodes,cells,m,force=(0,0,-1))
            tip.append(abs(r['displacement'][np.isclose(nodes[:,0],40),2].mean()))
        exact=40**3/(3*e*8*4**3/12)
        self.assertLess(abs(tip[1]-exact),abs(tip[0]-exact))
        self.assertLess(abs(tip[1]-exact)/exact,.12)
        moment=np.cross(nodes,r['forces']).sum(axis=0)+np.cross(nodes[r['fixed_nodes']],r['reactions']).sum(axis=0)
        np.testing.assert_allclose(moment,0,atol=1e-8)

    def test_path_layers(self):
        centers=np.array([[2,1,.2],[2,1,.4]])
        segments=np.array([[0,0,10,0,.2],[0,0,0,10,.4]])
        a,d=path_field(centers,segments,(0,0,0))
        np.testing.assert_allclose(a,[0,np.pi/2]);np.testing.assert_allclose(d,[1,2])
        a2,_=path_field(centers+[5,6,7],segments,(5,6,7));np.testing.assert_allclose(a,a2)

    def test_disconnected_rejected(self):
        n,c=voxel_mesh([[0,0,0],[3,0,0]],1,np.zeros(3))
        with self.assertRaisesRegex(ValueError,'不连通'):solve_solid(n,c)

    def test_stl_and_vtk(self):
        import trimesh
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'box.stl';mesh=trimesh.creation.box(extents=[20,8,6]);mesh.export(path)
            n,c=stl_mesh(path,2);r=solve_solid(n,c)
            self.assertAlmostEqual(r['volume'],960)
            out=Path(directory)/'result.vtk';write_vtk(out,r)
            self.assertIn('CELL_TYPES',out.read_text())

    def test_region_overlap(self):
        n,c=box_mesh()
        with self.assertRaisesRegex(ValueError,'共享节点'):solve_solid(n,c,fixed='X-',loaded='X-')

    def test_perforated_stl(self):
        import trimesh
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ring.stl'
            trimesh.creation.annulus(r_min=4,r_max=10,height=6,sections=48).export(path)
            n,c=stl_mesh(path,2)
            centers=n[c].mean(axis=1)
            self.assertGreater(np.linalg.norm(centers[:,:2],axis=1).min(),2)
            r=solve_solid(n,c,force=(1,0,0))
            np.testing.assert_allclose(r['reactions'].sum(axis=0),[-1,0,0],atol=1e-9)

    def test_gcode_drives_solid_stiffness(self):
        n,c=box_mesh((12,4,4),2)
        segments=np.array([[0,0,0,10,1],[0,0,0,10,3]])
        mapped=solve_solid(n,c,segments=segments,force=(1,0,0))
        reference=solve_solid(n,c,angle=90,force=(1,0,0))
        np.testing.assert_allclose(mapped['displacement'],reference['displacement'],atol=1e-12)


if __name__=='__main__':unittest.main(verbosity=2)
