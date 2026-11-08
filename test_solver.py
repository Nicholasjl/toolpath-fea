import tempfile
import unittest
from pathlib import Path
import numpy as np
from solver import Settings, constitutive, solve, read_gcode, map_angles


class SolverTests(unittest.TestCase):
    def test_isotropic_rotation(self):
        e,nu=2100,.3
        baseline=constitutive(e,e,e/(2*(1+nu)),nu,0)[0]
        for a in np.linspace(0,np.pi,13):
            np.testing.assert_allclose(constitutive(e,e,e/(2*(1+nu)),nu,a)[0],baseline,atol=1e-10)

    def test_exact_uniform_tension(self):
        # nu=0 makes the exact uniaxial solution compatible with full left clamp.
        c=Settings(nx=8,ny=4,nu12=0,fx=100,fy=0)
        r=solve(c)
        exact=100*r['nodes'][:,0]/(c.height*c.thickness*c.e1)
        np.testing.assert_allclose(r['displacement'][:,0],exact,atol=1e-12)
        np.testing.assert_allclose(r['reaction'],[-100,0],atol=1e-9)
        self.assertLess(r['relative_residual'],1e-10)
        c.angle=90
        r90=solve(c)
        np.testing.assert_allclose(r90['displacement'][:,0],exact*c.e1/c.e2,atol=1e-12)

    def test_cantilever_convergence(self):
        # Slender isotropic beam: Euler-Bernoulli tip displacement.
        c=Settings(length=100,height=10,thickness=2,e1=2400,e2=2400,g12=2400/2.6,nu12=.3,fy=-1,nx=20,ny=4)
        exact=abs(c.fy)*c.length**3/(3*c.e1*c.thickness*c.height**3/12)
        coarse=abs(solve(c)['displacement'][-1,1])
        c.nx,c.ny=100,20
        fine=abs(solve(c)['displacement'][-1,1])
        self.assertLess(abs(fine-exact),abs(coarse-exact))
        self.assertLess(abs(fine-exact)/exact,.03)

    def test_segment_mapping(self):
        segments=np.array([[0,0,100,0,0],[50,10,50,20,0]])
        a,d=map_angles(np.array([[1,1],[50,15]]),segments)
        np.testing.assert_allclose(a,[0,np.pi/2])
        np.testing.assert_allclose(d,[1,0])

    def test_gcode_modes(self):
        content='G21\nG90\nM82\nG1 Z0.2\nG1 X10 E1\nG1 E0.5\nG92 E0\nM83\nG1 Y10 E1\nG1 X0 E-1\nG91\nG1 X2 E1\n'
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'test.gcode'; p.write_text(content)
            segments=read_gcode(p)
            self.assertEqual(len(segments),3)
            np.testing.assert_allclose(segments[-1],[0,10,2,10,.2])
            p.write_text('G2 X10 Y10 E2')
            with self.assertRaises(ValueError): read_gcode(p)

    def test_invalid_material(self):
        with self.assertRaises(ValueError): solve(Settings(e1=1,e2=100,nu12=.5))


if __name__=='__main__': unittest.main(verbosity=2)
