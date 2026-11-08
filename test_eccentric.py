import unittest
import numpy as np
from solid import surface_load, box_mesh, solve_solid
from agent_worker import validate_case, JobError

class EccentricTests(unittest.TestCase):
    def test_resultant_and_translation(self):
        n=np.array([[0.,0,5],[4,0,5],[4,6,5],[0,6,5]])
        faces=np.array([[0,1,2],[0,2,3]])
        p=np.array([2,3,15]);force=np.array([0,-1.,0])
        f=surface_load(n,faces,force,p)
        np.testing.assert_allclose(f.sum(axis=0),force,atol=1e-14)
        np.testing.assert_allclose(np.cross(n-p,f).sum(axis=0),0,atol=1e-13)
        np.testing.assert_allclose(np.cross(n-[2,3,5],f).sum(axis=0),[10,0,0],atol=1e-13)
        np.testing.assert_allclose(surface_load(n+123,faces,force,p+123),f,atol=1e-13)
        np.testing.assert_allclose(surface_load(n,faces,force,[2,3,5]),surface_load(n,faces,force),atol=1e-14)

    def test_reactions_and_linear_scaling(self):
        n,c=box_mesh((8,4,2),1)
        a=solve_solid(n,c,force=(0,-1,0),load_point=[8,2,12],solver='direct')
        b=solve_solid(n,c,force=(0,-2,0),load_point=[8,2,12],solver='direct')
        np.testing.assert_allclose(a['reactions'].sum(axis=0),[0,1,0],atol=1e-9)
        moment=np.cross(n,a['forces']).sum(axis=0)+np.cross(n[a['fixed_nodes']],a['reactions']).sum(axis=0)
        np.testing.assert_allclose(moment,0,atol=1e-8)
        np.testing.assert_allclose(b['displacement'],2*a['displacement'],atol=1e-12)

    def test_invalid_point(self):
        for p in ([1,2], [1,float('nan'),2], [True,1,2], None):
            with self.assertRaises(JobError):validate_case({'load_point':p},'.')

if __name__=='__main__':unittest.main()
