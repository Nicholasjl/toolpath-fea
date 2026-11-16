import tempfile,unittest,json,subprocess,sys
from pathlib import Path
import numpy as np,trimesh
from solid import box_mesh,solve_solid,Material
from refinement import refine_tetra,measures
from meshing import tetra_mesh


class RefinementTests(unittest.TestCase):
    def test_cli_uniform_refinement_checks_both_meshes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);p=folder/'beam.stl';trimesh.creation.box(extents=[12,4,2]).export(p)
            case=folder/'case.json';case.write_text(json.dumps({'stl':str(p),'pitch':2,'refinement_strategy':'uniform','solver':'direct'}))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('run3d.py')),str(case),'--output',str(folder/'job'),'--timeout','60'],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stderr)
            a=json.loads(result.stdout)['analysis'];checks={x['code']:x for x in a['checks']}
            self.assertEqual(checks['REFINED_EQUILIBRIUM']['status'],'pass')
            self.assertEqual(a['runs'][1]['metrics']['elements'],8*a['runs'][0]['metrics']['elements'])
            self.assertAlmostEqual(checks['MESH_CONVERGENCE']['metrics']['volume_weighted_p95_refinement_ratio'],.125,places=8)
    def test_conforming_volume_and_budget(self):
        n,c=box_mesh((4,3,2),1);v,_=measures(n,c)
        nn,cc,info=refine_tetra(n,c,{},.5,10000)
        vv,_=measures(nn,cc)
        self.assertEqual(len(cc),8*len(c))
        np.testing.assert_allclose(vv.reshape(-1,8),np.repeat(v[:,None]/8,8,axis=1),rtol=1e-12)
        self.assertEqual(info['refinement']['euler_number'],2)
        with self.assertRaisesRegex(ValueError,'超出限制'):refine_tetra(n,c,{},.5,len(c))

    def test_cavity_is_preserved(self):
        outer=trimesh.creation.box(extents=[6,6,3]);inner=trimesh.creation.box(extents=[2,2,1]);inner.invert()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'cavity.stl';trimesh.util.concatenate([outer,inner]).export(p)
            n,c,info=tetra_mesh(p,2)
            nn,cc,fine=refine_tetra(n,c,info,1,100000)
        self.assertEqual(fine['cavities'],1)
        self.assertAlmostEqual(measures(nn,cc)[0].sum(),104,places=9)
        self.assertLess(fine['volume_weighted_p95_mm3']/info['volume_weighted_p95_mm3'],.13)

    def test_bending_refinement_improves_reference(self):
        E=2000.;nu=.3;G=E/(2*(1+nu));m=Material(E,E,E,G,G,G,nu,nu,nu)
        n,c=box_mesh((24,6,3),1);nn,cc,info=refine_tetra(n,c,{},.5,100000)
        values=[]
        for nodes,cells in [(n,c),(nn,cc)]:
            r=solve_solid(nodes,cells,m,force=[0,0,-1],solver='direct')
            values.append(abs(r['displacement'][np.isclose(nodes[:,0],24),2].mean()))
        exact=24**3/(3*E*(6*3**3/12))
        self.assertLess(abs(values[1]/exact-1),abs(values[0]/exact-1))
        self.assertLess(abs(values[1]/exact-1),.15)
