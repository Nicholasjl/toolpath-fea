import unittest
import numpy as np
from solid import box_mesh,boundary_faces,select_faces,solve_solid
from agent_worker import validate_case,JobError,diagnostics


class RegionUnionTests(unittest.TestCase):
    def test_four_mounts_keep_free_center_and_balance_remote_load(self):
        nodes,cells=box_mesh((10,10,2),1);faces,_=boundary_faces(cells)
        boxes=[[[x,y,-.01],[x+2,y+2,.01]] for x,y in [(0,0),(8,0),(0,8),(8,8)]]
        fixed={'boxes':boxes};mask=select_faces(nodes,faces,fixed)
        self.assertEqual(mask.sum(),32)
        center=nodes[faces[mask]].mean(axis=1)
        self.assertFalse(np.any(np.all((center[:,:2]>3)&(center[:,:2]<7),axis=1)))
        cfg=validate_case(dict(fixed=fixed,loaded=[[3,3,1.99],[7,7,2.01]],force=[0,-1,0],load_point=[5,5,12]),'.')
        result=solve_solid(nodes,cells,fixed=cfg['fixed'],loaded=cfg['loaded'],force=cfg['force'],load_point=cfg['load_point'])
        np.testing.assert_allclose(result['reactions'].sum(axis=0),[0,1,0],atol=1e-8)
        moment=np.cross(nodes,result['forces']).sum(axis=0)+np.cross(nodes[result['fixed_nodes']],result['reactions']).sum(axis=0)
        np.testing.assert_allclose(moment,0,atol=1e-7)
        checks=diagnostics(cfg,result,None)
        self.assertEqual(len([c for c in checks if c['code'].startswith('REGION_FIXED_')]),4)

    def test_missing_mount_is_rejected(self):
        nodes,cells=box_mesh((4,4,2),1);faces,_=boundary_faces(cells)
        with self.assertRaisesRegex(ValueError,'子区域'):
            select_faces(nodes,faces,{'boxes':[[[0,0,-1],[1,1,0]],[[9,9,9],[10,10,10]]]})

    def test_invalid_union(self):
        for value in [{'boxes':[]},{'boxes':[[[0,0,0],[1,1,float('nan')]]]},{'boxes':[[[1,0,0],[0,1,1]]]},{'boxes':['X-']},{'boxes':[[[False,0,0],[1,1,1]]]}]:
            with self.assertRaises(JobError):validate_case({'fixed':value},'.')


if __name__=='__main__':unittest.main()
