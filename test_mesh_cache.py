import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from mesh_cache import cached_mesh


class MeshCacheTests(unittest.TestCase):
    def test_reuse_invalidation_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            case={'stl':'part.stl','mesh_mode':'tetra','mesh_cache_dir':folder}
            calls=[]
            def generate():
                calls.append(1)
                return np.eye(4,3),np.array([[0,1,2,3]],dtype=np.int32),{'volume':1.0}
            with patch('mesh_cache.signature',return_value='one'):
                a=cached_mesh(case,generate);b=cached_mesh(case,generate)
                self.assertEqual(len(calls),1);np.testing.assert_array_equal(a[0],b[0])
                file=Path(folder)/'one.npz'
                with np.load(file) as data:contents=dict(data)
                contents['nodes'][0,0]=100
                np.savez(file,**contents)
                cached_mesh(case,generate);self.assertEqual(len(calls),2)
            with patch('mesh_cache.signature',return_value='two'):
                cached_mesh(case,generate);self.assertEqual(len(calls),3)

    def test_disabled_cache(self):
        calls=[]
        def generate():calls.append(1);return None
        cached_mesh({},generate);cached_mesh({},generate)
        self.assertEqual(len(calls),2)
