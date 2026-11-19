import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse.linalg import spsolve

from gpu_hex_solver import (
    assemble_coarse_csr,
    build_level_topology,
    extract_boundary_quads,
    quad_surface_load,
    recover_stress_streaming,
    select_quad_faces,
    solve_hex_gpu,
)
from hex_fem import (
    angle_to_bin,
    bin_to_angle,
    build_stiffness_lut,
    hex_B_matrices,
    hex_centroid_stress,
    hex_element_stiffness,
)
from solid import Material, local_stiffness
from toolpath_bricks import (
    FEATURE_OUTER_WALL,
    FEATURE_SOLID_INFILL,
    FEATURE_SPARSE_INFILL,
    FLAG_BEAD_OVERLAP,
    box_to_bricks,
    decode_feature,
    gcode_to_bricks,
    load_bricks,
    mask_to_bricks,
    save_bricks,
    unpack_payload,
)


class ToolpathGpuTests(unittest.TestCase):
    def test_hex_reference_element_properties(self):
        dx = dy = dz = 0.2
        m_ortho = Material()
        Ke0 = hex_element_stiffness(m_ortho, dx, dy, dz, 0.0)
        np.testing.assert_allclose(Ke0, Ke0.T, atol=1e-12)

        # Six rigid-body modes (3 translations + 3 infinitesimal rotations) must have zero energy.
        corners = np.array([[(a & 1) * dx, ((a >> 1) & 1) * dy, ((a >> 2) & 1) * dz] for a in range(8)])
        modes = np.zeros((24, 6), dtype=np.float64)
        for d in range(3):
            modes[d::3, d] = 1.0
        p = corners - corners.mean(axis=0)
        for axis in range(3):
            rot = np.cross(np.eye(3)[axis], p)
            modes[:, 3 + axis] = rot.ravel()
        energy_scale = np.max(np.abs(Ke0))
        np.testing.assert_allclose(Ke0 @ modes, 0.0, atol=1e-12 * energy_scale)

        # Pi-periodicity of rotated orthotropic H8 stiffness
        for deg in (0.0, 23.5, 45.0, 90.0, 135.0):
            r1 = math.radians(deg)
            r2 = math.radians(deg + 180.0)
            k1 = hex_element_stiffness(m_ortho, dx, dy, dz, r1)
            k2 = hex_element_stiffness(m_ortho, dx, dy, dz, r2)
            np.testing.assert_allclose(k1, k2, atol=1e-11 * energy_scale)

        # Isotropic rotation invariance
        e, nu = 2000.0, 0.3
        g = e / (2.0 * (1.0 + nu))
        m_iso = Material(e, e, e, g, g, g, nu, nu, nu)
        ke_iso_0 = hex_element_stiffness(m_iso, dx, dy, dz, 0.0)
        for rad in (0.3, 1.1, 2.4):
            ke_iso_r = hex_element_stiffness(m_iso, dx, dy, dz, rad)
            np.testing.assert_allclose(ke_iso_r, ke_iso_0, atol=1e-10 * np.max(np.abs(ke_iso_iso := ke_iso_0)))

        # LUT exactness at 0, 45, 90, 135 deg
        ke_lut64, ke_lut32, _ = build_stiffness_lut(m_ortho, dx, dy, dz, n_bins=360)
        self.assertEqual(ke_lut32.dtype, np.float32)
        for deg in (0.0, 45.0, 90.0, 135.0):
            b = int(angle_to_bin(math.radians(deg), 360))
            np.testing.assert_allclose(ke_lut64[b], hex_element_stiffness(m_ortho, dx, dy, dz, math.radians(deg)), atol=1e-11)

    def test_uniform_tension_0_and_90_at_02mm(self):
        m = Material(nu12=0.0, nu23=0.0, nu13=0.0)
        g0 = box_to_bricks((8.0, 2.4, 1.6), pitch=0.2, angle_deg=0.0)
        r0 = solve_hex_gpu(g0, material=m, fixed="X-", loaded="X+", force=(100.0, 0.0, 0.0), rtol=1e-8)
        x_coords = g0.origin[0] + r0["node_ijk"][:, 0] * g0.pitch[0]
        exact0 = 100.0 * x_coords / (2.4 * 1.6 * m.e1)
        np.testing.assert_allclose(r0["displacement"][:, 0], exact0, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(r0["reactions"].sum(axis=0), [-100.0, 0.0, 0.0], atol=1e-7)
        self.assertLess(r0["solver_info"]["relative_residual"], 1e-6)

        g90 = box_to_bricks((8.0, 2.4, 1.6), pitch=0.2, angle_deg=90.0)
        r90 = solve_hex_gpu(g90, material=m, fixed="X-", loaded="X+", force=(100.0, 0.0, 0.0), rtol=1e-8)
        exact90 = 100.0 * x_coords / (2.4 * 1.6 * m.e2)
        np.testing.assert_allclose(r90["displacement"][:, 0], exact90, rtol=1e-6, atol=1e-7)

    def test_cantilever_bending_convergence_05_to_02(self):
        e, nu = 2000.0, 0.3
        g = e / (2.0 * (1.0 + nu))
        m = Material(e, e, e, g, g, g, nu, nu, nu)
        exact = 40.0 ** 3 / (3.0 * e * 8.0 * 4.0 ** 3 / 12.0)
        tips = []
        for pitch in (0.5, 0.2):
            grid = box_to_bricks((40.0, 8.0, 4.0), pitch=pitch)
            res = solve_hex_gpu(grid, material=m, fixed="X-", loaded="X+", force=(0.0, 0.0, -1.0), rtol=1e-7)
            x_coords = grid.origin[0] + res["node_ijk"][:, 0] * grid.pitch[0]
            tip = float(abs(res["displacement"][np.isclose(x_coords, 40.0), 2].mean()))
            tips.append(tip)
            self.assertLess(res["solver_info"]["relative_residual"], 1e-6)
            self.assertLess(res["solver_info"]["moment_relative_error"], 1e-6)
        self.assertLess(abs(tips[1] - exact), abs(tips[0] - exact))
        self.assertLess(abs(tips[1] - exact) / exact, 0.05)

    def test_streaming_gcode_rasterizer_and_tbricks_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            gcode_path = tmp_dir / "synthetic.gcode"
            bricks_path = tmp_dir / "out.tbricks"
            # Construct synthetic G-code:
            # - Skirt & Custom (must be skipped)
            # - Z-hop travel G1 Z0.6 / G1 Z0.2 (must NOT trigger non-planar error)
            # - Layer 1 (Z=0.2): horizontal outer wall, 45 deg sparse infill, overlapping pass
            # - Layer 2 (Z=0.4): 90 deg vertical infill + disconnected small island far away
            gcode_text = "\n".join(
                [
                    "G21",
                    "G90",
                    "M83",
                    "; LAYER_HEIGHT: 0.2",
                    "; FEATURE: Custom",
                    "G1 X0.0 Y0.0 Z0.2 F1200",
                    "G1 X5.0 Y0.0 E0.2",
                    "; FEATURE: Skirt",
                    "G0 X-5.0 Y-5.0",
                    "G1 X15.0 Y-5.0 E0.5",
                    "; FEATURE: Outer wall",
                    "; LINE_WIDTH: 0.4",
                    "G0 X1.0 Y1.0 Z0.2",
                    "G1 Z0.6 F3000",  # travel Z-hop
                    "G1 Z0.2 F3000",
                    "G1 X9.0 Y1.0 E0.3",
                    "G1 X9.0 Y5.0 E0.15",
                    "G1 X1.0 Y5.0 E0.3",
                    "G1 X1.0 Y1.0 E0.15",
                    "; FEATURE: Sparse infill",
                    "; LINE_WIDTH: 0.4",
                    "G0 X1.0 Y1.0",
                    "G1 X9.0 Y5.0 E0.35",  # diagonal inside box
                    "G0 X1.0 Y3.0",
                    "G1 X9.0 Y3.0 E0.3",   # horizontal bead overlapping diagonal at center
                    "; CHANGE_LAYER",
                    "G1 Z0.8 F3000",
                    "G1 Z0.4 F3000",
                    "; FEATURE: Internal solid infill",
                    "; LINE_WIDTH: 0.4",
                    "G0 X1.0 Y1.0",
                    "G1 X9.0 Y1.0 E0.3",
                    "G1 X9.0 Y5.0 E0.15",
                    "G1 X1.0 Y5.0 E0.3",
                    "G1 X1.0 Y1.0 E0.15",
                    "G0 X5.0 Y1.0",
                    "G1 X5.0 Y5.0 E0.15",  # 90 deg vertical bead
                    # Disconnected floating island at X=30..32 (should be dropped!)
                    "G0 X30.0 Y30.0",
                    "G1 X32.0 Y30.0 E0.1",
                ]
            )
            gcode_path.write_text(gcode_text, encoding="utf-8")

            grid = gcode_to_bricks(gcode_path, out_bricks_path=bricks_path, pitch=0.2, keep_largest_component=True)
            self.assertEqual(grid.meta["layer_count"], 2)
            self.assertEqual(grid.meta["connectivity"]["components"], 2)
            self.assertEqual(grid.meta["connectivity"]["dropped_components"], 1)
            self.assertGreater(grid.meta["overlap_voxel_fraction"], 0.0)
            self.assertIn("outer_wall", grid.meta["feature_voxel_counts"])
            self.assertIn("sparse_infill", grid.meta["feature_voxel_counts"])
            self.assertIn("solid_infill", grid.meta["feature_voxel_counts"])

            # Verify Z alignment: layer 1 (0.2) + layer 2 (0.4) -> origin_z = 0.0, voxels iz in {0, 1}
            self.assertAlmostEqual(float(grid.origin[2]), 0.0, places=6)
            # Disconnected island at X=30 was pruned -> max x node <= 10 mm
            node_ijk, _, _, _ = build_level_topology(grid)
            x_max = float((grid.origin[0] + node_ijk[:, 0] * grid.pitch[0]).max())
            self.assertLess(x_max, 11.0)

            # Verify angles near diagonal 45 deg and vertical 90 deg exist in payload
            active_bit = ((grid.brick_masks[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)) != 0
            abin, _, flags = unpack_payload(grid.brick_payload)
            angles_deg = np.rad2deg(bin_to_angle(abin[active_bit], 360))
            self.assertTrue(np.any(np.isclose(angles_deg, 26.5, atol=2.0)))  # atan2(4, 8) ~ 26.565 deg
            self.assertTrue(np.any(np.isclose(angles_deg, 90.0, atol=0.6)))
            self.assertTrue(np.any((flags[active_bit] & FLAG_BEAD_OVERLAP) != 0))
            self.assertTrue(np.any(decode_feature(flags[active_bit]) == FEATURE_OUTER_WALL))
            self.assertTrue(np.any(decode_feature(flags[active_bit]) == FEATURE_SPARSE_INFILL))

            # Reload and verify atomic .tbricks checksum
            loaded = load_bricks(bricks_path, verify_checksum=True)
            np.testing.assert_array_equal(loaded.brick_coords, grid.brick_coords)
            np.testing.assert_array_equal(loaded.brick_masks, grid.brick_masks)
            np.testing.assert_array_equal(loaded.brick_payload, grid.brick_payload)

            # Tamper 1 byte -> must raise checksum mismatch
            raw = bytearray(bricks_path.read_bytes())
            raw[-1] ^= 0xFF
            bricks_path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_bricks(bricks_path, verify_checksum=True)

            # Disconnected rejection when keep_largest_component=False
            with self.assertRaisesRegex(ValueError, "不连通"):
                gcode_to_bricks(gcode_path, pitch=0.2, keep_largest_component=False)

    def test_voided_infill_eccentric_load_vs_direct_spsolve(self):
        # Build a multi-layer 16x16x6-voxel voided crosshatch lattice (pitch=0.2mm -> 3.2x3.2x1.2mm)
        # with varying density rho and alternating 0/90 deg toolpath angles + eccentric load_point.
        nx, ny, nz = 16, 16, 6
        pitch = 0.2
        occ = np.zeros((nx, ny, nz), dtype=bool)
        ang = np.zeros((nx, ny, nz), dtype=np.float64)
        rho = np.ones((nx, ny, nz), dtype=np.float64)

        # Solid outer wall (2 cells) + bottom/top skin + alternating 0/90 sparse ribs inside
        occ[:2, :, :] = True
        occ[-2:, :, :] = True
        occ[:, :2, :] = True
        occ[:, -2:, :] = True
        occ[:, :, 0] = True
        occ[:, :, -1] = True
        for iz in range(1, nz - 1):
            if iz % 2 == 1:
                occ[:, 4:6, iz] = True
                occ[:, 10:12, iz] = True
                ang[:, :, iz] = 0.0
            else:
                occ[4:6, :, iz] = True
                occ[10:12, :, iz] = True
                ang[:, :, iz] = math.pi / 2.0
        # Partial boundary density on ribs
        rho[4:6, 4:12, :] = 0.75

        ijk = np.argwhere(occ).astype(np.int32)
        grid = mask_to_bricks(
            ijk,
            pitch,
            angles_rad=ang[occ],
            rho=rho[occ],
        )

        m = Material()
        load_pt = (1.6, 1.6, 5.0)
        force = (0.0, -2.5, -1.0)
        res_gpu = solve_hex_gpu(
            grid,
            material=m,
            fixed="X-",
            loaded="X+",
            force=force,
            load_point=load_pt,
            rtol=1e-8,
            coarse_dof_target=200,
        )

        # Exact SciPy spsolve reference on identical brick grid
        node_ijk, _, brick_nodes, _ = build_level_topology(grid)
        node_xyz = grid.origin[None, :] + node_ijk.astype(np.float64) * grid.pitch[None, :]
        n_nodes = len(node_ijk)
        Ke_f64, _, D_local = build_stiffness_lut(m, pitch, pitch, pitch, n_bins=360)
        K_csr = assemble_coarse_csr(grid, brick_nodes, n_nodes, Ke_f64)
        quads, _ = extract_boundary_quads(grid, brick_nodes)
        fixed_q = select_quad_faces(node_xyz, quads, "X-")
        load_q = select_quad_faces(node_xyz, quads, "X+")
        f_ref = quad_surface_load(node_xyz, quads[load_q], n_nodes, force, load_point=load_pt).ravel()
        fixed_nodes = np.unique(quads[fixed_q])
        fixed_dofs = (fixed_nodes[:, None] * 3 + np.arange(3)).ravel()
        free_dofs = np.setdiff1d(np.arange(3 * n_nodes), fixed_dofs)
        u_ref = np.zeros(3 * n_nodes, dtype=np.float64)
        u_ref[free_dofs] = spsolve(K_csr[free_dofs][:, free_dofs], f_ref[free_dofs])
        u_ref = u_ref.reshape(-1, 3)

        rel_disp_err = np.linalg.norm(res_gpu["displacement"] - u_ref) / np.linalg.norm(u_ref)
        self.assertLess(rel_disp_err, 1e-5)
        self.assertLess(res_gpu["solver_info"]["relative_residual"], 1e-6)
        self.assertLess(res_gpu["solver_info"]["force_relative_error"], 1e-6)
        self.assertLess(res_gpu["solver_info"]["moment_relative_error"], 1e-6)

        # Compare streaming stress recovery + binary spill file vs direct NumPy hex_centroid_stress
        with tempfile.TemporaryDirectory() as tmp:
            spill_file = Path(tmp) / "stress.bin"
            stress_info = recover_stress_streaming(
                grid,
                brick_nodes,
                res_gpu["displacement"],
                material=m,
                chunk_bricks=3,
                spill_path=spill_file,
            )
            self.assertTrue(spill_file.is_file())
            # Verify direct evaluation on all active voxels
            _, B0 = hex_B_matrices(pitch, pitch, pitch)
            corner_offs = np.array([0, 1, 5, 6, 25, 26, 30, 31], dtype=np.int32)
            u_elems = []
            ang_elems = []
            for b in range(grid.num_bricks):
                for c in range(64):
                    if ((int(grid.brick_masks[b]) >> c) & 1) == 0:
                        continue
                    lx, ly, lz = c & 3, (c >> 2) & 3, c >> 4
                    base = lx + 5 * ly + 25 * lz
                    nids = brick_nodes[b, base + corner_offs]
                    u_elems.append(u_ref[nids].ravel())
                    abin = int(grid.brick_payload[b, c] & 0xFFFF)
                    ang_elems.append(float(bin_to_angle(abin, 360)))
            loc_ref, glob_ref = hex_centroid_stress(np.asarray(u_elems), D_local, B0, np.asarray(ang_elems))
            sx, sy, sz, txy, tyz, txz = glob_ref.T
            vm_ref = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy ** 2 + tyz ** 2 + txz ** 2))
            np.testing.assert_allclose(stress_info["_vm_f32"], vm_ref.astype(np.float32), rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(stress_info["_sl_f32"], loc_ref[:, 2].astype(np.float32), rtol=1e-4, atol=1e-5)

            # Verify binary spill layout matches exact voxel count
            raw = spill_file.read_bytes()
            expected_len = 16 + grid.num_voxels * 5 * 4
            self.assertEqual(len(raw), expected_len)


if __name__ == "__main__":
    unittest.main(verbosity=2)
