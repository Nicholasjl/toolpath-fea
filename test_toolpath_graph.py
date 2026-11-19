import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from beam_bond_fem import (
    _section_properties,
    eccentric_bond_stiffness_batch,
    timoshenko_beam_stiffness_batch,
)
from beam_bond_solver import (
    _bond_moduli,
    _bond_scale_factors,
    _beam_scale_factors,
    calibrate_graph_to_continuum,
    recover_graph_stress_streaming,
    solve_toolpath_graph_gpu,
)
from gpu_hex_solver import solve_hex_gpu
from solid import Material
from toolpath_bricks import gcode_to_bricks
from toolpath_graph import (
    BOND_INTERLAYER,
    BOND_SEAM_CLOSED,
    BOND_SEAM_SCARF,
    NODE_RETRACTION_START,
    NODE_SCARF_RAMP,
    NODE_SEAM_LOOP,
    NODE_START,
    NODE_STOP,
    SeamConfig,
    StrengthConfig,
    build_synthetic_graph,
    gcode_to_toolpath_graph,
    load_toolpath_graph,
)


def _six_rbm_12(pa, pb):
    """Build exact 6 rigid-body modes for a 2-node 6-DOF element at (pa, pb)."""
    pts = np.vstack([pa, pb])
    center = pts.mean(axis=0)
    modes = np.zeros((12, 6), dtype=np.float64)
    for d in range(3):
        modes[d, d] = 1.0
        modes[6 + d, d] = 1.0
    for ax in range(3):
        omega = np.eye(3)[ax]
        modes[0:3, 3 + ax] = np.cross(omega, pa - center)
        modes[3:6, 3 + ax] = omega
        modes[6:9, 3 + ax] = np.cross(omega, pb - center)
        modes[9:12, 3 + ax] = omega
    return modes


class ToolpathGraphTests(unittest.TestCase):
    def test_beam_and_eccentric_bond_symmetry_and_rbm(self):
        m = Material()
        p0 = np.array([[1.2, -0.5, 0.3], [0.0, 0.0, 0.1]])
        p1 = np.array([[2.7, 1.1, 0.3], [0.0, 0.0, 1.5]])
        Ke_b = timoshenko_beam_stiffness_batch(p0, p1, np.array([0.45, 0.42]), np.array([0.2, 0.2]), m)
        for i in range(2):
            np.testing.assert_allclose(Ke_b[i], Ke_b[i].T, atol=1e-11 * np.max(np.abs(Ke_b[i])))
            rbm = _six_rbm_12(p0[i], p1[i])
            np.testing.assert_allclose(Ke_b[i] @ rbm, 0.0, atol=1e-12 * np.max(np.abs(Ke_b[i])))

        # Eccentric bond with non-zero lever arms r_a and r_b
        pa = np.array([[1.0, 2.0, 0.1]])
        pb = np.array([[1.1, 2.05, 0.3]])
        pc = np.array([[1.05, 2.025, 0.2]])
        n_vec = np.array([[0.0, 0.0, 1.0]])
        t1 = np.array([[1.0, 0.0, 0.0]])
        t2 = np.array([[0.0, 1.0, 0.0]])
        Ke_k = eccentric_bond_stiffness_batch(
            pa, pb, pc, n_vec, t1, t2,
            np.array([0.4]), np.array([0.8]), np.array([0.5]), np.array([0.2]),
            np.array([m.e3]), np.array([m.g13]), np.array([m.g23]),
        )[0]
        np.testing.assert_allclose(Ke_k, Ke_k.T, atol=1e-12 * np.max(np.abs(Ke_k)))
        rbm_k = _six_rbm_12(pa[0], pb[0])
        np.testing.assert_allclose(Ke_k @ rbm_k, 0.0, atol=1e-12 * np.max(np.abs(Ke_k)))

    def test_single_beam_exact_tension_bending_and_torsion(self):
        L = 20.0
        w, h = 0.5, 0.25
        n_seg = 20
        xs = np.linspace(0.0, L, n_seg + 1)
        xyz = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
        edges = np.column_stack([np.arange(n_seg), np.arange(1, n_seg + 1)])
        g = build_synthetic_graph(xyz, w, h, edges)
        m = Material()
        A, I2, I3, J = _section_properties(w, h)
        E = m.e1
        G_xy = m.g12
        G_xz = m.g13
        G_t = math.sqrt(G_xy * G_xz)
        kappa = 5.0 / 6.0

        # 1. Axial tension Fx = 10 N
        r_ax = solve_toolpath_graph_gpu(g, material=m, seam_cfg=SeamConfig.ideal(), fixed="X-", loaded="X+", force=(10.0, 0.0, 0.0))
        exact_ux = 10.0 * L / (E * A)
        np.testing.assert_allclose(r_ax["displacement"][-1, 0], exact_ux, rtol=1e-10)

        # 2. In-plane XY bending Fy = 0.1 N
        r_by = solve_toolpath_graph_gpu(g, material=m, seam_cfg=SeamConfig.ideal(), fixed="X-", loaded="X+", force=(0.0, 0.1, 0.0))
        exact_uy = 0.1 * L**3 / (3.0 * E * I3) + 0.1 * L / (kappa * G_xy * A)
        np.testing.assert_allclose(r_by["displacement"][-1, 1], exact_uy, rtol=1e-10)
        self.assertLess(r_by["solver_info"]["moment_relative_error"], 1e-10)

        # 3. Out-of-plane Z bending Fz = -0.1 N
        r_bz = solve_toolpath_graph_gpu(g, material=m, seam_cfg=SeamConfig.ideal(), fixed="X-", loaded="X+", force=(0.0, 0.0, -0.1))
        exact_uz = -0.1 * L**3 / (3.0 * E * I2) - 0.1 * L / (kappa * G_xz * A)
        np.testing.assert_allclose(r_bz["displacement"][-1, 2], exact_uz, rtol=1e-10)
        self.assertLess(r_bz["solver_info"]["moment_relative_error"], 1e-10)

        # 4. Pure torsion check on analytical 1-beam Ke AND eccentric torque on a 2x2 stacked 4-node tip
        Ke1 = timoshenko_beam_stiffness_batch(np.array([[0.0, 0.0, 0.0]]), np.array([[L, 0.0, 0.0]]), np.array([w]), np.array([h]), m)[0]
        f_tor = np.zeros(12)
        f_tor[9] = 1.0  # Mx = 1 Nmm at free tip
        u_tor = np.zeros(12)
        u_tor[6:] = np.linalg.solve(Ke1[6:, 6:], f_tor[6:])
        np.testing.assert_allclose(u_tor[9], 1.0 * L / (G_t * J), rtol=1e-12)

        # 2x2 stacked beams (4 non-collinear tip nodes -> full rank-3 load inertia) under eccentric torque
        xyz_t = np.vstack([
            xyz,
            xyz + np.array([0.0, w, 0.0]),
            xyz + np.array([0.0, 0.0, h]),
            xyz + np.array([0.0, w, h]),
        ])
        s = n_seg + 1
        edges_t = np.vstack([edges, edges + s, edges + 2 * s, edges + 3 * s])
        # Lateral + interlayer couplers connecting the 4 parallel strands
        lat1 = np.column_stack([np.arange(s), np.arange(s) + s])
        lat2 = np.column_stack([np.arange(s) + 2 * s, np.arange(s) + 3 * s])
        il1 = np.column_stack([np.arange(s), np.arange(s) + 2 * s])
        il2 = np.column_stack([np.arange(s) + s, np.arange(s) + 3 * s])
        k_edges = np.vstack([lat1, lat2, il1, il2])
        n_lat = len(lat1) + len(lat2)
        n_il = len(il1) + len(il2)
        k_type = np.r_[np.full(n_lat, 2, dtype=np.uint8), np.full(n_il, 1, dtype=np.uint8)]
        k_pc = 0.5 * (xyz_t[k_edges[:, 0]] + xyz_t[k_edges[:, 1]])
        k_n = np.vstack([np.tile([0.0, 1.0, 0.0], (n_lat, 1)), np.tile([0.0, 0.0, 1.0], (n_il, 1))])
        k_t1 = np.tile([1.0, 0.0, 0.0], (n_lat + n_il, 1))
        k_t2 = np.vstack([np.tile([0.0, 0.0, 1.0], (n_lat, 1)), np.tile([0.0, 1.0, 0.0], (n_il, 1))])
        k_geom = np.vstack([
            np.tile([L / n_seg * h, L / n_seg, h, w], (n_lat, 1)),
            np.tile([L / n_seg * w, L / n_seg, w, h], (n_il, 1)),
        ])
        gt = build_synthetic_graph(
            xyz_t, w, h, edges_t,
            bond_edges=k_edges, bond_type=k_type, bond_pc=k_pc,
            bond_normal=k_n, bond_t1=k_t1, bond_t2=k_t2, bond_geom=k_geom,
        )
        r_gt = solve_toolpath_graph_gpu(
            gt, material=m, seam_cfg=SeamConfig.ideal(), fixed="X-", loaded="X+",
            force=(0.0, 0.0, -0.1), load_point=(L, 5.0, 0.5 * h),
        )
        self.assertLess(r_gt["solver_info"]["relative_residual"], 1e-6)
        self.assertLess(r_gt["solver_info"]["moment_relative_error"], 1e-6)

    def test_four_layer_eccentric_stack_composite_bending(self):
        # 4 stacked beads (h=0.2 each -> total H=0.8 mm, width w=0.5 mm, length L=20 mm).
        # With eccentric interlayer couplers, bending I_comp must reflect b*H^3/12 = 16x sum(I_layer)!
        e = 2400.0
        nu = 0.25
        g_iso = e / (2.0 * (1.0 + nu))
        m = Material(e, e, e, g_iso, g_iso, g_iso, nu, nu, nu)

        L, w, h = 20.0, 0.5, 0.2
        nx = 40
        dx = L / nx
        xs = np.linspace(0.0, L, nx + 1)
        xyz_list, b_list = [], []
        n_per = nx + 1
        for lay in range(4):
            zc = (lay + 0.5) * h
            xyz_list.append(np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, zc)]))
            off = lay * n_per
            b_list.append(np.column_stack([off + np.arange(nx), off + np.arange(1, nx + 1)]))
        xyz = np.vstack(xyz_list)
        beam_edges = np.vstack(b_list)

        k_edges, k_pc = [], []
        for lay in range(3):
            o0 = lay * n_per
            o1 = (lay + 1) * n_per
            for i in range(n_per):
                k_edges.append([o0 + i, o1 + i])
                k_pc.append([xs[i], 0.0, (lay + 1) * h])
        mk = len(k_edges)
        trib = np.full(n_per, dx)
        trib[0] = 0.5 * dx
        trib[-1] = 0.5 * dx
        l1_arr = np.tile(trib, 3)
        geom = np.column_stack([w * l1_arr, l1_arr, np.full(mk, w), np.full(mk, h)])

        graph = build_synthetic_graph(
            xyz, w, h, beam_edges,
            bond_edges=np.asarray(k_edges, dtype=np.int32),
            bond_type=np.full(mk, BOND_INTERLAYER, dtype=np.uint8),
            bond_pc=np.asarray(k_pc, dtype=np.float64),
            bond_normal=np.tile([0.0, 0.0, 1.0], (mk, 1)),
            bond_t1=np.tile([1.0, 0.0, 0.0], (mk, 1)),
            bond_t2=np.tile([0.0, 1.0, 0.0], (mk, 1)),
            bond_geom=geom,
        )
        res = solve_toolpath_graph_gpu(graph, material=m, seam_cfg=SeamConfig.ideal(), fixed="X-", loaded="X+", force=(0.0, 0.0, -0.1))
        tip_uz = float(abs(res["displacement"][xyz[:, 0] == L, 2].mean()))

        H_tot = 4 * h
        I_comp = w * H_tot**3 / 12.0
        exact_uz = 0.1 * L**3 / (3.0 * e * I_comp) + 0.1 * L / ((5.0 / 6.0) * g_iso * (w * H_tot))
        # Without eccentric arms it would be ~16x softer; with eccentric arms it matches composite beam < 5%!
        self.assertLess(abs(tip_uz - exact_uz) / exact_uz, 0.05)

    def test_closed_loop_z_seam_vs_scarf_seam(self):
        # Square closed ring (20 x 20 mm) with explicit Z-seam at top span X=10, Y=20 (free & carrying X tension!)
        pts = []
        pts.extend([[x, 20.0, 0.1] for x in np.linspace(10.0, 0.0, 6)])
        pts.extend([[0.0, y, 0.1] for y in np.linspace(18.0, 0.0, 10)])
        pts.extend([[x, 0.0, 0.1] for x in np.linspace(2.0, 20.0, 10)])
        pts.extend([[20.0, y, 0.1] for y in np.linspace(2.0, 20.0, 10)])
        pts.extend([[x, 20.0, 0.1] for x in np.linspace(18.0, 10.0, 5)])
        xyz = np.asarray(pts, dtype=np.float64)
        n = len(xyz)
        beams = np.column_stack([np.arange(n - 1), np.arange(1, n)])
        k_edges = np.array([[n - 1, 0]], dtype=np.int32)

        def _make_loop(btype):
            return build_synthetic_graph(
                xyz, 0.5, 0.2, beams,
                bond_edges=k_edges,
                bond_type=np.array([btype], dtype=np.uint8),
                bond_pc=np.array([[10.0, 20.0, 0.1]]),
                bond_normal=np.array([[-1.0, 0.0, 0.0]], dtype=np.float32),
                bond_t1=np.array([[0.0, -1.0, 0.0]], dtype=np.float32),
                bond_t2=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                bond_geom=np.array([[0.5 * 0.2, 0.5, 0.2, 0.1]], dtype=np.float32),
            )

        g_butt = _make_loop(BOND_SEAM_CLOSED)
        g_scarf = _make_loop(BOND_SEAM_SCARF)
        m = Material()

        u_ideal = solve_toolpath_graph_gpu(g_butt, m, SeamConfig.ideal(), fixed=[[-0.1, -0.1, 0.0], [0.1, 20.1, 0.2]], loaded=[[19.9, -0.1, 0.0], [20.1, 20.1, 0.2]], force=(1.0, 0.0, 0.0))
        u_weak = solve_toolpath_graph_gpu(g_butt, m, SeamConfig(seam_stiffness_factor=0.1), fixed=[[-0.1, -0.1, 0.0], [0.1, 20.1, 0.2]], loaded=[[19.9, -0.1, 0.0], [20.1, 20.1, 0.2]], force=(1.0, 0.0, 0.0))
        u_broken = solve_toolpath_graph_gpu(g_butt, m, SeamConfig(seam_stiffness_factor=0.001), fixed=[[-0.1, -0.1, 0.0], [0.1, 20.1, 0.2]], loaded=[[19.9, -0.1, 0.0], [20.1, 20.1, 0.2]], force=(1.0, 0.0, 0.0))
        u_scarf = solve_toolpath_graph_gpu(g_scarf, m, SeamConfig(seam_stiffness_factor=0.1, scarf_seam_stiffness_factor=0.85), fixed=[[-0.1, -0.1, 0.0], [0.1, 20.1, 0.2]], loaded=[[19.9, -0.1, 0.0], [20.1, 20.1, 0.2]], force=(1.0, 0.0, 0.0))

        c_ideal = u_ideal["solver_info"]["compliance_mm_per_N"]
        c_weak = u_weak["solver_info"]["compliance_mm_per_N"]
        c_broken = u_broken["solver_info"]["compliance_mm_per_N"]
        c_scarf = u_scarf["solver_info"]["compliance_mm_per_N"]
        self.assertLess(c_ideal, c_scarf)
        self.assertLess(c_scarf, c_weak)
        self.assertLess(c_weak, c_broken)

        # Seam opening jump grows monotonically as seam weakens
        j_ideal = float(np.linalg.norm(u_ideal["displacement"][0] - u_ideal["displacement"][-1]))
        j_weak = float(np.linalg.norm(u_weak["displacement"][0] - u_weak["displacement"][-1]))
        j_broken = float(np.linalg.norm(u_broken["displacement"][0] - u_broken["displacement"][-1]))
        self.assertLess(j_ideal, j_weak)
        self.assertLess(j_weak, j_broken)

    def test_streaming_gcode_to_tgraph_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_d = Path(tmp)
            gcode_p = tmp_d / "sample.gcode"
            graph_p = tmp_d / "sample.tgraph"
            gcode_text = "\n".join(
                [
                    "G21",
                    "G90",
                    "M83",
                    "; LAYER_HEIGHT: 0.2",
                    "; FEATURE: Skirt",
                    "G1 X-5.0 Y-5.0 Z0.2 F3000",
                    "G1 X20.0 Y-5.0 E0.5",
                    "; FEATURE: Outer wall",
                    "; LINE_WIDTH: 0.5",
                    "G1 E-0.8 F2400",
                    "G0 X0.0 Y0.0 Z0.2 F12000",
                    "G1 Z0.6 F3000",
                    "G1 Z0.2 F3000",
                    "G1 E0.8 F2400",
                    "G1 X10.0 Y0.0 E0.4 F3000",
                    "G1 X10.0 Y10.0 E0.4",
                    "G1 X0.0 Y10.0 E0.4",
                    "G1 X0.0 Y0.0 E0.4",
                    "; FEATURE: Sparse infill",
                    "; LINE_WIDTH: 0.42",
                    "G0 X0.25 Y5.0",
                    "G1 X9.75 Y5.0 E0.35",
                    "; CHANGE_LAYER",
                    "G1 Z0.8 F3000",
                    "G1 Z0.4 F3000",
                    "; FEATURE: Outer wall",
                    "; LINE_WIDTH: 0.5",
                    "G0 X0.0 Y0.0",
                    "G1 X10.0 Y0.0 E0.4",
                    "G1 X10.0 Y10.0 E0.4",
                    "G1 X0.0 Y10.0 E0.4",
                    "; LINE_WIDTH: 0.2",
                    "G1 X0.0 Y0.0 E0.2",
                    # Disconnected floating island far away -> dropped
                    "G0 X50.0 Y50.0",
                    "G1 X54.0 Y50.0 E0.2",
                ]
            )
            gcode_p.write_text(gcode_text, encoding="utf-8")

            g = gcode_to_toolpath_graph(gcode_p, out_graph_path=graph_p, max_seg_len=1.0)
            self.assertEqual(g.meta["layer_count"], 2)
            self.assertEqual(g.meta["connectivity"]["dropped_components"], 1)
            self.assertIn("seam_closed", g.meta["bond_counts"])
            self.assertIn("seam_scarf", g.meta["bond_counts"])
            self.assertIn("interlayer", g.meta["bond_counts"])
            self.assertIn("start_anchor", g.meta["bond_counts"])
            self.assertIn("stop_anchor", g.meta["bond_counts"])
            self.assertTrue(np.any((g.node_flags & NODE_RETRACTION_START) != 0))
            self.assertTrue(np.any((g.node_flags & NODE_SCARF_RAMP) != 0))
            self.assertGreater(float(g.bond_dt[g.bond_type == BOND_INTERLAYER].max()), 0.0)

            g2 = load_toolpath_graph(graph_p, verify_checksum=True)
            np.testing.assert_allclose(g2.node_xyz, g.node_xyz)
            np.testing.assert_array_equal(g2.bond_edges, g.bond_edges)

            # Tamper byte -> checksum mismatch
            raw = bytearray(graph_p.read_bytes())
            raw[-1] ^= 0xFF
            graph_p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_toolpath_graph(graph_p, verify_checksum=True)

    def test_perforated_multilayer_frame_gpu_mg_vs_direct_spsolve(self):
        # Build a 4-layer 24x24 mm multi-bead frame with central cutout & eccentric load_point
        # and force GPU Galerkin aggregation MG (coarse_dof_target=120) vs exact SciPy spsolve!
        with tempfile.TemporaryDirectory() as tmp:
            gcode_p = Path(tmp) / "frame.gcode"
            lines = ["G21", "G90", "M83", "; LAYER_HEIGHT: 0.2"]
            for lay in range(4):
                z = 0.2 * (lay + 1)
                lines.append(f"; CHANGE_LAYER\nG1 Z{z:.2f} F3000")
                lines.append("; FEATURE: Outer wall\n; LINE_WIDTH: 0.5")
                for offset_in in (0.0, 0.5, 1.0):
                    lo, hi = offset_in, 24.0 - offset_in
                    lines.extend([
                        f"G0 X{lo:.2f} Y{lo:.2f}",
                        f"G1 X{hi:.2f} Y{lo:.2f} E1.0",
                        f"G1 X{hi:.2f} Y{hi:.2f} E1.0",
                        f"G1 X{lo:.2f} Y{hi:.2f} E1.0",
                        f"G1 X{lo:.2f} Y{lo:.2f} E1.0",
                    ])
                lines.append("; FEATURE: Sparse infill\n; LINE_WIDTH: 0.42")
                for c in (4.0, 8.0, 16.0, 20.0):
                    if lay % 2 == 0:
                        lines.append(f"G0 X1.0 Y{c:.2f}\nG1 X23.0 Y{c:.2f} E0.8")
                    else:
                        lines.append(f"G0 X{c:.2f} Y1.0\nG1 X{c:.2f} Y23.0 E0.8")
            gcode_p.write_text("\n".join(lines), encoding="utf-8")

            graph = gcode_to_toolpath_graph(gcode_p, max_seg_len=1.5)
            m = Material()
            seam_cfg = SeamConfig()
            force = (0.0, -1.5, -0.5)
            load_pt = (24.0, 12.0, 6.0)

            res_gpu = solve_toolpath_graph_gpu(
                graph,
                material=m,
                seam_cfg=seam_cfg,
                fixed="X-",
                loaded="X+",
                force=force,
                load_point=load_pt,
                rtol=1e-8,
                coarse_dof_target=120,
            )
            self.assertGreater(len(res_gpu["solver_info"]["levels"]), 1)
            self.assertLess(res_gpu["solver_info"]["relative_residual"], 1e-6)
            self.assertLess(res_gpu["solver_info"]["force_relative_error"], 1e-6)
            self.assertLess(res_gpu["solver_info"]["moment_relative_error"], 1e-6)

            # Direct exact SciPy spsolve reference (coarse_dof_target > total DOFs triggers exact spsolve path)
            res_ref = solve_toolpath_graph_gpu(
                graph,
                material=m,
                seam_cfg=seam_cfg,
                fixed="X-",
                loaded="X+",
                force=force,
                load_point=load_pt,
                coarse_dof_target=graph.num_nodes * 6 + 100,
            )
            self.assertEqual(res_ref["solver_info"]["coarse_solver"], "spsolve_direct")
            rel_err = np.linalg.norm(res_gpu["displacement_6dof"] - res_ref["displacement_6dof"]) / np.linalg.norm(
                res_ref["displacement_6dof"]
            )
            self.assertLess(rel_err, 1e-5)

            # Streaming stress audit check
            stress_info = recover_graph_stress_streaming(graph, res_gpu["displacement_6dof"], material=m, seam_cfg=seam_cfg)
            self.assertGreater(stress_info["beam_von_mises_max_MPa"], 0.0)
            self.assertIn("seam_closed", stress_info["bond_by_type"])
            self.assertIn("interlayer", stress_info["bond_by_type"])

    def test_failure_index_thermal_decay_and_continuum_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            gcode_p = Path(tmp) / "coupon.gcode"
            # 4-layer 10 x 2.0 x 0.8 mm rectangular multi-bead coupon printed slowly (F300) so dt_sec is several seconds
            lines = ["G21", "G90", "M83", "; LAYER_HEIGHT: 0.2"]
            for lay in range(4):
                z = 0.2 * (lay + 1)
                lines.append(f"; CHANGE_LAYER\nG1 Z{z:.2f} F600")
                lines.append("; FEATURE: Outer wall\n; LINE_WIDTH: 0.4")
                # Non-overlapping closed outer perimeter + internal infill beads
                lines.extend([
                    "G0 X5.0 Y0.2 F3000",
                    "G1 X10.0 Y0.2 E0.2 F300",
                    "G1 X10.0 Y1.8 E0.1",
                    "G1 X0.0 Y1.8 E0.4",
                    "G1 X0.0 Y0.2 E0.1",
                    "G1 X5.0 Y0.2 E0.2",
                ])
                lines.append("; FEATURE: Solid infill\n; LINE_WIDTH: 0.4")
                for y_off in (0.6, 1.0, 1.4):
                    lines.append(f"G0 X0.35 Y{y_off:.2f} F3000")
                    lines.append(f"G1 X9.65 Y{y_off:.2f} E0.4 F300")
            gcode_p.write_text("\n".join(lines), encoding="utf-8")

            graph = gcode_to_toolpath_graph(gcode_p, max_seg_len=0.5)
            m = Material()
            strength = StrengthConfig()
            force = (0.0, 0.0, -0.5)

            # (a) Weakening Z-seam increases failure_index & reduces FoS on seam_closed
            sc_ideal = SeamConfig.ideal()
            sc_weak_seam = SeamConfig(
                seam_stiffness_factor=0.35,
                scarf_seam_stiffness_factor=1.0,
                start_anchor_factor=1.0,
                stop_anchor_factor=1.0,
                start_zone_beam_factor=1.0,
                stop_zone_beam_factor=1.0,
                lateral_weld_factor=1.0,
                interlayer_weld_factor=1.0,
            )
            r_id = solve_toolpath_graph_gpu(graph, m, sc_ideal, fixed="X-", loaded="X+", force=force)
            st_id = recover_graph_stress_streaming(graph, r_id["displacement_6dof"], m, sc_ideal, strength)
            r_ws = solve_toolpath_graph_gpu(graph, m, sc_weak_seam, fixed="X-", loaded="X+", force=force)
            st_ws = recover_graph_stress_streaming(graph, r_ws["displacement_6dof"], m, sc_weak_seam, strength)

            fi_id = st_id["bond_by_type"]["seam_closed"]["failure_index_max"]
            fi_ws = st_ws["bond_by_type"]["seam_closed"]["failure_index_max"]
            self.assertGreater(fi_ws, fi_id)
            self.assertLess(st_ws["bond_by_type"]["seam_closed"]["fos_min"], st_id["bond_by_type"]["seam_closed"]["fos_min"])
            self.assertEqual(st_ws["strength_config"], strength.to_dict())

            # (b) Turning on thermal_decay_tau_s softens interlayer welds on nonzero dt_sec -> higher compliance & interlayer FI
            sc_therm = SeamConfig(
                seam_stiffness_factor=0.35,
                scarf_seam_stiffness_factor=1.0,
                start_anchor_factor=1.0,
                stop_anchor_factor=1.0,
                start_zone_beam_factor=1.0,
                stop_zone_beam_factor=1.0,
                lateral_weld_factor=1.0,
                interlayer_weld_factor=1.0,
                thermal_decay_tau_s=10.0,
            )
            r_th = solve_toolpath_graph_gpu(graph, m, sc_therm, fixed="X-", loaded="X+", force=force)
            st_th = recover_graph_stress_streaming(graph, r_th["displacement_6dof"], m, sc_therm, strength)
            self.assertGreater(r_th["solver_info"]["compliance_mm_per_N"], r_ws["solver_info"]["compliance_mm_per_N"])
            self.assertGreater(
                st_th["bond_by_type"]["interlayer"]["failure_index_max"],
                st_ws["bond_by_type"]["interlayer"]["failure_index_max"],
            )
            self.assertTrue(st_th["safety_audit"]["thermal_decay_active"])

            # (c) Calibrate graph against 0.2 mm GPU H8 continuum within < 2% without altering StrengthConfig allowables
            grid = gcode_to_bricks(gcode_p, pitch=0.2)
            r_hex = solve_hex_gpu(grid, material=m, fixed="X-", loaded="X+", force=force)
            target_wu = r_hex["solver_info"]["work_conjugate_displacement_mm"]
            cal_cfg, cal_audit = calibrate_graph_to_continuum(
                graph,
                m,
                target_wu,
                fixed="X-",
                loaded="X+",
                force=force,
                base_seam_cfg=SeamConfig.ideal(),
                mode="split",
                max_evals=3,
            )
            self.assertLess(cal_audit["relative_error"], 0.02)
            r_cal = solve_toolpath_graph_gpu(graph, m, cal_cfg, fixed="X-", loaded="X+", force=force)
            wu_cal = r_cal["solver_info"]["work_conjugate_displacement_mm"]
            self.assertLess(abs(wu_cal - target_wu) / target_wu, 0.02)
            st_cal = recover_graph_stress_streaming(graph, r_cal["displacement_6dof"], m, cal_cfg, strength)
            self.assertEqual(st_cal["strength_config"], strength.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
