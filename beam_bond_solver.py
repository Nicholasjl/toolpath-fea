"""GPU-accelerated 6-DOF BSR + Galerkin Connected-Aggregate Multigrid solver for toolpath graphs."""
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np
import psutil
from scipy.sparse import bsr_matrix, coo_matrix, diags as sp_diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve
import warp as wp

from beam_bond_fem import (
    beam_internal_stress_batch,
    bond_interface_stress_batch,
    eccentric_bond_stiffness_batch,
    timoshenko_beam_stiffness_batch,
)
from solid import Material
from toolpath_bricks import FEATURE_NAMES
from toolpath_graph import (
    BOND_INTERLAYER,
    BOND_LATERAL,
    BOND_SEAM_CLOSED,
    BOND_SEAM_SCARF,
    BOND_START_ANCHOR,
    BOND_STOP_ANCHOR,
    BOND_TYPE_NAMES,
    NODE_START_ZONE,
    NODE_STOP_ZONE,
    SeamConfig,
    ToolpathGraph,
)


_WARP_INIT = False


def _get_device(device="auto"):
    global _WARP_INIT
    if not _WARP_INIT:
        wp.init()
        _WARP_INIT = True
    if device == "auto":
        return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    return device


_vec6f = wp.types.vector(length=6, dtype=wp.float32)


@wp.kernel
def _bsr6_matvec_f32_kernel(
    n_nodes: wp.int32,
    indptr: wp.array(dtype=wp.int32, ndim=1),
    indices: wp.array(dtype=wp.int32, ndim=1),
    data: wp.array(dtype=wp.float32, ndim=3),
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    x: wp.array(dtype=wp.float32, ndim=1),
    y: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    if fixed_mask[i] != wp.uint8(0):
        for d in range(6):
            y[6 * i + d] = wp.float32(0.0)
        return
    acc = _vec6f()
    for d in range(6):
        acc[d] = wp.float32(0.0)
    row_start = indptr[i]
    row_end = indptr[i + 1]
    for k in range(row_start, row_end):
        j = indices[k]
        if fixed_mask[j] == wp.uint8(0):
            xj = _vec6f()
            base_j = 6 * j
            for c in range(6):
                xj[c] = x[base_j + c]
            for r in range(6):
                s = wp.float32(0.0)
                for c in range(6):
                    s += data[k, r, c] * xj[c]
                acc[r] = acc[r] + s
    base_i = 6 * i
    for d in range(6):
        y[base_i + d] = acc[d]


@wp.kernel
def _block6_precond_f32_kernel(
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    d_inv: wp.array(dtype=wp.float32, ndim=3),
    r: wp.array(dtype=wp.float32, ndim=1),
    z: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    base = 6 * i
    if fixed_mask[i] != wp.uint8(0):
        for d in range(6):
            z[base + d] = wp.float32(0.0)
        return
    ri = _vec6f()
    for c in range(6):
        ri[c] = r[base + c]
    for row in range(6):
        s = wp.float32(0.0)
        for c in range(6):
            s += d_inv[i, row, c] * ri[c]
        z[base + row] = s


@wp.kernel
def _block6_cheb_init_f32_kernel(
    inv_c: wp.float32,
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    d_inv: wp.array(dtype=wp.float32, ndim=3),
    r: wp.array(dtype=wp.float32, ndim=1),
    d_vec: wp.array(dtype=wp.float32, ndim=1),
    x: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    base = 6 * i
    if fixed_mask[i] != wp.uint8(0):
        for k in range(6):
            d_vec[base + k] = wp.float32(0.0)
            x[base + k] = wp.float32(0.0)
        return
    ri = _vec6f()
    for c in range(6):
        ri[c] = r[base + c]
    for row in range(6):
        s = wp.float32(0.0)
        for c in range(6):
            s += d_inv[i, row, c] * ri[c]
        step = inv_c * s
        d_vec[base + row] = step
        x[base + row] = x[base + row] + step


@wp.kernel
def _block6_cheb_step_f32_kernel(
    alpha: wp.float32,
    beta: wp.float32,
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    d_inv: wp.array(dtype=wp.float32, ndim=3),
    r: wp.array(dtype=wp.float32, ndim=1),
    d_vec: wp.array(dtype=wp.float32, ndim=1),
    x: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    base = 6 * i
    if fixed_mask[i] != wp.uint8(0):
        for k in range(6):
            d_vec[base + k] = wp.float32(0.0)
            x[base + k] = wp.float32(0.0)
        return
    ri = _vec6f()
    for c in range(6):
        ri[c] = r[base + c]
    for row in range(6):
        s = wp.float32(0.0)
        for c in range(6):
            s += d_inv[i, row, c] * ri[c]
        step = beta * d_vec[base + row] + alpha * s
        d_vec[base + row] = step
        x[base + row] = x[base + row] + step


@wp.kernel
def _axpby_6_f32_kernel(
    alpha: wp.float32,
    x: wp.array(dtype=wp.float32, ndim=1),
    beta: wp.float32,
    y: wp.array(dtype=wp.float32, ndim=1),
    out: wp.array(dtype=wp.float32, ndim=1),
):
    tid = wp.tid()
    out[tid] = alpha * x[tid] + beta * y[tid]


@wp.kernel
def _apply_fixed_6_f32_kernel(
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    vec: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    if fixed_mask[i] != wp.uint8(0):
        base = 6 * i
        for d in range(6):
            vec[base + d] = wp.float32(0.0)


@wp.kernel
def _agg_prolong_add_f32_kernel(
    agg_id: wp.array(dtype=wp.int32, ndim=1),
    r_ij: wp.array(dtype=wp.float32, ndim=2),
    coarse_vec: wp.array(dtype=wp.float32, ndim=1),
    fine_vec: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    J = agg_id[i]
    if J < 0:
        return
    bc = 6 * J
    bf = 6 * i
    ux = coarse_vec[bc]
    uy = coarse_vec[bc + 1]
    uz = coarse_vec[bc + 2]
    tx = coarse_vec[bc + 3]
    ty = coarse_vec[bc + 4]
    tz = coarse_vec[bc + 5]
    rx = r_ij[i, 0]
    ry = r_ij[i, 1]
    rz = r_ij[i, 2]
    # u_i += u_J + theta_J x r_ij
    fine_vec[bf] = fine_vec[bf] + ux + (ty * rz - tz * ry)
    fine_vec[bf + 1] = fine_vec[bf + 1] + uy + (tz * rx - tx * rz)
    fine_vec[bf + 2] = fine_vec[bf + 2] + uz + (tx * ry - ty * rx)
    fine_vec[bf + 3] = fine_vec[bf + 3] + tx
    fine_vec[bf + 4] = fine_vec[bf + 4] + ty
    fine_vec[bf + 5] = fine_vec[bf + 5] + tz


@wp.kernel
def _agg_restrict_f32_kernel(
    agg_id: wp.array(dtype=wp.int32, ndim=1),
    r_ij: wp.array(dtype=wp.float32, ndim=2),
    fine_vec: wp.array(dtype=wp.float32, ndim=1),
    coarse_vec: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    J = agg_id[i]
    if J < 0:
        return
    bf = 6 * i
    bc = 6 * J
    fx = fine_vec[bf]
    fy = fine_vec[bf + 1]
    fz = fine_vec[bf + 2]
    mx = fine_vec[bf + 3]
    my = fine_vec[bf + 4]
    mz = fine_vec[bf + 5]
    rx = r_ij[i, 0]
    ry = r_ij[i, 1]
    rz = r_ij[i, 2]
    # f_J += f_i ; m_J += m_i + r_ij x f_i
    wp.atomic_add(coarse_vec, bc, fx)
    wp.atomic_add(coarse_vec, bc + 1, fy)
    wp.atomic_add(coarse_vec, bc + 2, fz)
    wp.atomic_add(coarse_vec, bc + 3, mx + (ry * fz - rz * fy))
    wp.atomic_add(coarse_vec, bc + 4, my + (rz * fx - rx * fz))
    wp.atomic_add(coarse_vec, bc + 5, mz + (rx * fy - ry * fx))


@wp.kernel
def _dot_6_f32_kernel(
    a: wp.array(dtype=wp.float32, ndim=1),
    b: wp.array(dtype=wp.float32, ndim=1),
    out_sum: wp.array(dtype=wp.float64, ndim=1),
):
    tid = wp.tid()
    bucket = tid & 255
    val = wp.float64(a[tid]) * wp.float64(b[tid])
    wp.atomic_add(out_sum, bucket, val)


def _gpu_dot_f32(a, b, scratch_256, device):
    scratch_256.zero_()
    wp.launch(_dot_6_f32_kernel, dim=len(a), inputs=[a, b, scratch_256], device=device)
    return float(scratch_256.numpy().sum())


def _beam_scale_factors(beam_flags, seam_cfg: SeamConfig):
    sc = np.ones(len(beam_flags), dtype=np.float64)
    in_start = (beam_flags & NODE_START_ZONE) != 0
    in_stop = (beam_flags & NODE_STOP_ZONE) != 0
    if seam_cfg.start_zone_beam_factor != 1.0:
        sc[in_start] *= seam_cfg.start_zone_beam_factor
    if seam_cfg.stop_zone_beam_factor != 1.0:
        sc[in_stop & ~in_start] *= seam_cfg.stop_zone_beam_factor
    return sc


def _bond_scale_factors(bond_type, bond_dt, seam_cfg: SeamConfig):
    fac = np.ones(len(bond_type), dtype=np.float64)
    fac[bond_type == BOND_INTERLAYER] = seam_cfg.interlayer_weld_factor
    fac[bond_type == BOND_LATERAL] = seam_cfg.lateral_weld_factor
    fac[bond_type == BOND_SEAM_CLOSED] = seam_cfg.seam_stiffness_factor
    fac[bond_type == BOND_SEAM_SCARF] = seam_cfg.scarf_seam_stiffness_factor
    fac[bond_type == BOND_START_ANCHOR] = seam_cfg.start_anchor_factor
    fac[bond_type == BOND_STOP_ANCHOR] = seam_cfg.stop_anchor_factor
    if seam_cfg.thermal_decay_tau_s > 0.0:
        inter = bond_type == BOND_INTERLAYER
        if inter.any():
            decay = np.maximum(
                seam_cfg.thermal_min_factor,
                np.exp(-np.maximum(bond_dt[inter], 0.0) / seam_cfg.thermal_decay_tau_s),
            )
            fac[inter] *= decay
    return fac


def _bond_moduli(bond_type, material: Material):
    e_b = np.full(len(bond_type), float(material.e2), dtype=np.float64)
    g1 = np.full(len(bond_type), float(material.g12), dtype=np.float64)
    g2 = np.full(len(bond_type), float(material.g23), dtype=np.float64)

    inter = bond_type == BOND_INTERLAYER
    e_b[inter] = float(material.e3)
    g1[inter] = float(material.g13)
    g2[inter] = float(material.g23)

    seam = (bond_type == BOND_SEAM_CLOSED) | (bond_type == BOND_SEAM_SCARF)
    e_b[seam] = float(material.e1)
    g1[seam] = float(material.g12)
    g2[seam] = float(material.g13)
    return e_b, g1, g2


def select_graph_nodes(graph: ToolpathGraph, region):
    """Select graph nodes using PathFEA face, box, or multi-box union specs."""
    xyz = graph.node_xyz
    if isinstance(region, dict):
        if set(region) != {"boxes"} or not isinstance(region["boxes"], list) or not region["boxes"]:
            raise ValueError("选区并集必须包含非空 boxes 列表")
        masks = [select_graph_nodes(graph, box) for box in region["boxes"]]
        if any(not m.any() for m in masks):
            raise ValueError("选区并集中的一个子区域没有选中走线节点；不能静默丢失安装点")
        return np.logical_or.reduce(masks)
    if isinstance(region, str):
        axis = "XYZ".index(region[0])
        edge = xyz[:, axis].min() if region[1] == "-" else xyz[:, axis].max()
        return np.isclose(xyz[:, axis], edge, atol=1e-5, rtol=0.0)
    box = np.asarray(region, dtype=np.float64)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
        raise ValueError("选区格式：xmin,ymin,zmin,xmax,ymax,zmax；最大值不能小于最小值")
    half = 0.5 * np.column_stack([graph.node_w, graph.node_w, graph.node_h]).astype(np.float64)
    bmin = xyz - half
    bmax = xyz + half
    return np.all((bmax >= box[0] - 1e-7) & (bmin <= box[1] + 1e-7), axis=1)


def _node_footprint_weights(graph: ToolpathGraph, loaded_nodes):
    n = graph.num_nodes
    trib = np.zeros(n, dtype=np.float64)
    if graph.num_beams:
        p0 = graph.node_xyz[graph.beam_edges[:, 0]]
        p1 = graph.node_xyz[graph.beam_edges[:, 1]]
        lens = np.linalg.norm(p1 - p0, axis=1)
        np.add.at(trib, graph.beam_edges[:, 0], 0.5 * lens)
        np.add.at(trib, graph.beam_edges[:, 1], 0.5 * lens)
    area = trib[loaded_nodes] * graph.node_w[loaded_nodes].astype(np.float64)
    if np.all(area <= 0):
        area = np.ones(len(loaded_nodes), dtype=np.float64)
    else:
        pos_med = float(np.median(area[area > 0]))
        area = np.where(area > 0, area, pos_med)
    return area


def graph_surface_load(graph: ToolpathGraph, loaded_nodes, force, load_point=None):
    """Distribute force and remote moment onto 6-DOF loaded nodes (translational DOFs)."""
    force = np.asarray(force, dtype=np.float64)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("force must be a finite 3-vector")
    if len(loaded_nodes) == 0:
        raise ValueError("Load selection has zero nodes")

    p = graph.node_xyz[loaded_nodes]
    raw_w = _node_footprint_weights(graph, loaded_nodes)
    weights = raw_w / raw_w.sum()
    centroid = weights @ p
    f_loc = weights[:, None] * force
    if load_point is not None:
        point = np.asarray(load_point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("load_point must be a finite 3-vector")
        r = p - centroid
        inertia = np.eye(3) * np.sum(weights * np.sum(r * r, axis=1)) - r.T @ (weights[:, None] * r)
        moment = np.cross(point - centroid, force)
        if np.linalg.matrix_rank(inertia) < 3:
            raise ValueError("Load surface cannot transmit a general moment")
        omega = np.linalg.solve(inertia, moment)
        f_loc += weights[:, None] * np.cross(omega, r)

    f_full = np.zeros((graph.num_nodes, 6), dtype=np.float64)
    f_full[loaded_nodes, :3] = f_loc
    return f_full.ravel()


def matvec_elements_f64(graph: ToolpathGraph, material: Material, seam_cfg: SeamConfig, u_64, chunk_size=32768):
    """Exact fp64 streaming element-by-element matrix-vector product (zero global matrix)."""
    u_nodes = np.asarray(u_64, dtype=np.float64).reshape(graph.num_nodes, 6)
    y_nodes = np.zeros((graph.num_nodes, 6), dtype=np.float64)

    # Beams
    beam_sc = _beam_scale_factors(graph.beam_flags, seam_cfg)
    mb = graph.num_beams
    for start in range(0, mb, chunk_size):
        stop = min(mb, start + chunk_size)
        e = graph.beam_edges[start:stop]
        Ke = timoshenko_beam_stiffness_batch(
            graph.node_xyz[e[:, 0]],
            graph.node_xyz[e[:, 1]],
            graph.beam_w[start:stop],
            graph.beam_h[start:stop],
            material,
            scale=beam_sc[start:stop],
        )
        ue = np.concatenate([u_nodes[e[:, 0]], u_nodes[e[:, 1]]], axis=1)
        fe = np.einsum("eij,ej->ei", Ke, ue)
        np.add.at(y_nodes, e[:, 0], fe[:, :6])
        np.add.at(y_nodes, e[:, 1], fe[:, 6:])

    # Bonds
    mk = graph.num_bonds
    if mk:
        bond_fac = _bond_scale_factors(graph.bond_type, graph.bond_dt, seam_cfg)
        e_b, g1, g2 = _bond_moduli(graph.bond_type, material)
        for start in range(0, mk, chunk_size):
            stop = min(mk, start + chunk_size)
            e = graph.bond_edges[start:stop]
            geom = graph.bond_geom[start:stop].astype(np.float64)
            Ke = eccentric_bond_stiffness_batch(
                graph.node_xyz[e[:, 0]],
                graph.node_xyz[e[:, 1]],
                graph.bond_pc[start:stop],
                graph.bond_normal[start:stop],
                graph.bond_t1[start:stop],
                graph.bond_t2[start:stop],
                geom[:, 0],
                geom[:, 1],
                geom[:, 2],
                geom[:, 3],
                e_b[start:stop],
                g1[start:stop],
                g2[start:stop],
                factor=bond_fac[start:stop],
            )
            ue = np.concatenate([u_nodes[e[:, 0]], u_nodes[e[:, 1]]], axis=1)
            fe = np.einsum("eij,ej->ei", Ke, ue)
            np.add.at(y_nodes, e[:, 0], fe[:, :6])
            np.add.at(y_nodes, e[:, 1], fe[:, 6:])

    return y_nodes.ravel()


def assemble_bsr6_f32(graph: ToolpathGraph, material: Material, seam_cfg: SeamConfig, fixed_mask_np, chunk_size=32768):
    """Assemble free-DOF 6x6 BSR matrix in float32 using bounded-memory chunks."""
    n = graph.num_nodes
    fixed_bool = fixed_mask_np != 0

    def _collect_free_edges(edges):
        if len(edges) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
        a = edges[:, 0]
        b = edges[:, 1]
        both_free = (~fixed_bool[a]) & (~fixed_bool[b]) & (a != b)
        af = a[both_free].astype(np.int64)
        bf = b[both_free].astype(np.int64)
        return np.r_[af, bf], np.r_[bf, af]

    br, bc = _collect_free_edges(graph.beam_edges)
    kr, kc = _collect_free_edges(graph.bond_edges)
    diag_idx = np.arange(n, dtype=np.int64)
    all_rows = np.concatenate([diag_idx, br, kr])
    all_cols = np.concatenate([diag_idx, bc, kc])
    keys = all_rows * np.int64(n) + all_cols
    uniq_keys = np.unique(keys)
    del all_rows, all_cols, br, bc, kr, kc, keys

    rows_u = (uniq_keys // np.int64(n)).astype(np.int32)
    cols_u = (uniq_keys % np.int64(n)).astype(np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(np.bincount(rows_u, minlength=n), out=indptr[1:])
    indices = cols_u
    nnz_blk = len(uniq_keys)

    diag_keys = diag_idx * np.int64(n) + diag_idx
    diag_pos = np.searchsorted(uniq_keys, diag_keys).astype(np.int32)
    del diag_idx, diag_keys, rows_u

    data = np.zeros((nnz_blk, 36), dtype=np.float32)
    # Fixed nodes get I_6 on diagonal for clean indexing
    if fixed_bool.any():
        data[diag_pos[fixed_bool]] = np.eye(6, dtype=np.float32).ravel()

    def _scatter_elements(edges, ke_f64):
        a = edges[:, 0].astype(np.int64)
        b = edges[:, 1].astype(np.int64)
        free_a = ~fixed_bool[a]
        free_b = ~fixed_bool[b]
        if free_a.any():
            np.add.at(data, diag_pos[a[free_a]], ke_f64[free_a, :6, :6].reshape(-1, 36).astype(np.float32))
        if free_b.any():
            np.add.at(data, diag_pos[b[free_b]], ke_f64[free_b, 6:, 6:].reshape(-1, 36).astype(np.float32))
        both = free_a & free_b & (a != b)
        if both.any():
            ab = a[both]
            bb = b[both]
            p_ab = np.searchsorted(uniq_keys, ab * np.int64(n) + bb)
            p_ba = np.searchsorted(uniq_keys, bb * np.int64(n) + ab)
            np.add.at(data, p_ab, ke_f64[both, :6, 6:].reshape(-1, 36).astype(np.float32))
            np.add.at(data, p_ba, ke_f64[both, 6:, :6].reshape(-1, 36).astype(np.float32))

    beam_sc = _beam_scale_factors(graph.beam_flags, seam_cfg)
    mb = graph.num_beams
    for start in range(0, mb, chunk_size):
        stop = min(mb, start + chunk_size)
        e = graph.beam_edges[start:stop]
        Ke = timoshenko_beam_stiffness_batch(
            graph.node_xyz[e[:, 0]],
            graph.node_xyz[e[:, 1]],
            graph.beam_w[start:stop],
            graph.beam_h[start:stop],
            material,
            scale=beam_sc[start:stop],
        )
        _scatter_elements(e, Ke)

    mk = graph.num_bonds
    if mk:
        bond_fac = _bond_scale_factors(graph.bond_type, graph.bond_dt, seam_cfg)
        e_b, g1, g2 = _bond_moduli(graph.bond_type, material)
        for start in range(0, mk, chunk_size):
            stop = min(mk, start + chunk_size)
            e = graph.bond_edges[start:stop]
            geom = graph.bond_geom[start:stop].astype(np.float64)
            Ke = eccentric_bond_stiffness_batch(
                graph.node_xyz[e[:, 0]],
                graph.node_xyz[e[:, 1]],
                graph.bond_pc[start:stop],
                graph.bond_normal[start:stop],
                graph.bond_t1[start:stop],
                graph.bond_t2[start:stop],
                geom[:, 0],
                geom[:, 1],
                geom[:, 2],
                geom[:, 3],
                e_b[start:stop],
                g1[start:stop],
                g2[start:stop],
                factor=bond_fac[start:stop],
            )
            _scatter_elements(e, Ke)

    data_3d = data.reshape(nnz_blk, 6, 6)
    diag_blocks = data_3d[diag_pos].astype(np.float64)
    # Regularize diagonal slightly if a free node had tiny numerical zero, invert safely
    d_inv = np.linalg.inv(diag_blocks).astype(np.float32)
    d_inv = 0.5 * (d_inv + d_inv.transpose(0, 2, 1))
    return indptr, indices, data_3d, d_inv, uniq_keys, diag_pos


def _connected_spatial_aggregates(node_xyz, fixed_mask_np, indptr, indices, box_mm):
    """Partition free nodes into connected spatial cell aggregates (never bridge holes/seams)."""
    n = len(node_xyz)
    free_bool = fixed_mask_np == 0
    free_idx = np.flatnonzero(free_bool)
    if len(free_idx) == 0:
        return np.full(n, -1, dtype=np.int32), np.zeros((0, 3), dtype=np.float64)

    origin = node_xyz[free_idx].min(axis=0)
    cell_ijk = np.floor((node_xyz[free_idx] - origin[None, :]) / box_mm).astype(np.int64)
    dims = cell_ijk.max(axis=0) + 1
    cell_key_free = cell_ijk[:, 2] * (dims[0] * dims[1]) + cell_ijk[:, 1] * dims[0] + cell_ijk[:, 0]
    cell_key = np.full(n, -1, dtype=np.int64)
    cell_key[free_idx] = cell_key_free

    # Keep only edges strictly inside the same spatial box on free nodes
    rows_all = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    cols_all = indices
    same_box = (rows_all != cols_all) & (cell_key[rows_all] >= 0) & (cell_key[rows_all] == cell_key[cols_all])
    r_in = rows_all[same_box]
    c_in = cols_all[same_box]
    g_in = coo_matrix((np.ones(len(r_in), dtype=np.int8), (r_in, c_in)), shape=(n, n)).tocsr()
    _, comp_labels = connected_components(g_in, directed=False, return_labels=True)

    # Remap components on free nodes to compact 0..M-1
    free_labels = comp_labels[free_idx]
    _, agg_compact = np.unique(free_labels, return_inverse=True)
    agg_id = np.full(n, -1, dtype=np.int32)
    agg_id[free_idx] = agg_compact.astype(np.int32)
    n_agg = int(agg_compact.max()) + 1

    counts = np.bincount(agg_compact, minlength=n_agg).astype(np.float64)
    agg_xyz = np.zeros((n_agg, 3), dtype=np.float64)
    for d in range(3):
        np.add.at(agg_xyz[:, d], agg_compact, node_xyz[free_idx, d])
    agg_xyz /= counts[:, None]
    return agg_id, agg_xyz


def _galerkin_coarsen_bsr6(indptr, indices, data_3d, node_xyz, fixed_mask_np, agg_id, agg_xyz, out_fp64=False):
    """Exact Galerkin coarse matrix K_c = P^T K P in 6x6 BSR."""
    n = len(node_xyz)
    n_agg = len(agg_xyz)
    r_ij = np.zeros((n, 3), dtype=np.float64)
    free_mask = agg_id >= 0
    r_ij[free_mask] = node_xyz[free_mask] - agg_xyz[agg_id[free_mask]]

    rows_all = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))
    cols_all = indices
    valid = (agg_id[rows_all] >= 0) & (agg_id[cols_all] >= 0)
    v_idx = np.flatnonzero(valid)

    r_fine = rows_all[v_idx]
    c_fine = cols_all[v_idx]
    r_agg = agg_id[r_fine].astype(np.int64)
    c_agg = agg_id[c_fine].astype(np.int64)

    c_keys = r_agg * np.int64(n_agg) + c_agg
    uniq_ckeys, inv_c = np.unique(c_keys, return_inverse=True)
    nnz_c = len(uniq_ckeys)
    c_rows = (uniq_ckeys // np.int64(n_agg)).astype(np.int32)
    c_cols = (uniq_ckeys % np.int64(n_agg)).astype(np.int32)
    c_indptr = np.zeros(n_agg + 1, dtype=np.int32)
    np.cumsum(np.bincount(c_rows, minlength=n_agg), out=c_indptr[1:])
    c_indices = c_cols

    def _skew_3(r):
        S = np.zeros((len(r), 3, 3), dtype=np.float64)
        S[:, 0, 1] = -r[:, 2]
        S[:, 0, 2] = r[:, 1]
        S[:, 1, 0] = r[:, 2]
        S[:, 1, 2] = -r[:, 0]
        S[:, 2, 0] = -r[:, 1]
        S[:, 2, 1] = r[:, 0]
        return S

    c_data = np.zeros((nnz_c, 6, 6), dtype=np.float64)
    chunk = 65536
    for start in range(0, len(v_idx), chunk):
        stop = min(len(v_idx), start + chunk)
        sl = v_idx[start:stop]
        dest = inv_c[start:stop]
        K = data_3d[sl].astype(np.float64)
        Si = _skew_3(r_ij[r_fine[start:stop]])
        Sj = _skew_3(r_ij[c_fine[start:stop]])
        K00 = K[:, :3, :3]
        K01 = K[:, :3, 3:]
        K10 = K[:, 3:, :3]
        K11 = K[:, 3:, 3:]
        C = np.empty_like(K)
        C[:, :3, :3] = K00
        C[:, :3, 3:] = K01 - K00 @ Sj
        C[:, 3:, :3] = K10 + Si @ K00
        C[:, 3:, 3:] = K11 + Si @ K01 - K10 @ Sj - Si @ K00 @ Sj
        np.add.at(c_data, dest, C)

    c_diag_pos = np.searchsorted(uniq_ckeys, np.arange(n_agg, dtype=np.int64) * np.int64(n_agg + 1))
    diag_blk = c_data[c_diag_pos]
    d_inv = np.linalg.inv(diag_blk).astype(np.float32)
    d_inv = 0.5 * (d_inv + d_inv.transpose(0, 2, 1))
    out_data = c_data if out_fp64 else c_data.astype(np.float32)
    return c_indptr, c_indices, out_data, d_inv, r_ij.astype(np.float32)


@dataclass
class _BSR6Level:
    n_nodes: int
    n_dofs: int
    fixed_mask_np: np.ndarray
    indptr_dev: object
    indices_dev: object
    data_dev: object
    d_inv_dev: object
    fixed_mask_dev: object
    lambda_max: float
    agg_id_dev: object = None
    r_ij_dev: object = None
    x_dev: object = None
    b_dev: object = None
    r_dev: object = None
    d_dev: object = None
    ax_dev: object = None


def _launch_bsr6_matvec(lvl: _BSR6Level, x_dev, y_dev, device):
    wp.launch(
        _bsr6_matvec_f32_kernel,
        dim=lvl.n_nodes,
        inputs=[lvl.n_nodes, lvl.indptr_dev, lvl.indices_dev, lvl.data_dev, lvl.fixed_mask_dev, x_dev, y_dev],
        device=device,
    )


def _estimate_lambda_max_6(lvl: _BSR6Level, scratch_256, tmp_x, tmp_y, device, n_power=8):
    rng = np.random.default_rng(20261119)
    v0 = rng.normal(size=lvl.n_dofs).astype(np.float32)
    v0[np.repeat(lvl.fixed_mask_np != 0, 6)] = 0.0
    nrm = float(np.linalg.norm(v0))
    if nrm <= 0:
        return 2.0
    v0 /= nrm
    wp.copy(tmp_x, wp.from_numpy(v0, dtype=wp.float32))
    lam = 2.0
    for _ in range(n_power):
        _launch_bsr6_matvec(lvl, tmp_x, tmp_y, device)
        wp.launch(
            _block6_precond_f32_kernel,
            dim=lvl.n_nodes,
            inputs=[lvl.fixed_mask_dev, lvl.d_inv_dev, tmp_y, tmp_y],
            device=device,
        )
        lam = _gpu_dot_f32(tmp_x, tmp_y, scratch_256, device)
        nrm2 = _gpu_dot_f32(tmp_y, tmp_y, scratch_256, device)
        if nrm2 <= 0:
            break
        inv_n = np.float32(1.0 / np.sqrt(nrm2))
        wp.launch(_axpby_6_f32_kernel, dim=lvl.n_dofs, inputs=[inv_n, tmp_y, np.float32(0.0), tmp_y, tmp_x], device=device)
    return max(float(lam) * 1.15, 0.5)


def _chebyshev_smooth_6(lvl: _BSR6Level, b_dev, x_dev, r_dev, d_dev, ax_dev, degree, device, zero_init=False):
    lam_max = lvl.lambda_max
    lam_min = 0.12 * lam_max
    c_val = 0.5 * (lam_max + lam_min)
    d_val = 0.5 * (lam_max - lam_min)
    if zero_init:
        x_dev.zero_()
        wp.copy(r_dev, b_dev)
    else:
        _launch_bsr6_matvec(lvl, x_dev, ax_dev, device)
        wp.launch(_axpby_6_f32_kernel, dim=lvl.n_dofs, inputs=[np.float32(1.0), b_dev, np.float32(-1.0), ax_dev, r_dev], device=device)
    wp.launch(_apply_fixed_6_f32_kernel, dim=lvl.n_nodes, inputs=[lvl.fixed_mask_dev, r_dev], device=device)
    wp.launch(
        _block6_cheb_init_f32_kernel,
        dim=lvl.n_nodes,
        inputs=[np.float32(1.0 / c_val), lvl.fixed_mask_dev, lvl.d_inv_dev, r_dev, d_dev, x_dev],
        device=device,
    )
    if degree <= 1:
        return
    two_sigma = 2.0 * c_val / max(d_val, 1e-12)
    w_prev = d_val / c_val
    for _k in range(1, degree):
        _launch_bsr6_matvec(lvl, x_dev, ax_dev, device)
        wp.launch(_axpby_6_f32_kernel, dim=lvl.n_dofs, inputs=[np.float32(1.0), b_dev, np.float32(-1.0), ax_dev, r_dev], device=device)
        wp.launch(_apply_fixed_6_f32_kernel, dim=lvl.n_nodes, inputs=[lvl.fixed_mask_dev, r_dev], device=device)
        w_cur = 1.0 / (two_sigma - w_prev)
        alpha = (2.0 / max(d_val, 1e-12)) * w_cur
        beta = w_prev * w_cur
        w_prev = w_cur
        wp.launch(
            _block6_cheb_step_f32_kernel,
            dim=lvl.n_nodes,
            inputs=[np.float32(alpha), np.float32(beta), lvl.fixed_mask_dev, lvl.d_inv_dev, r_dev, d_dev, x_dev],
            device=device,
        )


def _rigid_modes_6(xyz):
    p = xyz - xyz.mean(axis=0)
    scale = max(float(np.ptp(p, axis=0).max()), 1e-15)
    p = p / scale
    R = np.zeros((len(xyz), 6, 6), dtype=np.float64)
    R[:, :3, :3] = np.eye(3)
    R[:, 3:, 3:] = np.eye(3) / scale
    for ax in range(3):
        R[:, :3, ax + 3] = np.cross(np.eye(3)[ax], p)
    return R.reshape(-1, 6)


def solve_toolpath_graph_gpu(
    graph: ToolpathGraph,
    material=None,
    seam_cfg: SeamConfig = None,
    fixed="X-",
    loaded="X+",
    force=(0.0, 0.0, -10.0),
    load_point=None,
    rtol=1e-7,
    max_outer_ir=20,
    max_inner_iter=250,
    coarse_dof_target=30000,
    smooth_degree=3,
    device="auto",
):
    """Solve 6-DOF filament-bond network on GPU with Galerkin aggregation MG + fp64 IR."""
    t_total = time.perf_counter()
    material = material or Material()
    seam_cfg = seam_cfg or SeamConfig()
    force = np.asarray(force, dtype=np.float64)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("force must be a finite 3-vector")
    if not 0 < rtol <= 1e-4:
        raise ValueError("rtol must be in (0, 1e-4]")
    used_device = _get_device(device)

    t_setup = time.perf_counter()
    fixed_mask_bool = select_graph_nodes(graph, fixed)
    loaded_mask_bool = select_graph_nodes(graph, loaded)
    if not fixed_mask_bool.any() or not loaded_mask_bool.any():
        raise ValueError("固定或加载选区没有选中走线节点")
    fixed_nodes = np.flatnonzero(fixed_mask_bool)
    loaded_nodes = np.flatnonzero(loaded_mask_bool)
    if np.intersect1d(fixed_nodes, loaded_nodes).size:
        raise ValueError("固定区域与加载区域共享走线节点，请分开选区")

    f_64 = graph_surface_load(graph, loaded_nodes, force, load_point=load_point)
    fixed_mask_np = fixed_mask_bool.astype(np.uint8)
    free_dof_mask = np.repeat(fixed_mask_np == 0, 6)
    free_dofs_idx = np.flatnonzero(free_dof_mask)

    n_nodes = graph.num_nodes
    n_dofs = 6 * n_nodes
    f_norm = float(np.linalg.norm(f_64[free_dof_mask]))

    # Fast exact path for small graphs (<= coarse_dof_target DOFs)
    if len(free_dofs_idx) <= coarse_dof_target:
        indptr0, indices0, data0, d_inv0, _, _ = assemble_bsr6_f32(graph, material, seam_cfg, fixed_mask_np)
        # Assemble exact fp64 BSR from elements or directly solve via fp64 elements? Let's assemble fp64 CSR for small graph!
        # Wait: assemble_bsr6_f32 already assembled float32; for small graphs let's assemble fp64 CSR directly so direct solve is machine-exact.
        rows_list, cols_list, vals_list = [], [], []
        beam_sc = _beam_scale_factors(graph.beam_flags, seam_cfg)
        if graph.num_beams:
            e = graph.beam_edges
            Ke_b = timoshenko_beam_stiffness_batch(
                graph.node_xyz[e[:, 0]], graph.node_xyz[e[:, 1]], graph.beam_w, graph.beam_h, material, scale=beam_sc
            )
            dofs = np.column_stack([e[:, 0:1] * 6 + np.arange(6), e[:, 1:2] * 6 + np.arange(6)])
            rows_list.append(np.repeat(dofs, 12, axis=1).ravel())
            cols_list.append(np.tile(dofs, (1, 12)).ravel())
            vals_list.append(Ke_b.ravel())
        if graph.num_bonds:
            e = graph.bond_edges
            geom = graph.bond_geom.astype(np.float64)
            e_b, g1, g2 = _bond_moduli(graph.bond_type, material)
            Ke_k = eccentric_bond_stiffness_batch(
                graph.node_xyz[e[:, 0]],
                graph.node_xyz[e[:, 1]],
                graph.bond_pc,
                graph.bond_normal,
                graph.bond_t1,
                graph.bond_t2,
                geom[:, 0],
                geom[:, 1],
                geom[:, 2],
                geom[:, 3],
                e_b,
                g1,
                g2,
                factor=_bond_scale_factors(graph.bond_type, graph.bond_dt, seam_cfg),
            )
            dofs = np.column_stack([e[:, 0:1] * 6 + np.arange(6), e[:, 1:2] * 6 + np.arange(6)])
            rows_list.append(np.repeat(dofs, 12, axis=1).ravel())
            cols_list.append(np.tile(dofs, (1, 12)).ravel())
            vals_list.append(Ke_k.ravel())
        K_csr = coo_matrix((np.concatenate(vals_list), (np.concatenate(rows_list), np.concatenate(cols_list))), shape=(n_dofs, n_dofs)).tocsr()
        u_64_host = np.zeros(n_dofs, dtype=np.float64)
        if f_norm > 0:
            u_64_host[free_dofs_idx] = spsolve(K_csr[free_dofs_idx][:, free_dofs_idx], f_64[free_dofs_idx])
        y_final = matvec_elements_f64(graph, material, seam_cfg, u_64_host)
        setup_seconds = time.perf_counter() - t_setup
        solve_seconds = 0.0
        levels_summary = [{"level": 0, "nodes": n_nodes, "free_dofs": int(len(free_dofs_idx)), "nnz_blocks": int(len(indices0))}]
        outer_iters = 1
        total_inner_iters = 0
        residual_history = [float(np.linalg.norm((y_final - f_64)[free_dof_mask]) / max(f_norm, 1e-20))]
        coarse_solver_name = "spsolve_direct"
        used_jacobi_fallback = False
        vram_used_mib = vram_peak_mib = vram_free_mib = None
    else:
        indptr0, indices0, data0, d_inv0, _, _ = assemble_bsr6_f32(graph, material, seam_cfg, fixed_mask_np)
        cg_x = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
        cg_r = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
        cg_p = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
        cg_ap = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
        cg_z = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
        cg_d_tmp = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
        cg_v_r = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
        scratch_256 = wp.zeros(256, dtype=wp.float64, device=used_device)

        lvl0 = _BSR6Level(
            n_nodes=n_nodes,
            n_dofs=n_dofs,
            fixed_mask_np=fixed_mask_np,
            indptr_dev=wp.array(indptr0, dtype=wp.int32, device=used_device),
            indices_dev=wp.array(indices0, dtype=wp.int32, device=used_device),
            data_dev=wp.array(data0, dtype=wp.float32, device=used_device),
            d_inv_dev=wp.array(d_inv0, dtype=wp.float32, device=used_device),
            fixed_mask_dev=wp.array(fixed_mask_np, dtype=wp.uint8, device=used_device),
            lambda_max=2.0,
        )
        lvl0.lambda_max = _estimate_lambda_max_6(lvl0, scratch_256, cg_x, cg_r, used_device)
        levels = [lvl0]
        levels_summary = [
            {"level": 0, "nodes": n_nodes, "free_dofs": int(len(free_dofs_idx)), "nnz_blocks": int(len(indices0)), "lambda_max": float(lvl0.lambda_max)}
        ]

        # Build up to 2 Galerkin connected-aggregate levels until coarsest <= coarse_dof_target
        cur_xyz = graph.node_xyz
        cur_fixed = fixed_mask_np
        cur_indptr, cur_indices, cur_data = indptr0, indices0, data0
        box_mm = 2.0
        for lvl_idx in (1, 2):
            cur_free = int(6 * (cur_fixed == 0).sum())
            if cur_free <= coarse_dof_target:
                break
            agg_id, agg_xyz = _connected_spatial_aggregates(cur_xyz, cur_fixed, cur_indptr, cur_indices, box_mm)
            c_free_dofs = 6 * len(agg_xyz)
            if c_free_dofs == 0 or c_free_dofs >= int(cur_free * 0.7):
                box_mm *= 2.0
                continue
            levels[-1].agg_id_dev = wp.array(agg_id, dtype=wp.int32, device=used_device)
            need_more = (c_free_dofs > coarse_dof_target) and (lvl_idx == 1)
            c_indptr, c_indices, c_data, c_d_inv, r_ij_f32 = _galerkin_coarsen_bsr6(
                cur_indptr, cur_indices, cur_data, cur_xyz, cur_fixed, agg_id, agg_xyz, out_fp64=not need_more
            )
            levels[-1].r_ij_dev = wp.array(r_ij_f32, dtype=wp.float32, device=used_device)
            c_fixed = np.zeros(len(agg_xyz), dtype=np.uint8)
            c_nd = 6 * len(agg_xyz)

            c_lvl = _BSR6Level(
                n_nodes=len(agg_xyz),
                n_dofs=c_nd,
                fixed_mask_np=c_fixed,
                indptr_dev=wp.array(c_indptr, dtype=wp.int32, device=used_device),
                indices_dev=wp.array(c_indices, dtype=wp.int32, device=used_device),
                data_dev=wp.array(c_data.astype(np.float32, copy=False), dtype=wp.float32, device=used_device),
                d_inv_dev=wp.array(c_d_inv, dtype=wp.float32, device=used_device),
                fixed_mask_dev=wp.array(c_fixed, dtype=wp.uint8, device=used_device),
                lambda_max=2.0,
                x_dev=wp.zeros(c_nd, dtype=wp.float32, device=used_device),
                b_dev=wp.zeros(c_nd, dtype=wp.float32, device=used_device),
                r_dev=wp.zeros(c_nd, dtype=wp.float32, device=used_device),
                d_dev=wp.zeros(c_nd, dtype=wp.float32, device=used_device),
                ax_dev=wp.zeros(c_nd, dtype=wp.float32, device=used_device),
            )
            c_lvl.lambda_max = _estimate_lambda_max_6(c_lvl, scratch_256, c_lvl.x_dev, c_lvl.r_dev, used_device)
            levels.append(c_lvl)
            levels_summary.append(
                {
                    "level": lvl_idx,
                    "nodes": len(agg_xyz),
                    "free_dofs": int(c_nd),
                    "nnz_blocks": int(len(c_indices)),
                    "box_mm": float(box_mm),
                    "lambda_max": float(c_lvl.lambda_max),
                }
            )
            cur_xyz, cur_fixed = agg_xyz, c_fixed
            cur_indptr, cur_indices, cur_data = c_indptr, c_indices, c_data
            box_mm = 6.0

        # Release huge Level-0 CPU BSR arrays once uploaded to GPU!
        del indptr0, indices0, data0, d_inv0

        # Setup coarsest-level exact splu or PyAMG V-cycle
        coarsest = levels[-1]
        coarse_lu = None
        coarse_precond = None
        coarse_scale = None
        coarse_solver_name = "none"
        if len(levels) > 1:
            K_c = bsr_matrix(
                (cur_data.astype(np.float64, copy=False), cur_indices, cur_indptr),
                shape=(coarsest.n_dofs, coarsest.n_dofs),
                blocksize=(6, 6),
            )
            if coarsest.n_dofs <= 35000:
                coarse_lu = splu(K_c.tocsc())
                coarse_solver_name = "splu"
            else:
                import pyamg

                A_csc = K_c.tocsr()
                diag_c = A_csc.diagonal()
                coarse_scale = 1.0 / np.sqrt(np.maximum(diag_c, 1e-30))
                S = sp_diags(coarse_scale)
                A_scaled = (S @ A_csc @ S).tocsr().tobsr(blocksize=(6, 6))
                B_rbm = _rigid_modes_6(cur_xyz) / coarse_scale[:, None]
                ml = pyamg.smoothed_aggregation_solver(A_scaled, B=B_rbm, symmetry="symmetric", smooth="energy", max_coarse=120)
                coarse_precond = ml.aspreconditioner(cycle="V")
                coarse_solver_name = "pyamg_vcycle"
            del K_c

        def _solve_coarsest(b_dev, x_dev):
            rhs = b_dev.numpy().astype(np.float64)
            if coarse_lu is not None:
                sol = coarse_lu.solve(rhs)
            elif coarse_precond is not None:
                sol = coarse_scale * coarse_precond(coarse_scale * rhs)
            else:
                sol = np.zeros_like(rhs)
            wp.copy(x_dev, wp.from_numpy(sol.astype(np.float32), dtype=wp.float32))

        def _vcycle(lvl_idx, b_dev, x_dev, r_dev, d_dev, ax_dev):
            if lvl_idx == len(levels) - 1:
                _solve_coarsest(b_dev, x_dev)
                return
            lvl = levels[lvl_idx]
            _chebyshev_smooth_6(lvl, b_dev, x_dev, r_dev, d_dev, ax_dev, smooth_degree, used_device, zero_init=True)
            _launch_bsr6_matvec(lvl, x_dev, ax_dev, used_device)
            wp.launch(_axpby_6_f32_kernel, dim=lvl.n_dofs, inputs=[np.float32(1.0), b_dev, np.float32(-1.0), ax_dev, r_dev], device=used_device)
            wp.launch(_apply_fixed_6_f32_kernel, dim=lvl.n_nodes, inputs=[lvl.fixed_mask_dev, r_dev], device=used_device)
            nxt = levels[lvl_idx + 1]
            nxt.b_dev.zero_()
            wp.launch(_agg_restrict_f32_kernel, dim=lvl.n_nodes, inputs=[lvl.agg_id_dev, lvl.r_ij_dev, r_dev, nxt.b_dev], device=used_device)
            _vcycle(lvl_idx + 1, nxt.b_dev, nxt.x_dev, nxt.r_dev, nxt.d_dev, nxt.ax_dev)
            wp.launch(_agg_prolong_add_f32_kernel, dim=lvl.n_nodes, inputs=[lvl.agg_id_dev, lvl.r_ij_dev, nxt.x_dev, x_dev], device=used_device)
            _chebyshev_smooth_6(lvl, b_dev, x_dev, r_dev, d_dev, ax_dev, smooth_degree, used_device, zero_init=False)

        setup_seconds = time.perf_counter() - t_setup
        t_solve = time.perf_counter()
        u_64_host = np.zeros(n_dofs, dtype=np.float64)
        outer_iters = 0
        total_inner_iters = 0
        residual_history = []
        used_jacobi_fallback = False

        if f_norm > 0.0:
            for outer in range(max_outer_ir):
                outer_iters = outer + 1
                if outer == 0:
                    r_64 = f_64.copy()
                else:
                    y_64 = matvec_elements_f64(graph, material, seam_cfg, u_64_host)
                    r_64 = f_64 - y_64
                r_64[~free_dof_mask] = 0.0
                r_norm = float(np.linalg.norm(r_64[free_dof_mask]))
                rel_res = r_norm / f_norm
                residual_history.append(rel_res)
                if rel_res <= rtol:
                    break

                r_scaled = (r_64 / r_norm).astype(np.float32)
                cg_x.zero_()
                wp.copy(cg_r, wp.from_numpy(r_scaled, dtype=wp.float32))
                r0_f32 = math.sqrt(max(_gpu_dot_f32(cg_r, cg_r, scratch_256, used_device), 1e-30))
                inner_target = max(min(2e-4, rtol / max(rel_res, 1e-30) * 0.1), 2e-5)

                def _apply_precond(rhs_d, z_d):
                    if len(levels) > 1 and not used_jacobi_fallback:
                        _vcycle(0, rhs_d, z_d, cg_v_r, cg_d_tmp, cg_ap)
                    else:
                        wp.launch(
                            _block6_precond_f32_kernel,
                            dim=lvl0.n_nodes,
                            inputs=[lvl0.fixed_mask_dev, lvl0.d_inv_dev, rhs_d, z_d],
                            device=used_device,
                        )

                _apply_precond(cg_r, cg_z)
                rz_old = _gpu_dot_f32(cg_r, cg_z, scratch_256, used_device)
                if rz_old <= 0.0:
                    used_jacobi_fallback = True
                    _apply_precond(cg_r, cg_z)
                    rz_old = _gpu_dot_f32(cg_r, cg_z, scratch_256, used_device)
                wp.copy(cg_p, cg_z)

                for _it in range(1, max_inner_iter + 1):
                    total_inner_iters += 1
                    _launch_bsr6_matvec(lvl0, cg_p, cg_ap, used_device)
                    pAp = _gpu_dot_f32(cg_p, cg_ap, scratch_256, used_device)
                    if pAp <= 0.0:
                        break
                    alpha = np.float32(rz_old / pAp)
                    wp.launch(_axpby_6_f32_kernel, dim=n_dofs, inputs=[np.float32(1.0), cg_x, alpha, cg_p, cg_x], device=used_device)
                    wp.launch(_axpby_6_f32_kernel, dim=n_dofs, inputs=[np.float32(1.0), cg_r, np.float32(-alpha), cg_ap, cg_r], device=used_device)
                    r_cur = math.sqrt(max(_gpu_dot_f32(cg_r, cg_r, scratch_256, used_device), 0.0))
                    if r_cur / r0_f32 <= inner_target:
                        break
                    _apply_precond(cg_r, cg_z)
                    rz_new = _gpu_dot_f32(cg_r, cg_z, scratch_256, used_device)
                    if rz_new <= 0.0:
                        break
                    beta = np.float32(rz_new / rz_old)
                    wp.launch(_axpby_6_f32_kernel, dim=n_dofs, inputs=[np.float32(1.0), cg_z, beta, cg_p, cg_p], device=used_device)
                    rz_old = rz_new

                u_64_host[free_dof_mask] += cg_x.numpy().astype(np.float64)[free_dof_mask] * r_norm

        solve_seconds = time.perf_counter() - t_solve

        vram_used_mib = vram_peak_mib = vram_free_mib = None
        if used_device.startswith("cuda"):
            try:
                vram_used_mib = float(wp.get_mempool_used_mem_current(used_device) / 1024**2)
                vram_peak_mib = float(wp.get_mempool_used_mem_high(used_device) / 1024**2)
                vram_free_mib = float(wp.get_device(used_device).free_memory / 1024**2)
            except Exception:
                pass

        del levels, lvl0, cg_x, cg_r, cg_p, cg_ap, cg_z, cg_d_tmp, cg_v_r, scratch_256
        if used_device.startswith("cuda"):
            try:
                wp.synchronize_device(used_device)
            except Exception:
                pass

        y_final = matvec_elements_f64(graph, material, seam_cfg, u_64_host)

    # Final strict fp64 residual + global force/moment equilibrium check (including fixed reaction moments!)
    residual_vec = y_final - f_64
    final_rel_res = float(np.linalg.norm(residual_vec[free_dof_mask]) / max(f_norm, 1e-20))
    if not np.isfinite(u_64_host).all() or final_rel_res > 1e-6:
        raise ValueError(f"走线网络求解未通过原始系统 fp64 残差校核：relative_residual={final_rel_res:.3g}")

    reactions_6 = residual_vec.reshape(n_nodes, 6)[fixed_nodes]
    reaction_force_sum = reactions_6[:, :3].sum(axis=0)
    rel_force_err = float(np.linalg.norm(reaction_force_sum + force) / max(np.linalg.norm(force), 1.0))

    ref_center = graph.node_xyz.mean(axis=0)
    f_loaded = f_64.reshape(n_nodes, 6)[loaded_nodes]
    moment_load = np.cross(graph.node_xyz[loaded_nodes] - ref_center, f_loaded[:, :3]).sum(axis=0) + f_loaded[:, 3:].sum(axis=0)
    moment_react = np.cross(graph.node_xyz[fixed_nodes] - ref_center, reactions_6[:, :3]).sum(axis=0) + reactions_6[:, 3:].sum(axis=0)
    extent = np.ptp(graph.node_xyz, axis=0)
    rel_moment_err = float(np.linalg.norm(moment_load + moment_react) / max(np.linalg.norm(force) * float(extent.max()), 1.0))
    if rel_force_err >= 1e-6 or rel_moment_err >= 1e-6:
        raise ValueError(
            f"走线网络求解未通过全局平衡校核：force_err={rel_force_err:.3g}, moment_err={rel_moment_err:.3g}"
        )

    u_6 = u_64_host.reshape(n_nodes, 6)
    disp_xyz = u_6[:, :3]
    max_disp = float(np.linalg.norm(disp_xyz, axis=1).max())
    strain_energy = float(0.5 * np.dot(u_64_host, f_64))
    force_norm_val = float(np.linalg.norm(force))
    work_conj_disp = float(2.0 * strain_energy / force_norm_val) if force_norm_val > 0 else None
    compliance = float(2.0 * strain_energy / (force_norm_val ** 2)) if force_norm_val > 0 else None
    rss_mib = float(psutil.Process().memory_info().rss / 1024**2)

    solver_info = {
        "device": used_device,
        "nodes": n_nodes,
        "beams": graph.num_beams,
        "bonds": graph.num_bonds,
        "free_dofs": int(len(free_dofs_idx)),
        "levels": levels_summary,
        "outer_ir_iterations": outer_iters,
        "total_inner_pcg_iterations": total_inner_iters,
        "residual_history": residual_history,
        "relative_residual": final_rel_res,
        "coarse_solver": coarse_solver_name,
        "used_jacobi_fallback": used_jacobi_fallback,
        "setup_seconds": setup_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": time.perf_counter() - t_total,
        "force_relative_error": rel_force_err,
        "moment_relative_error": rel_moment_err,
        "strain_energy_Nmm": strain_energy,
        "work_conjugate_displacement_mm": work_conj_disp,
        "compliance_mm_per_N": compliance,
        "max_displacement_mm": max_disp,
        "reaction_N": reaction_force_sum.tolist(),
        "fixed_nodes_count": int(len(fixed_nodes)),
        "loaded_nodes_count": int(len(loaded_nodes)),
        "seam_config": seam_cfg.to_dict(),
        "process_rss_mib": rss_mib,
        "vram_reported_mib": vram_used_mib,
        "vram_peak_mib": vram_peak_mib,
        "vram_free_after_solve_mib": vram_free_mib,
    }
    return {
        "displacement_6dof": u_6,
        "displacement": disp_xyz,
        "fixed_nodes": fixed_nodes,
        "loaded_nodes": loaded_nodes,
        "reactions_6dof": reactions_6,
        "forces_6dof": f_64.reshape(n_nodes, 6),
        "solver_info": solver_info,
    }


def recover_graph_stress_streaming(
    graph: ToolpathGraph,
    displacement_6dof,
    material=None,
    seam_cfg: SeamConfig = None,
    chunk_size=32768,
    spill_path=None,
):
    """Stream beams & eccentric bonds to compute feature/seam/start-stop stress audits."""
    t0 = time.perf_counter()
    material = material or Material()
    seam_cfg = seam_cfg or SeamConfig()
    u_nodes = np.asarray(displacement_6dof, dtype=np.float64)

    # Beams
    mb = graph.num_beams
    beam_sc = _beam_scale_factors(graph.beam_flags, seam_cfg)
    b_vm = np.empty(mb, dtype=np.float32)
    b_ax = np.empty(mb, dtype=np.float32)
    b_fib = np.empty(mb, dtype=np.float32)
    for start in range(0, mb, chunk_size):
        stop = min(mb, start + chunk_size)
        e = graph.beam_edges[start:stop]
        ue = np.concatenate([u_nodes[e[:, 0]], u_nodes[e[:, 1]]], axis=1)
        st = beam_internal_stress_batch(
            graph.node_xyz[e[:, 0]],
            graph.node_xyz[e[:, 1]],
            graph.beam_w[start:stop],
            graph.beam_h[start:stop],
            material,
            ue,
            scale=beam_sc[start:stop],
        )
        b_vm[start:stop] = st["von_mises_max"].astype(np.float32)
        b_ax[start:stop] = st["sigma_axial"].astype(np.float32)
        b_fib[start:stop] = st["sigma_fiber_max"].astype(np.float32)

    # Bonds
    mk = graph.num_bonds
    k_sn = np.empty(mk, dtype=np.float32)
    k_sn_peak = np.empty(mk, dtype=np.float32)
    k_tau = np.empty(mk, dtype=np.float32)
    k_eq = np.empty(mk, dtype=np.float32)
    k_eq_peak = np.empty(mk, dtype=np.float32)

    if mk:
        bond_fac = _bond_scale_factors(graph.bond_type, graph.bond_dt, seam_cfg)
        e_b, g1, g2 = _bond_moduli(graph.bond_type, material)
        for start in range(0, mk, chunk_size):
            stop = min(mk, start + chunk_size)
            e = graph.bond_edges[start:stop]
            geom = graph.bond_geom[start:stop].astype(np.float64)
            ue = np.concatenate([u_nodes[e[:, 0]], u_nodes[e[:, 1]]], axis=1)
            st = bond_interface_stress_batch(
                graph.node_xyz[e[:, 0]],
                graph.node_xyz[e[:, 1]],
                graph.bond_pc[start:stop],
                graph.bond_normal[start:stop],
                graph.bond_t1[start:stop],
                graph.bond_t2[start:stop],
                geom[:, 1],
                geom[:, 2],
                geom[:, 3],
                e_b[start:stop],
                g1[start:stop],
                g2[start:stop],
                ue,
                factor=bond_fac[start:stop],
            )
            k_sn[start:stop] = st["sigma_n"].astype(np.float32)
            k_sn_peak[start:stop] = st["sigma_n_peak"].astype(np.float32)
            k_tau[start:stop] = st["tau"].astype(np.float32)
            k_eq[start:stop] = st["equiv_traction"].astype(np.float32)
            k_eq_peak[start:stop] = st["equiv_peak_traction"].astype(np.float32)

    if spill_path is not None:
        spill_p = Path(spill_path)
        spill_p.parent.mkdir(parents=True, exist_ok=True)
        with spill_p.open("wb") as fh:
            import struct

            fh.write(b"TPGST1\x00\x00" + struct.pack("<QQ", mb, mk))
            fh.write(np.column_stack([b_vm, b_ax, b_fib]).astype("<f4", copy=False).tobytes())
            if mk:
                fh.write(np.column_stack([k_sn, k_sn_peak, k_tau, k_eq, k_eq_peak]).astype("<f4", copy=False).tobytes())
    else:
        spill_p = None

    # Feature beam summary & start/stop zone comparison
    beam_by_feature = {}
    for fcode, fname in FEATURE_NAMES.items():
        msk = graph.beam_feature == fcode
        if not msk.any():
            continue
        beam_by_feature[fname] = {
            "beams": int(msk.sum()),
            "von_mises_max_MPa": float(b_vm[msk].max()),
            "von_mises_p95_MPa": float(np.percentile(b_vm[msk], 95)),
            "axial_max_abs_MPa": float(np.abs(b_ax[msk]).max()),
        }

    start_zone = (graph.beam_flags & NODE_START_ZONE) != 0
    stop_zone = (graph.beam_flags & NODE_STOP_ZONE) != 0
    interior = ~(start_zone | stop_zone)
    beam_zones = {}
    for zname, zmask in (("start_zone", start_zone), ("stop_zone", stop_zone), ("interior", interior)):
        if zmask.any():
            beam_zones[zname] = {
                "beams": int(zmask.sum()),
                "von_mises_max_MPa": float(b_vm[zmask].max()),
                "von_mises_p95_MPa": float(np.percentile(b_vm[zmask], 95)),
            }

    # Bond stress breakdown & top seam/start/stop hotspots
    bond_by_type = {}
    for bcode, bname in BOND_TYPE_NAMES.items():
        msk = graph.bond_type == bcode
        if not msk.any():
            continue
        bond_by_type[bname] = {
            "bonds": int(msk.sum()),
            "sigma_n_max_MPa": float(k_sn[msk].max()),
            "sigma_n_peak_max_MPa": float(k_sn_peak[msk].max()),
            "tau_max_MPa": float(k_tau[msk].max()),
            "equiv_traction_max_MPa": float(k_eq[msk].max()),
            "equiv_traction_p95_MPa": float(np.percentile(k_eq[msk], 95)),
            "equiv_peak_traction_max_MPa": float(k_eq_peak[msk].max()),
        }

    def _top_bond_hotspots(mask, score_arr, label, top_k=20):
        idx_all = np.flatnonzero(mask)
        if not len(idx_all):
            return []
        vals = score_arr[idx_all]
        k = min(top_k, len(idx_all))
        pick = idx_all[np.argpartition(vals, -k)[-k:]]
        pick = pick[np.argsort(score_arr[pick])[::-1]]
        return [
            {
                "bond_type": BOND_TYPE_NAMES.get(int(graph.bond_type[i]), "unknown"),
                label: float(score_arr[i]),
                "sigma_n_MPa": float(k_sn[i]),
                "tau_MPa": float(k_tau[i]),
                "xyz_mm": graph.bond_pc[i].tolist(),
            }
            for i in pick
        ]

    seam_mask = np.isin(
        graph.bond_type,
        [BOND_SEAM_CLOSED, BOND_SEAM_SCARF, BOND_START_ANCHOR, BOND_STOP_ANCHOR],
    )

    return {
        "beam_count": int(mb),
        "bond_count": int(mk),
        "beam_von_mises_max_MPa": float(b_vm.max()) if mb else 0.0,
        "beam_von_mises_p95_MPa": float(np.percentile(b_vm, 95)) if mb else 0.0,
        "bond_sigma_n_max_MPa": float(k_sn.max()) if mk else 0.0,
        "bond_tau_max_MPa": float(k_tau.max()) if mk else 0.0,
        "bond_equiv_traction_max_MPa": float(k_eq.max()) if mk else 0.0,
        "beam_by_feature": beam_by_feature,
        "beam_by_zone": beam_zones,
        "bond_by_type": bond_by_type,
        "top_seam_and_anchor_hotspots": _top_bond_hotspots(seam_mask, k_eq_peak, "equiv_peak_traction_MPa"),
        "top_interlayer_hotspots": _top_bond_hotspots(graph.bond_type == BOND_INTERLAYER, k_eq_peak, "equiv_peak_traction_MPa"),
        "recovery_seconds": time.perf_counter() - t0,
        "spill_path": str(spill_p.resolve()) if spill_p is not None else None,
        "_beam_vm": b_vm,
        "_bond_sn": k_sn,
        "_bond_eq_peak": k_eq_peak,
    }
