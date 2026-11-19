"""GPU-accelerated Matrix-Free H8 solver with hybrid GMG-PCG and mixed-precision IR."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import psutil
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spsolve

from hex_fem import (
    HEX_LOCAL_OFFSETS,
    angle_to_bin,
    bin_to_angle,
    build_stiffness_lut,
    hex_B_matrices,
    hex_centroid_stress,
)
from numerics import rigid_modes
from solid import Material, local_stiffness
from toolpath_bricks import (
    FEATURE_NAMES,
    FLAG_BEAD_INTERFACE,
    FLAG_SEAM_LOOP,
    FLAG_START_STOP_ZONE,
    BrickGrid,
    decode_feature,
    popcount_uint64,
    unpack_payload,
)


@dataclass
class HexInterfaceConfig:
    """Per-voxel interface stiffness scaling factors for H8 Route-2 weakening."""

    seam_factor: float = 1.0
    start_stop_factor: float = 1.0
    bead_interface_factor: float = 1.0
    interlayer_shear_factor: float = 1.0

    @classmethod
    def ideal(cls):
        return cls()

    @classmethod
    def default_weakened(cls):
        return cls(
            seam_factor=0.5,
            start_stop_factor=0.8,
            bead_interface_factor=0.85,
            interlayer_shear_factor=0.85,
        )

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("hex_interface_config must be a JSON object")
        d = dict(data)
        weakened = bool(d.pop("weakened", False))
        cfg = cls.default_weakened() if weakened else cls()
        valid = set(cls.__dataclass_fields__)
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"Unknown hex_interface_config keys: {sorted(unknown)}")
        for k, v in d.items():
            fv = float(v)
            if not math.isfinite(fv) or fv <= 0.0:
                raise ValueError(f"hex_interface_config.{k} must be positive finite")
            setattr(cfg, k, fv)
        return cfg

    def is_weakened(self):
        return any(
            abs(v - 1.0) > 1e-9
            for v in (
                self.seam_factor,
                self.start_stop_factor,
                self.bead_interface_factor,
                self.interlayer_shear_factor,
            )
        )

    def flag_scale_lut(self):
        lut = np.ones(8, dtype=np.float64)
        for code in range(8):
            has_ss = bool(code & 1)
            has_iface = bool(code & 2)
            has_seam = bool(code & 4)
            if has_seam and has_ss:
                end_fac = min(float(self.seam_factor), float(self.start_stop_factor))
            elif has_seam:
                end_fac = float(self.seam_factor)
            elif has_ss:
                end_fac = float(self.start_stop_factor)
            else:
                end_fac = 1.0
            iface_fac = float(self.bead_interface_factor) if has_iface else 1.0
            lut[code] = end_fac * iface_fac
        return lut

    def effective_material(self, material: Material):
        f = float(self.interlayer_shear_factor)
        if abs(f - 1.0) <= 1e-12:
            return material
        eff = Material(
            e1=material.e1,
            e2=material.e2,
            e3=material.e3 * f,
            g12=material.g12,
            g23=material.g23 * f,
            g13=material.g13 * f,
            nu12=material.nu12,
            nu23=material.nu23,
            nu13=material.nu13,
        )
        local_stiffness(eff)
        return eff

    def to_dict(self):
        return {
            "seam_factor": float(self.seam_factor),
            "start_stop_factor": float(self.start_stop_factor),
            "bead_interface_factor": float(self.bead_interface_factor),
            "interlayer_shear_factor": float(self.interlayer_shear_factor),
        }


_WARP_INIT = False


def _get_device(device="auto"):
    import warp as wp

    global _WARP_INIT
    if not _WARP_INIT:
        wp.init()
        _WARP_INIT = True
    if device == "auto":
        return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    return device


import warp as wp

_vec8i = wp.types.vector(length=8, dtype=wp.int32)
_vec24f = wp.types.vector(length=24, dtype=wp.float32)
_vec24d = wp.types.vector(length=24, dtype=wp.float64)


@wp.kernel
def _matvec_color_f32_kernel(
    color_cells: wp.array(dtype=wp.int32, ndim=1),
    brick_nodes: wp.array(dtype=wp.int32, ndim=2),
    brick_payload: wp.array(dtype=wp.uint32, ndim=2),
    ke_lut: wp.array(dtype=wp.float32, ndim=3),
    flag_scale_lut: wp.array(dtype=wp.float32, ndim=1),
    x: wp.array(dtype=wp.float32, ndim=1),
    y: wp.array(dtype=wp.float32, ndim=1),
):
    tid = wp.tid()
    code = color_cells[tid]
    bid = code >> 6
    c = code & 63
    lx = c & 3
    ly = (c >> 2) & 3
    lz = c >> 4
    base = lx + 5 * ly + 25 * lz

    p = brick_payload[bid, c]
    abin = wp.int32(p & wp.uint32(0xFFFF))
    fcode = wp.int32((p >> wp.uint32(29)) & wp.uint32(7))
    rho = wp.float32((p >> wp.uint32(16)) & wp.uint32(0xFF)) * wp.float32(1.0 / 255.0) * flag_scale_lut[fcode]

    n0 = brick_nodes[bid, base]
    n1 = brick_nodes[bid, base + 1]
    n2 = brick_nodes[bid, base + 5]
    n3 = brick_nodes[bid, base + 6]
    n4 = brick_nodes[bid, base + 25]
    n5 = brick_nodes[bid, base + 26]
    n6 = brick_nodes[bid, base + 30]
    n7 = brick_nodes[bid, base + 31]

    ue = _vec24f()
    ue[0] = x[3 * n0]
    ue[1] = x[3 * n0 + 1]
    ue[2] = x[3 * n0 + 2]
    ue[3] = x[3 * n1]
    ue[4] = x[3 * n1 + 1]
    ue[5] = x[3 * n1 + 2]
    ue[6] = x[3 * n2]
    ue[7] = x[3 * n2 + 1]
    ue[8] = x[3 * n2 + 2]
    ue[9] = x[3 * n3]
    ue[10] = x[3 * n3 + 1]
    ue[11] = x[3 * n3 + 2]
    ue[12] = x[3 * n4]
    ue[13] = x[3 * n4 + 1]
    ue[14] = x[3 * n4 + 2]
    ue[15] = x[3 * n5]
    ue[16] = x[3 * n5 + 1]
    ue[17] = x[3 * n5 + 2]
    ue[18] = x[3 * n6]
    ue[19] = x[3 * n6 + 1]
    ue[20] = x[3 * n6 + 2]
    ue[21] = x[3 * n7]
    ue[22] = x[3 * n7 + 1]
    ue[23] = x[3 * n7 + 2]

    nids = _vec8i(n0, n1, n2, n3, n4, n5, n6, n7)
    for a in range(8):
        nid = nids[a]
        base_i = a * 3
        for d in range(3):
            row = base_i + d
            acc = wp.float32(0.0)
            for j in range(24):
                acc += ke_lut[abin, row, j] * ue[j]
            dof = nid * 3 + d
            y[dof] = y[dof] + rho * acc


@wp.kernel
def _matvec_color_f64_kernel(
    color_cells: wp.array(dtype=wp.int32, ndim=1),
    brick_nodes: wp.array(dtype=wp.int32, ndim=2),
    brick_payload: wp.array(dtype=wp.uint32, ndim=2),
    ke_lut: wp.array(dtype=wp.float64, ndim=3),
    flag_scale_lut: wp.array(dtype=wp.float64, ndim=1),
    x: wp.array(dtype=wp.float64, ndim=1),
    y: wp.array(dtype=wp.float64, ndim=1),
):
    tid = wp.tid()
    code = color_cells[tid]
    bid = code >> 6
    c = code & 63
    lx = c & 3
    ly = (c >> 2) & 3
    lz = c >> 4
    base = lx + 5 * ly + 25 * lz

    p = brick_payload[bid, c]
    abin = wp.int32(p & wp.uint32(0xFFFF))
    fcode = wp.int32((p >> wp.uint32(29)) & wp.uint32(7))
    rho = wp.float64((p >> wp.uint32(16)) & wp.uint32(0xFF)) * wp.float64(1.0 / 255.0) * flag_scale_lut[fcode]

    n0 = brick_nodes[bid, base]
    n1 = brick_nodes[bid, base + 1]
    n2 = brick_nodes[bid, base + 5]
    n3 = brick_nodes[bid, base + 6]
    n4 = brick_nodes[bid, base + 25]
    n5 = brick_nodes[bid, base + 26]
    n6 = brick_nodes[bid, base + 30]
    n7 = brick_nodes[bid, base + 31]

    ue = _vec24d()
    ue[0] = x[3 * n0]
    ue[1] = x[3 * n0 + 1]
    ue[2] = x[3 * n0 + 2]
    ue[3] = x[3 * n1]
    ue[4] = x[3 * n1 + 1]
    ue[5] = x[3 * n1 + 2]
    ue[6] = x[3 * n2]
    ue[7] = x[3 * n2 + 1]
    ue[8] = x[3 * n2 + 2]
    ue[9] = x[3 * n3]
    ue[10] = x[3 * n3 + 1]
    ue[11] = x[3 * n3 + 2]
    ue[12] = x[3 * n4]
    ue[13] = x[3 * n4 + 1]
    ue[14] = x[3 * n4 + 2]
    ue[15] = x[3 * n5]
    ue[16] = x[3 * n5 + 1]
    ue[17] = x[3 * n5 + 2]
    ue[18] = x[3 * n6]
    ue[19] = x[3 * n6 + 1]
    ue[20] = x[3 * n6 + 2]
    ue[21] = x[3 * n7]
    ue[22] = x[3 * n7 + 1]
    ue[23] = x[3 * n7 + 2]

    nids = _vec8i(n0, n1, n2, n3, n4, n5, n6, n7)
    for a in range(8):
        nid = nids[a]
        base_i = a * 3
        for d in range(3):
            row = base_i + d
            acc = wp.float64(0.0)
            for j in range(24):
                acc += ke_lut[abin, row, j] * ue[j]
            dof = nid * 3 + d
            y[dof] = y[dof] + rho * acc


@wp.kernel
def _diag_color_f32_kernel(
    color_cells: wp.array(dtype=wp.int32, ndim=1),
    brick_nodes: wp.array(dtype=wp.int32, ndim=2),
    brick_payload: wp.array(dtype=wp.uint32, ndim=2),
    ke_diag: wp.array(dtype=wp.float32, ndim=2),
    flag_scale_lut: wp.array(dtype=wp.float32, ndim=1),
    diag: wp.array(dtype=wp.float32, ndim=1),
):
    tid = wp.tid()
    code = color_cells[tid]
    bid = code >> 6
    c = code & 63
    lx = c & 3
    ly = (c >> 2) & 3
    lz = c >> 4
    base = lx + 5 * ly + 25 * lz

    p = brick_payload[bid, c]
    abin = wp.int32(p & wp.uint32(0xFFFF))
    fcode = wp.int32((p >> wp.uint32(29)) & wp.uint32(7))
    rho = wp.float32((p >> wp.uint32(16)) & wp.uint32(0xFF)) * wp.float32(1.0 / 255.0) * flag_scale_lut[fcode]

    n0 = brick_nodes[bid, base]
    n1 = brick_nodes[bid, base + 1]
    n2 = brick_nodes[bid, base + 5]
    n3 = brick_nodes[bid, base + 6]
    n4 = brick_nodes[bid, base + 25]
    n5 = brick_nodes[bid, base + 26]
    n6 = brick_nodes[bid, base + 30]
    n7 = brick_nodes[bid, base + 31]
    nids = _vec8i(n0, n1, n2, n3, n4, n5, n6, n7)
    for a in range(8):
        nid = nids[a]
        base_i = a * 3
        for d in range(3):
            dof = nid * 3 + d
            diag[dof] = diag[dof] + rho * ke_diag[abin, base_i + d]


@wp.kernel
def _apply_fixed_mask_f32_kernel(
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    vec: wp.array(dtype=wp.float32, ndim=1),
):
    nid = wp.tid()
    if fixed_mask[nid] != wp.uint8(0):
        vec[3 * nid] = wp.float32(0.0)
        vec[3 * nid + 1] = wp.float32(0.0)
        vec[3 * nid + 2] = wp.float32(0.0)


@wp.kernel
def _apply_fixed_mask_f64_kernel(
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    vec: wp.array(dtype=wp.float64, ndim=1),
):
    nid = wp.tid()
    if fixed_mask[nid] != wp.uint8(0):
        vec[3 * nid] = wp.float64(0.0)
        vec[3 * nid + 1] = wp.float64(0.0)
        vec[3 * nid + 2] = wp.float64(0.0)


@wp.kernel
def _axpby_f32_kernel(
    alpha: wp.float32,
    x: wp.array(dtype=wp.float32, ndim=1),
    beta: wp.float32,
    y: wp.array(dtype=wp.float32, ndim=1),
    out: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    out[i] = alpha * x[i] + beta * y[i]


@wp.kernel
def _jacobi_update_f32_kernel(
    omega: wp.float32,
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    diag_inv: wp.array(dtype=wp.float32, ndim=1),
    b: wp.array(dtype=wp.float32, ndim=1),
    ax: wp.array(dtype=wp.float32, ndim=1),
    x: wp.array(dtype=wp.float32, ndim=1),
):
    dof = wp.tid()
    nid = dof // 3
    if fixed_mask[nid] != wp.uint8(0):
        x[dof] = wp.float32(0.0)
    else:
        x[dof] = x[dof] + omega * diag_inv[dof] * (b[dof] - ax[dof])


@wp.kernel
def _cheb_init_f32_kernel(
    inv_c: wp.float32,
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    diag_inv: wp.array(dtype=wp.float32, ndim=1),
    r: wp.array(dtype=wp.float32, ndim=1),
    d: wp.array(dtype=wp.float32, ndim=1),
    x: wp.array(dtype=wp.float32, ndim=1),
):
    dof = wp.tid()
    if fixed_mask[dof // 3] != wp.uint8(0):
        d[dof] = wp.float32(0.0)
        x[dof] = wp.float32(0.0)
    else:
        step = inv_c * diag_inv[dof] * r[dof]
        d[dof] = step
        x[dof] = x[dof] + step


@wp.kernel
def _cheb_step_f32_kernel(
    alpha: wp.float32,
    beta: wp.float32,
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    diag_inv: wp.array(dtype=wp.float32, ndim=1),
    r: wp.array(dtype=wp.float32, ndim=1),
    d: wp.array(dtype=wp.float32, ndim=1),
    x: wp.array(dtype=wp.float32, ndim=1),
):
    dof = wp.tid()
    if fixed_mask[dof // 3] != wp.uint8(0):
        d[dof] = wp.float32(0.0)
        x[dof] = wp.float32(0.0)
    else:
        step = beta * d[dof] + alpha * diag_inv[dof] * r[dof]
        d[dof] = step
        x[dof] = x[dof] + step


@wp.kernel
def _diag_precond_f32_kernel(
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    diag_inv: wp.array(dtype=wp.float32, ndim=1),
    r: wp.array(dtype=wp.float32, ndim=1),
    z: wp.array(dtype=wp.float32, ndim=1),
):
    dof = wp.tid()
    if fixed_mask[dof // 3] != wp.uint8(0):
        z[dof] = wp.float32(0.0)
    else:
        z[dof] = diag_inv[dof] * r[dof]


@wp.kernel
def _prolong_add_f32_kernel(
    fine_parents: wp.array(dtype=wp.int32, ndim=2),
    fine_weights: wp.array(dtype=wp.float32, ndim=1),
    coarse_vec: wp.array(dtype=wp.float32, ndim=1),
    fine_vec: wp.array(dtype=wp.float32, ndim=1),
):
    nid = wp.tid()
    w = fine_weights[nid]
    if w == wp.float32(0.0):
        return
    acc0 = wp.float32(0.0)
    acc1 = wp.float32(0.0)
    acc2 = wp.float32(0.0)
    for k in range(8):
        cid = fine_parents[nid, k]
        if cid >= 0:
            acc0 += coarse_vec[3 * cid]
            acc1 += coarse_vec[3 * cid + 1]
            acc2 += coarse_vec[3 * cid + 2]
    fine_vec[3 * nid] = fine_vec[3 * nid] + w * acc0
    fine_vec[3 * nid + 1] = fine_vec[3 * nid + 1] + w * acc1
    fine_vec[3 * nid + 2] = fine_vec[3 * nid + 2] + w * acc2


@wp.kernel
def _restrict_f32_kernel(
    fine_parents: wp.array(dtype=wp.int32, ndim=2),
    fine_weights: wp.array(dtype=wp.float32, ndim=1),
    fine_vec: wp.array(dtype=wp.float32, ndim=1),
    coarse_vec: wp.array(dtype=wp.float32, ndim=1),
):
    nid = wp.tid()
    w = fine_weights[nid]
    if w == wp.float32(0.0):
        return
    v0 = w * fine_vec[3 * nid]
    v1 = w * fine_vec[3 * nid + 1]
    v2 = w * fine_vec[3 * nid + 2]
    for k in range(8):
        cid = fine_parents[nid, k]
        if cid >= 0:
            wp.atomic_add(coarse_vec, 3 * cid, v0)
            wp.atomic_add(coarse_vec, 3 * cid + 1, v1)
            wp.atomic_add(coarse_vec, 3 * cid + 2, v2)


@wp.kernel
def _dot_reduce_f32_kernel(
    a: wp.array(dtype=wp.float32, ndim=1),
    b: wp.array(dtype=wp.float32, ndim=1),
    out_sum: wp.array(dtype=wp.float64, ndim=1),
):
    # Warp tid-strided reduction into 256 partial float64 accumulators (fast & deterministic).
    tid = wp.tid()
    bucket = tid & 255
    val = wp.float64(a[tid]) * wp.float64(b[tid])
    wp.atomic_add(out_sum, bucket, val)


def _gpu_dot_f32(a, b, scratch_256, device):
    scratch_256.zero_()
    wp.launch(_dot_reduce_f32_kernel, dim=len(a), inputs=[a, b, scratch_256], device=device)
    return float(scratch_256.numpy().sum())


@wp.kernel
def _dot_reduce_f64_kernel(
    a: wp.array(dtype=wp.float64, ndim=1),
    b: wp.array(dtype=wp.float64, ndim=1),
    out_sum: wp.array(dtype=wp.float64, ndim=1),
):
    tid = wp.tid()
    bucket = tid & 255
    wp.atomic_add(out_sum, bucket, a[tid] * b[tid])


def _gpu_dot_f64(a, b, scratch_256, device):
    scratch_256.zero_()
    wp.launch(_dot_reduce_f64_kernel, dim=len(a), inputs=[a, b, scratch_256], device=device)
    return float(scratch_256.numpy().sum())


@wp.kernel
def _axpby_f64_kernel(
    alpha: wp.float64,
    x: wp.array(dtype=wp.float64, ndim=1),
    beta: wp.float64,
    y: wp.array(dtype=wp.float64, ndim=1),
    out: wp.array(dtype=wp.float64, ndim=1),
):
    i = wp.tid()
    out[i] = alpha * x[i] + beta * y[i]


@wp.kernel
def _scale_f32_to_f64_masked_kernel(
    scale: wp.float64,
    fixed_mask: wp.array(dtype=wp.uint8, ndim=1),
    x32: wp.array(dtype=wp.float32, ndim=1),
    out64: wp.array(dtype=wp.float64, ndim=1),
):
    i = wp.tid()
    if fixed_mask[i // 3] != wp.uint8(0):
        out64[i] = wp.float64(0.0)
    else:
        out64[i] = scale * wp.float64(x32[i])


@wp.kernel
def _scale_f64_to_f32_kernel(
    scale: wp.float64,
    x64: wp.array(dtype=wp.float64, ndim=1),
    out32: wp.array(dtype=wp.float32, ndim=1),
):
    i = wp.tid()
    out32[i] = wp.float32(scale * x64[i])


def _build_brick_lookup(brick_coords, brick_shape):
    nbx, nby, nbz = (int(v) for v in brick_shape)
    bkeys = (
        brick_coords[:, 2].astype(np.int64) * (nby * nbx)
        + brick_coords[:, 1].astype(np.int64) * nbx
        + brick_coords[:, 0].astype(np.int64)
    )
    order = np.argsort(bkeys)
    sorted_keys = bkeys[order]
    return bkeys, order, sorted_keys, (nbx, nby, nbz)


def _lookup_bricks(target_keys, valid_mask, order, sorted_keys):
    idx = np.full(len(target_keys), -1, dtype=np.int32)
    if not valid_mask.any():
        return idx
    tk = target_keys[valid_mask]
    pos = np.searchsorted(sorted_keys, tk)
    in_b = pos < len(sorted_keys)
    pos_safe = np.where(in_b, pos, 0)
    hit = in_b & (sorted_keys[pos_safe] == tk)
    sub = np.flatnonzero(valid_mask)[hit]
    idx[sub] = order[pos_safe[hit]]
    return idx


def build_level_topology(grid: BrickGrid):
    """Build compact active node numbering, brick_nodes (B, 125), and 8-color voxel lists."""
    coords = grid.brick_coords
    masks = grid.brick_masks
    nx, ny, nz = grid.shape
    nbx, nby, nbz = grid.brick_shape
    n_bricks = len(coords)

    # Mark which of the 5x5x5=125 local nodes in each brick are touched by active voxels.
    active_125 = np.zeros((n_bricks, 125), dtype=bool)
    for c in range(64):
        occ = ((masks >> np.uint64(c)) & np.uint64(1)) != 0
        if not occ.any():
            continue
        lx, ly, lz = c & 3, (c >> 2) & 3, c >> 4
        base = lx + 5 * ly + 25 * lz
        for off in (0, 1, 5, 6, 25, 26, 30, 31):
            active_125[occ, base + off] = True

    b_idx, ln_idx = np.nonzero(active_125)
    px = ln_idx % 5
    py = (ln_idx // 5) % 5
    pz = ln_idx // 25
    gx = 4 * coords[b_idx, 0].astype(np.int64) + px
    gy = 4 * coords[b_idx, 1].astype(np.int64) + py
    gz = 4 * coords[b_idx, 2].astype(np.int64) + pz

    stride_y = np.int64(nx + 1)
    stride_z = np.int64((nx + 1) * (ny + 1))
    nkeys = gz * stride_z + gy * stride_y + gx
    uniq_keys, inv = np.unique(nkeys, return_inverse=True)
    n_nodes = len(uniq_keys)

    brick_nodes = np.full((n_bricks, 125), -1, dtype=np.int32)
    brick_nodes[b_idx, ln_idx] = inv.astype(np.int32)
    del active_125, b_idx, ln_idx, px, py, pz, gx, gy, gz, nkeys, inv

    iz_n = (uniq_keys // stride_z).astype(np.int32)
    rem = uniq_keys % stride_z
    iy_n = (rem // stride_y).astype(np.int32)
    ix_n = (rem % stride_y).astype(np.int32)
    node_ijk = np.column_stack([ix_n, iy_n, iz_n])

    # Build 8 parity colors over all active cells (bid << 6 | c).
    color_cells = []
    for color in range(8):
        cx = color & 1
        cy = (color >> 1) & 1
        cz = (color >> 2) & 1
        c_list = [lx + 4 * ly + 16 * lz for lz in (cz, cz + 2) for ly in (cy, cy + 2) for lx in (cx, cx + 2)]
        parts = []
        for c in c_list:
            hit = np.flatnonzero(((masks >> np.uint64(c)) & np.uint64(1)) != 0).astype(np.int32)
            if len(hit):
                parts.append((hit << 6) | c)
        color_cells.append(np.concatenate(parts) if parts else np.empty((0,), dtype=np.int32))

    return node_ijk, uniq_keys, brick_nodes, color_cells


def extract_boundary_quads(grid: BrickGrid, brick_nodes):
    """Extract exterior boundary quad faces (F, 4) with outward normal winding."""
    coords = grid.brick_coords
    masks = grid.brick_masks
    nbx, nby, nbz = grid.brick_shape
    bkeys, order, sorted_keys, _ = _build_brick_lookup(coords, (nbx, nby, nbz))

    nbr_x_pos = _lookup_bricks(bkeys + 1, coords[:, 0] < nbx - 1, order, sorted_keys)
    nbr_x_neg = _lookup_bricks(bkeys - 1, coords[:, 0] > 0, order, sorted_keys)
    nbr_y_pos = _lookup_bricks(bkeys + nbx, coords[:, 1] < nby - 1, order, sorted_keys)
    nbr_y_neg = _lookup_bricks(bkeys - nbx, coords[:, 1] > 0, order, sorted_keys)
    nbr_z_pos = _lookup_bricks(bkeys + nbx * nby, coords[:, 2] < nbz - 1, order, sorted_keys)
    nbr_z_neg = _lookup_bricks(bkeys - nbx * nby, coords[:, 2] > 0, order, sorted_keys)

    # Precomputed 4 local-node offsets in brick_nodes (5x5x5) oriented outward from unit cube.
    face_specs = [
        # (dx, dy, dz, nbr_brick_arr, boundary_l_val, [ln0, ln1, ln2, ln3])
        (-1, 0, 0, nbr_x_neg, 0, (0, 25, 30, 5)),
        (1, 0, 0, nbr_x_pos, 3, (1, 6, 31, 26)),
        (0, -1, 0, nbr_y_neg, 0, (0, 1, 26, 25)),
        (0, 1, 0, nbr_y_pos, 3, (5, 30, 31, 6)),
        (0, 0, -1, nbr_z_neg, 0, (0, 5, 6, 1)),
        (0, 0, 1, nbr_z_pos, 3, (25, 26, 31, 30)),
    ]

    quad_parts = []
    owner_parts = []
    for dx, dy, dz, nbr_arr, bnd_val, corner_offs in face_specs:
        for c in range(64):
            occ = ((masks >> np.uint64(c)) & np.uint64(1)) != 0
            if not occ.any():
                continue
            lx, ly, lz = c & 3, (c >> 2) & 3, c >> 4
            coord_val = lx if dx != 0 else (ly if dy != 0 else lz)
            if coord_val != bnd_val:
                c_nbr = (lx + dx) + 4 * (ly + dy) + 16 * (lz + dz)
                nbr_occ = ((masks >> np.uint64(c_nbr)) & np.uint64(1)) != 0
                ext_b = np.flatnonzero(occ & ~nbr_occ)
            else:
                nlx = (lx + dx) & 3
                nly = (ly + dy) & 3
                nlz = (lz + dz) & 3
                c_nbr = nlx + 4 * nly + 16 * nlz
                nb = nbr_arr
                has_nb = nb >= 0
                nbr_occ = np.zeros(len(masks), dtype=bool)
                nbr_occ[has_nb] = ((masks[nb[has_nb]] >> np.uint64(c_nbr)) & np.uint64(1)) != 0
                ext_b = np.flatnonzero(occ & ~nbr_occ)
            if not len(ext_b):
                continue
            base = lx + 5 * ly + 25 * lz
            q = np.column_stack([brick_nodes[ext_b, base + off] for off in corner_offs])
            quad_parts.append(q)
            owner_parts.append(((ext_b.astype(np.int32) << 6) | c))

    quads = np.concatenate(quad_parts, axis=0) if quad_parts else np.empty((0, 4), dtype=np.int32)
    owners = np.concatenate(owner_parts, axis=0) if owner_parts else np.empty((0,), dtype=np.int32)
    return quads, owners


def select_quad_faces(node_xyz, quad_faces, region):
    """Select exterior quad faces matching PathFEA's face/box/boxes union conventions."""
    if isinstance(region, dict):
        if set(region) != {"boxes"} or not isinstance(region["boxes"], list) or not region["boxes"]:
            raise ValueError("选区并集必须包含非空 boxes 列表")
        masks = [select_quad_faces(node_xyz, quad_faces, box) for box in region["boxes"]]
        if any(not m.any() for m in masks):
            raise ValueError("选区并集中的一个子区域没有选中表面；不能静默丢失安装点")
        return np.logical_or.reduce(masks)
    if isinstance(region, str):
        axis = "XYZ".index(region[0])
        edge = node_xyz[:, axis].min() if region[1] == "-" else node_xyz[:, axis].max()
        return np.all(np.isclose(node_xyz[quad_faces, axis], edge, atol=1e-7, rtol=0.0), axis=1)
    box = np.asarray(region, dtype=np.float64)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
        raise ValueError("选区格式：xmin,ymin,zmin,xmax,ymax,zmax；最大值不能小于最小值")
    centers = node_xyz[quad_faces].mean(axis=1)
    return np.all((centers >= box[0]) & (centers <= box[1]), axis=1)


def quad_surface_load(node_xyz, selected_quads, n_nodes, force, load_point=None):
    """Area-weighted nodal force + equivalent remote couple matching solid.py:surface_load."""
    force = np.asarray(force, dtype=np.float64)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("force must be a finite 3-vector")
    if len(selected_quads) == 0:
        raise ValueError("Load surface has zero area")

    uniq_nodes, inv = np.unique(selected_quads.ravel(), return_inverse=True)
    local_quads = inv.reshape(-1, 4)
    p = node_xyz[uniq_nodes]
    q = p[local_quads]
    areas = 0.5 * np.linalg.norm(np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]), axis=1)
    total_area = float(areas.sum())
    if total_area <= 0:
        raise ValueError("Load surface has zero area")

    weights = np.zeros(len(uniq_nodes), dtype=np.float64)
    quarter = areas / (4.0 * total_area)
    for j in range(4):
        np.add.at(weights, local_quads[:, j], quarter)

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

    f_full = np.zeros((n_nodes, 3), dtype=np.float64)
    f_full[uniq_nodes] = f_loc
    return f_full


def coarsen_brick_grid(grid: BrickGrid, flag_scale_lut=None):
    """Coarsen 4x4x4 bricks 2x2x2 -> 1 while packing into coarse 4x4x4 bricks."""
    coords = grid.brick_coords
    masks = grid.brick_masks
    payload = grid.brick_payload
    nx, ny, nz = grid.shape
    n_bins = int(grid.meta.get("n_angle_bins", 360))

    nx_c = ((max(1, (nx + 1) // 2) + 3) // 4) * 4
    ny_c = ((max(1, (ny + 1) // 2) + 3) // 4) * 4
    nz_c = ((max(1, (nz + 1) // 2) + 3) // 4) * 4
    nbx_c, nby_c = nx_c // 4, ny_c // 4

    cb_coords_all = coords >> 1
    cb_keys = (
        cb_coords_all[:, 2].astype(np.int64) * (nbx_c * nby_c)
        + cb_coords_all[:, 1].astype(np.int64) * nbx_c
        + cb_coords_all[:, 0].astype(np.int64)
    )
    uniq_cb, inv_cb = np.unique(cb_keys, return_inverse=True)
    n_cb = len(uniq_cb)
    bz_c = (uniq_cb // (nbx_c * nby_c)).astype(np.int32)
    rem = uniq_cb % (nbx_c * nby_c)
    by_c = (rem // nbx_c).astype(np.int32)
    bx_c = (rem % nbx_c).astype(np.int32)
    c_brick_coords = np.column_stack([bx_c, by_c, bz_c])

    sum_rho = np.zeros((n_cb, 64), dtype=np.float64)
    sum_c2 = np.zeros((n_cb, 64), dtype=np.float64)
    sum_s2 = np.zeros((n_cb, 64), dtype=np.float64)
    c_masks = np.zeros(n_cb, dtype=np.uint64)

    abin_all, rho_q_all, flags_all = unpack_payload(payload)
    rho_all = rho_q_all.astype(np.float64) / 255.0
    if flag_scale_lut is not None:
        fl_lut = np.asarray(flag_scale_lut, dtype=np.float64)
        rho_all = rho_all * fl_lut[(flags_all >> 5) & 7]
    ang_all = bin_to_angle(abin_all, n_bins)
    c2_all = rho_all * np.cos(2.0 * ang_all)
    s2_all = rho_all * np.sin(2.0 * ang_all)

    sub_bx = (coords[:, 0] & 1) * 2
    sub_by = (coords[:, 1] & 1) * 2
    sub_bz = (coords[:, 2] & 1) * 2

    for c in range(64):
        occ = ((masks >> np.uint64(c)) & np.uint64(1)) != 0
        if not occ.any():
            continue
        lx, ly, lz = c & 3, (c >> 2) & 3, c >> 4
        clx = sub_bx[occ] + (lx >> 1)
        cly = sub_by[occ] + (ly >> 1)
        clz = sub_bz[occ] + (lz >> 1)
        cc = clx + 4 * cly + 16 * clz
        dest_b = inv_cb[occ]
        np.bitwise_or.at(c_masks, dest_b, np.uint64(1) << cc.astype(np.uint64))
        np.add.at(sum_rho, (dest_b, cc), rho_all[occ, c])
        np.add.at(sum_c2, (dest_b, cc), c2_all[occ, c])
        np.add.at(sum_s2, (dest_b, cc), s2_all[occ, c])

    c_rho_q = np.clip(np.rint((sum_rho / 8.0) * 255.0), 1, 255).astype(np.int32)
    c_theta = 0.5 * np.arctan2(sum_s2, sum_c2)
    c_abin = angle_to_bin(c_theta, n_bins)
    occ_c = ((c_masks[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)) != 0
    c_payload = np.where(occ_c, (c_abin & 0xFFFF) | ((c_rho_q & 0xFF) << 16), 0).astype(np.uint32)

    return BrickGrid(
        origin=grid.origin.copy(),
        pitch=grid.pitch * 2.0,
        shape=(nx_c, ny_c, nz_c),
        brick_coords=c_brick_coords,
        brick_masks=c_masks,
        brick_payload=c_payload,
        meta={"n_angle_bins": n_bins},
    )


def build_prolongation_map(fine_ijk, fine_fixed_mask, coarse_grid: BrickGrid, coarse_keys):
    """Build (fine_parents[N_f, 8], fine_weights[N_f], coarse_fixed_mask[N_c])."""
    nx_c, ny_c, _ = coarse_grid.shape
    stride_y = np.int64(nx_c + 1)
    stride_z = np.int64((nx_c + 1) * (ny_c + 1))

    fx = fine_ijk[:, 0].astype(np.int64)
    fy = fine_ijk[:, 1].astype(np.int64)
    fz = fine_ijk[:, 2].astype(np.int64)
    cx0 = fx >> 1
    cy0 = fy >> 1
    cz0 = fz >> 1
    ox_flag = fx & 1
    oy_flag = fy & 1
    oz_flag = fz & 1
    odd_count = ox_flag + oy_flag + oz_flag
    fine_weights = (0.5 ** odd_count).astype(np.float32)

    n_fine = len(fine_ijk)
    n_coarse = len(coarse_keys)
    fine_parents = np.full((n_fine, 8), -1, dtype=np.int32)
    coarse_fixed = np.zeros(n_coarse, dtype=np.uint8)

    # coarse_keys came from np.unique so it is already sorted.
    for slot in range(8):
        ox = slot & 1
        oy = (slot >> 1) & 1
        oz = (slot >> 2) & 1
        valid = ((ox == 0) | (ox_flag == 1)) & ((oy == 0) | (oy_flag == 1)) & ((oz == 0) | (oz_flag == 1))
        if not valid.any():
            continue
        sub = np.flatnonzero(valid)
        tkeys = (cz0[sub] + oz) * stride_z + (cy0[sub] + oy) * stride_y + (cx0[sub] + ox)
        pos = np.searchsorted(coarse_keys, tkeys)
        in_b = pos < n_coarse
        pos_s = np.where(in_b, pos, 0)
        matched = in_b & (coarse_keys[pos_s] == tkeys)
        if not matched.all():
            raise RuntimeError("Missing coarse parent node in nested brick prolongation")
        cids = pos_s.astype(np.int32)
        fine_parents[sub, slot] = cids
        fixed_sub = fine_fixed_mask[sub] != 0
        if fixed_sub.any():
            coarse_fixed[cids[fixed_sub]] = 1

    # Enforce P[fixed_fine, :] = 0 and P[:, fixed_coarse] = 0 so R = P^T is exact.
    fine_weights[fine_fixed_mask != 0] = 0.0
    for slot in range(8):
        cids = fine_parents[:, slot]
        valid = cids >= 0
        if valid.any():
            drop = np.zeros(len(cids), dtype=bool)
            drop[valid] = coarse_fixed[cids[valid]] != 0
            fine_parents[drop, slot] = -1
    fine_parents[fine_fixed_mask != 0, :] = -1
    return fine_parents, fine_weights, coarse_fixed


def assemble_coarse_csr(grid: BrickGrid, brick_nodes, n_nodes, Ke_f64, flag_scale_lut=None):
    """Assemble global sparse stiffness matrix (3*N, 3*N) in CSR for coarsest level."""
    levels = []
    masks = grid.brick_masks
    payload = grid.brick_payload
    fl_lut = None if flag_scale_lut is None else np.asarray(flag_scale_lut, dtype=np.float64)

    all_bids = []
    all_cs = []
    for c in range(64):
        occ = np.flatnonzero(((masks >> np.uint64(c)) & np.uint64(1)) != 0)
        if len(occ):
            all_bids.append(occ)
            all_cs.append(np.full(len(occ), c, dtype=np.int32))
    bids = np.concatenate(all_bids)
    cs = np.concatenate(all_cs)
    n_elems = len(bids)

    corner_offs = np.array([0, 1, 5, 6, 25, 26, 30, 31], dtype=np.int32)
    chunk_size = 4096
    for start in range(0, n_elems, chunk_size):
        stop = min(n_elems, start + chunk_size)
        b_ch = bids[start:stop]
        c_ch = cs[start:stop]
        lx = c_ch & 3
        ly = (c_ch >> 2) & 3
        lz = c_ch >> 4
        bases = lx + 5 * ly + 25 * lz
        nids = brick_nodes[b_ch[:, None], bases[:, None] + corner_offs[None, :]]
        dofs = (nids[:, :, None] * 3 + np.arange(3, dtype=np.int32)).reshape(-1, 24)
        p_ch = payload[b_ch, c_ch]
        abin = (p_ch & np.uint32(0xFFFF)).astype(np.int32)
        rho = ((p_ch >> np.uint32(16)) & np.uint32(0xFF)).astype(np.float64) / 255.0
        if fl_lut is not None:
            fcode = ((p_ch >> np.uint32(29)) & np.uint32(7)).astype(np.int32)
            rho = rho * fl_lut[fcode]
        ke = Ke_f64[abin] * rho[:, None, None]
        rows = np.repeat(dofs, 24, axis=1).ravel()
        cols = np.tile(dofs, (1, 24)).ravel()
        block = coo_matrix((ke.ravel(), (rows, cols)), shape=(3 * n_nodes, 3 * n_nodes)).tocsr()
        lvl = 0
        while lvl < len(levels) and levels[lvl] is not None:
            block = levels[lvl] + block
            levels[lvl] = None
            lvl += 1
        if lvl == len(levels):
            levels.append(block)
        else:
            levels[lvl] = block
    K = None
    for block in levels:
        if block is not None:
            K = block if K is None else K + block
    K.eliminate_zeros()
    return K


@dataclass
class _GPUMGLevel:
    grid: BrickGrid
    n_nodes: int
    n_dofs: int
    fixed_mask_np: np.ndarray
    brick_nodes_dev: object
    brick_payload_dev: object
    color_cells_dev: list
    ke_f32_dev: object
    flag_scale_lut_dev: object
    fixed_mask_dev: object
    diag_inv_dev: object
    lambda_max: float
    # Transfer to next coarser level (None on coarsest)
    fine_parents_dev: object = None
    fine_weights_dev: object = None
    # Scratch vectors on coarse levels (None on level 0)
    x_dev: object = None
    b_dev: object = None
    r_dev: object = None
    ax_dev: object = None
    d_dev: object = None


def _launch_matvec_f32(level: _GPUMGLevel, x_dev, y_dev, device):
    y_dev.zero_()
    for c_dev in level.color_cells_dev:
        if len(c_dev) == 0:
            continue
        wp.launch(
            _matvec_color_f32_kernel,
            dim=len(c_dev),
            inputs=[c_dev, level.brick_nodes_dev, level.brick_payload_dev, level.ke_f32_dev, level.flag_scale_lut_dev, x_dev, y_dev],
            device=device,
        )
    wp.launch(_apply_fixed_mask_f32_kernel, dim=level.n_nodes, inputs=[level.fixed_mask_dev, y_dev], device=device)


def _launch_matvec_f64(color_cells_dev, brick_nodes_dev, brick_payload_dev, ke_f64_dev, flag_scale_lut_f64_dev, fixed_mask_dev, n_nodes, x_dev, y_dev, device, apply_fixed_mask=True):
    y_dev.zero_()
    for c_dev in color_cells_dev:
        if len(c_dev) == 0:
            continue
        wp.launch(
            _matvec_color_f64_kernel,
            dim=len(c_dev),
            inputs=[c_dev, brick_nodes_dev, brick_payload_dev, ke_f64_dev, flag_scale_lut_f64_dev, x_dev, y_dev],
            device=device,
        )
    if apply_fixed_mask:
        wp.launch(_apply_fixed_mask_f64_kernel, dim=n_nodes, inputs=[fixed_mask_dev, y_dev], device=device)


def _estimate_lambda_max(level: _GPUMGLevel, scratch_256, tmp_x, tmp_y, device, n_power=8):
    rng = np.random.default_rng(20261119)
    v0 = rng.normal(size=level.n_dofs).astype(np.float32)
    v0[np.repeat(level.fixed_mask_np != 0, 3)] = 0.0
    nrm = float(np.linalg.norm(v0))
    if nrm <= 0:
        return 2.0
    v0 /= nrm
    wp.copy(tmp_x, wp.from_numpy(v0, dtype=wp.float32))
    lam = 2.0
    for _ in range(n_power):
        _launch_matvec_f32(level, tmp_x, tmp_y, device)
        wp.launch(_diag_precond_f32_kernel, dim=level.n_dofs, inputs=[level.fixed_mask_dev, level.diag_inv_dev, tmp_y, tmp_y], device=device)
        lam = _gpu_dot_f32(tmp_x, tmp_y, scratch_256, device)
        nrm2 = _gpu_dot_f32(tmp_y, tmp_y, scratch_256, device)
        if nrm2 <= 0:
            break
        inv_n = np.float32(1.0 / np.sqrt(nrm2))
        wp.launch(_axpby_f32_kernel, dim=level.n_dofs, inputs=[inv_n, tmp_y, np.float32(0.0), tmp_y, tmp_x], device=device)
    return max(float(lam) * 1.15, 0.5)


def _chebyshev_smooth_f32(level: _GPUMGLevel, b_dev, x_dev, r_dev, d_dev, ax_dev, degree, device, zero_init=False):
    lam_max = level.lambda_max
    lam_min = 0.12 * lam_max
    c_val = 0.5 * (lam_max + lam_min)
    d_val = 0.5 * (lam_max - lam_min)
    if zero_init:
        x_dev.zero_()
        wp.copy(r_dev, b_dev)
    else:
        _launch_matvec_f32(level, x_dev, ax_dev, device)
        wp.launch(_axpby_f32_kernel, dim=level.n_dofs, inputs=[np.float32(1.0), b_dev, np.float32(-1.0), ax_dev, r_dev], device=device)
    wp.launch(_apply_fixed_mask_f32_kernel, dim=level.n_nodes, inputs=[level.fixed_mask_dev, r_dev], device=device)
    wp.launch(_cheb_init_f32_kernel, dim=level.n_dofs, inputs=[np.float32(1.0 / c_val), level.fixed_mask_dev, level.diag_inv_dev, r_dev, d_dev, x_dev], device=device)
    if degree <= 1:
        return
    two_sigma = 2.0 * c_val / max(d_val, 1e-12)
    w_prev = d_val / c_val
    for _k in range(1, degree):
        _launch_matvec_f32(level, x_dev, ax_dev, device)
        wp.launch(_axpby_f32_kernel, dim=level.n_dofs, inputs=[np.float32(1.0), b_dev, np.float32(-1.0), ax_dev, r_dev], device=device)
        wp.launch(_apply_fixed_mask_f32_kernel, dim=level.n_nodes, inputs=[level.fixed_mask_dev, r_dev], device=device)
        w_cur = 1.0 / (two_sigma - w_prev)
        alpha = (2.0 / max(d_val, 1e-12)) * w_cur
        beta = w_prev * w_cur
        w_prev = w_cur
        wp.launch(_cheb_step_f32_kernel, dim=level.n_dofs, inputs=[np.float32(alpha), np.float32(beta), level.fixed_mask_dev, level.diag_inv_dev, r_dev, d_dev, x_dev], device=device)


def solve_hex_gpu(
    grid: BrickGrid,
    material=None,
    interface_cfg: HexInterfaceConfig = None,
    fixed="X-",
    loaded="X+",
    force=(0.0, 0.0, -10.0),
    load_point=None,
    build_axis="Z",
    rtol=1e-7,
    max_outer_ir=30,
    max_inner_iter=200,
    max_levels=6,
    coarse_dof_target=12000,
    smooth_degree=2,
    device="auto",
    progress=None,
):
    """Solve H8 orthotropic linear elasticity on sparse 4x4x4 bricks with RTX 4070 GMG-PCG + fp64 Truncated FCG."""
    t_total = time.perf_counter()
    progress = progress or (lambda stage: None)
    if build_axis not in ("X", "Y", "Z"):
        raise ValueError("build_axis must be X/Y/Z")
    if not 0 < rtol <= 1e-4:
        raise ValueError("rtol must be in (0, 1e-4]")
    material = material or Material()
    local_stiffness(material)
    interface_cfg = interface_cfg or HexInterfaceConfig()
    eff_material = interface_cfg.effective_material(material)
    flag_lut_f64 = interface_cfg.flag_scale_lut()
    flag_lut_f32 = flag_lut_f64.astype(np.float32)
    unit_lut_f32 = np.ones(8, dtype=np.float32)

    force = np.asarray(force, dtype=np.float64)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("force must be a finite 3-vector")

    used_device = _get_device(device)
    n_bins = int(grid.meta.get("n_angle_bins", 360))
    dx, dy, dz = (float(x) for x in grid.pitch)

    progress("topology_and_boundary")
    t_setup = time.perf_counter()
    node_ijk, node_keys, brick_nodes, color_cells = build_level_topology(grid)
    n_nodes = len(node_ijk)
    n_dofs = 3 * n_nodes
    node_xyz = grid.origin[None, :] + node_ijk.astype(np.float64) * grid.pitch[None, :]

    quad_faces, _ = extract_boundary_quads(grid, brick_nodes)
    fixed_mask_q = select_quad_faces(node_xyz, quad_faces, fixed)
    load_mask_q = select_quad_faces(node_xyz, quad_faces, loaded)
    if not fixed_mask_q.any() or not load_mask_q.any():
        raise ValueError("固定或加载选区没有选中表面四边形")

    fixed_nodes = np.unique(quad_faces[fixed_mask_q])
    loaded_nodes = np.unique(quad_faces[load_mask_q])
    if np.intersect1d(fixed_nodes, loaded_nodes).size:
        raise ValueError("固定区域与加载区域共享节点，请分开选区")

    f_64 = quad_surface_load(node_xyz, quad_faces[load_mask_q], n_nodes, force, load_point=load_point).ravel()
    fixed_mask_np = np.zeros(n_nodes, dtype=np.uint8)
    fixed_mask_np[fixed_nodes] = 1
    free_mask_np = np.repeat(fixed_mask_np == 0, 3)

    # Save small boundary metadata arrays and release large node_xyz/quad_faces if giant.
    fixed_xyz = node_xyz[fixed_nodes].copy()
    loaded_xyz = node_xyz[loaded_nodes].copy()
    loaded_forces_compact = f_64.reshape(-1, 3)[loaded_nodes].copy()
    quad_area_stats = {}
    for name, msk in (("fixed", fixed_mask_q), ("loaded", load_mask_q)):
        q = node_xyz[quad_faces[msk]]
        a = 0.5 * np.linalg.norm(np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]), axis=1)
        quad_area_stats[name] = {
            "faces": int(msk.sum()),
            "area_mm2": float(a.sum()),
            "bounds": [q.min(axis=(0, 1)).tolist(), q.max(axis=(0, 1)).tolist()],
        }
    bbox_min = node_xyz.min(axis=0).tolist()
    bbox_max = node_xyz.max(axis=0).tolist()
    node_mean = node_xyz.mean(axis=0)
    node_extent = np.ptp(node_xyz, axis=0)
    del node_xyz, quad_faces, fixed_mask_q, load_mask_q

    # Build GMG hierarchy (upload each level to GPU).
    progress("multigrid_setup")
    levels = []
    cur_grid = grid
    cur_ijk = node_ijk
    cur_keys = node_keys
    cur_bnodes = brick_nodes
    cur_colors = color_cells
    cur_fixed = fixed_mask_np

    # Temporary scratch for lambda_max estimation on L0 (will be reused as CG vectors).
    cg_x = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
    cg_r = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
    cg_p = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
    cg_ap = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
    cg_z = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)
    scratch_256 = wp.zeros(256, dtype=wp.float64, device=used_device)

    ke_l0_f64, ke_l0_f32, D_local = build_stiffness_lut(eff_material, dx, dy, dz, n_bins=n_bins, build_axis=build_axis)
    flag_lut_f64_dev = wp.array(flag_lut_f64, dtype=wp.float64, device=used_device)

    def _make_gpu_level(lvl_idx, g, bnodes, colors, fmask, ke32_np, lut_f32):
        nn = len(fmask)
        nd = 3 * nn
        bn_dev = wp.array(bnodes, dtype=wp.int32, device=used_device)
        bp_dev = wp.array(g.brick_payload, dtype=wp.uint32, device=used_device)
        cc_dev = [wp.array(c, dtype=wp.int32, device=used_device) for c in colors]
        ke_dev = wp.array(ke32_np, dtype=wp.float32, device=used_device)
        lut_dev = wp.array(lut_f32, dtype=wp.float32, device=used_device)
        fm_dev = wp.array(fmask, dtype=wp.uint8, device=used_device)
        ke_diag_np = np.ascontiguousarray(np.diagonal(ke32_np, axis1=1, axis2=2), dtype=np.float32)
        kd_dev = wp.array(ke_diag_np, dtype=wp.float32, device=used_device)
        diag_dev = wp.zeros(nd, dtype=wp.float32, device=used_device)
        for c_d in cc_dev:
            if len(c_d):
                wp.launch(_diag_color_f32_kernel, dim=len(c_d), inputs=[c_d, bn_dev, bp_dev, kd_dev, lut_dev, diag_dev], device=used_device)
        diag_host = diag_dev.numpy()
        free_d = ~np.repeat(fmask != 0, 3)
        if np.any(diag_host[free_d] <= 0):
            raise ValueError("自由自由度存在非正对角刚度")
        dinv_host = np.where(free_d & (diag_host > 0), 1.0 / np.maximum(diag_host, 1e-30), 0.0).astype(np.float32)
        dinv_dev = wp.array(dinv_host, dtype=wp.float32, device=used_device)
        del diag_dev, kd_dev, diag_host, dinv_host
        lvl_obj = _GPUMGLevel(
            grid=g,
            n_nodes=nn,
            n_dofs=nd,
            fixed_mask_np=fmask,
            brick_nodes_dev=bn_dev,
            brick_payload_dev=bp_dev,
            color_cells_dev=cc_dev,
            ke_f32_dev=ke_dev,
            flag_scale_lut_dev=lut_dev,
            fixed_mask_dev=fm_dev,
            diag_inv_dev=dinv_dev,
            lambda_max=2.0,
        )
        if lvl_idx > 0:
            lvl_obj.x_dev = wp.zeros(nd, dtype=wp.float32, device=used_device)
            lvl_obj.b_dev = wp.zeros(nd, dtype=wp.float32, device=used_device)
            lvl_obj.r_dev = wp.zeros(nd, dtype=wp.float32, device=used_device)
            lvl_obj.ax_dev = wp.zeros(nd, dtype=wp.float32, device=used_device)
            lvl_obj.d_dev = wp.zeros(nd, dtype=wp.float32, device=used_device)
        return lvl_obj

    lvl0 = _make_gpu_level(0, cur_grid, cur_bnodes, cur_colors, cur_fixed, ke_l0_f32, flag_lut_f32)
    lvl0.lambda_max = _estimate_lambda_max(lvl0, scratch_256, cg_x, cg_r, used_device)
    levels.append(lvl0)

    # Coarsen while free DOFs exceed coarse_dof_target and depth < max_levels.
    ke_cur_f64 = ke_l0_f64
    for lvl_idx in range(1, max_levels):
        cur_free_dofs = int(3 * (cur_fixed == 0).sum())
        if cur_free_dofs <= coarse_dof_target:
            break
        c_grid = coarsen_brick_grid(cur_grid, flag_scale_lut=flag_lut_f64 if lvl_idx == 1 else None)
        c_ijk, c_keys, c_bnodes, c_colors = build_level_topology(c_grid)
        f_parents, f_weights, c_fixed = build_prolongation_map(cur_ijk, cur_fixed, c_grid, c_keys)
        c_free_dofs = int(3 * (c_fixed == 0).sum())
        if c_free_dofs == 0 or c_free_dofs >= int(cur_free_dofs * 0.75):
            break
        levels[-1].fine_parents_dev = wp.array(f_parents, dtype=wp.int32, device=used_device)
        levels[-1].fine_weights_dev = wp.array(f_weights, dtype=wp.float32, device=used_device)

        ke_c_f64, ke_c_f32, _ = build_stiffness_lut(eff_material, *c_grid.pitch, n_bins=n_bins, build_axis=build_axis)
        c_lvl = _make_gpu_level(lvl_idx, c_grid, c_bnodes, c_colors, c_fixed, ke_c_f32, unit_lut_f32)
        c_lvl.lambda_max = _estimate_lambda_max(c_lvl, scratch_256, c_lvl.x_dev, c_lvl.r_dev, used_device)
        levels.append(c_lvl)

        cur_grid, cur_ijk, cur_keys, cur_bnodes, cur_colors, cur_fixed = (
            c_grid,
            c_ijk,
            c_keys,
            c_bnodes,
            c_colors,
            c_fixed,
        )
        ke_cur_f64 = ke_c_f64
        del f_parents, f_weights

    # Setup coarsest-level exact/AMG solver on CPU.
    coarsest = levels[-1]
    coarse_free_mask = np.repeat(coarsest.fixed_mask_np == 0, 3)
    coarse_free_idx = np.flatnonzero(coarse_free_mask)
    coarse_solver_name = "none"
    coarse_lu = None
    coarse_amg = None
    coarse_scale = None
    coarse_precond = None

    if len(levels) > 1 or len(coarse_free_idx) <= coarse_dof_target:
        K_c = assemble_coarse_csr(
            cur_grid,
            cur_bnodes,
            coarsest.n_nodes,
            ke_cur_f64,
            flag_scale_lut=flag_lut_f64 if len(levels) == 1 else None,
        )
        A_c = K_c[coarse_free_idx][:, coarse_free_idx].tocsc()
        if len(coarse_free_idx) <= 45000:
            coarse_lu = splu(A_c)
            coarse_solver_name = "splu"
        else:
            import pyamg

            c_xyz = cur_grid.origin[None, :] + cur_ijk.astype(np.float64) * cur_grid.pitch[None, :]
            diag_c = A_c.diagonal()
            coarse_scale = 1.0 / np.sqrt(np.maximum(diag_c, 1e-30))
            from scipy.sparse import diags as sp_diags

            S = sp_diags(coarse_scale)
            A_scaled = (S @ A_c @ S).tocsr().tobsr(blocksize=(3, 3))
            B_rbm = rigid_modes(c_xyz)[coarse_free_idx] / coarse_scale[:, None]
            coarse_amg = pyamg.smoothed_aggregation_solver(
                A_scaled, B=B_rbm, symmetry="symmetric", smooth="energy", max_coarse=120
            )
            coarse_precond = coarse_amg.aspreconditioner(cycle="V")
            coarse_solver_name = "pyamg_vcycle"
        del K_c, A_c

    def _solve_coarsest_f32(b_dev, x_dev):
        b_host = b_dev.numpy()
        rhs_c = b_host[coarse_free_idx].astype(np.float64)
        x_host = np.zeros(coarsest.n_dofs, dtype=np.float32)
        if coarse_lu is not None:
            sol_c = coarse_lu.solve(rhs_c)
        elif coarse_precond is not None:
            sol_c = coarse_scale * coarse_precond(coarse_scale * rhs_c)
        else:
            sol_c = np.zeros_like(rhs_c)
        x_host[coarse_free_idx] = sol_c.astype(np.float32)
        wp.copy(x_dev, wp.from_numpy(x_host, dtype=wp.float32))

    def _vcycle(lvl_idx, b_dev, x_dev, r_dev, d_dev, ax_dev):
        if lvl_idx == len(levels) - 1:
            _solve_coarsest_f32(b_dev, x_dev)
            return
        lvl = levels[lvl_idx]
        # Pre-smooth from zero initial guess: x_dev = S(b_dev)
        _chebyshev_smooth_f32(lvl, b_dev, x_dev, r_dev, d_dev, ax_dev, smooth_degree, used_device, zero_init=True)
        # Residual r = b - K x
        _launch_matvec_f32(lvl, x_dev, ax_dev, used_device)
        wp.launch(_axpby_f32_kernel, dim=lvl.n_dofs, inputs=[np.float32(1.0), b_dev, np.float32(-1.0), ax_dev, r_dev], device=used_device)
        wp.launch(_apply_fixed_mask_f32_kernel, dim=lvl.n_nodes, inputs=[lvl.fixed_mask_dev, r_dev], device=used_device)
        # Restrict r -> coarse b
        nxt = levels[lvl_idx + 1]
        nxt.b_dev.zero_()
        wp.launch(
            _restrict_f32_kernel,
            dim=lvl.n_nodes,
            inputs=[lvl.fine_parents_dev, lvl.fine_weights_dev, r_dev, nxt.b_dev],
            device=used_device,
        )
        wp.launch(_apply_fixed_mask_f32_kernel, dim=nxt.n_nodes, inputs=[nxt.fixed_mask_dev, nxt.b_dev], device=used_device)
        # Recurse
        _vcycle(lvl_idx + 1, nxt.b_dev, nxt.x_dev, nxt.r_dev, nxt.d_dev, nxt.ax_dev)
        # Prolongate x_coarse -> x_fine
        wp.launch(
            _prolong_add_f32_kernel,
            dim=lvl.n_nodes,
            inputs=[lvl.fine_parents_dev, lvl.fine_weights_dev, nxt.x_dev, x_dev],
            device=used_device,
        )
        # Post-smooth symmetrically
        _chebyshev_smooth_f32(lvl, b_dev, x_dev, r_dev, d_dev, ax_dev, smooth_degree, used_device, zero_init=False)

    # Allocate Level-0 fp64 LUT and temporary fp64 buffers (host u_64, GPU scratch x64/y64).
    ke_l0_f64_dev = wp.array(ke_l0_f64, dtype=wp.float64, device=used_device)
    u_64_host = np.zeros(n_dofs, dtype=np.float64)
    u64_dev = wp.zeros(n_dofs, dtype=wp.float64, device=used_device)
    y64_dev = wp.zeros(n_dofs, dtype=wp.float64, device=used_device)
    # Need one extra vector b0_dev on level 0 for V-cycle smoother scratch (use cg_d_tmp).
    cg_d_tmp = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)

    setup_seconds = time.perf_counter() - t_setup
    progress("linear_solve")
    t_solve = time.perf_counter()

    f_norm = float(np.linalg.norm(f_64[free_mask_np]))
    outer_iters = 0
    total_inner_iters = 0
    residual_history = []
    used_jacobi_fallback = False

    if f_norm > 0.0:
        # Handle tiny single-level problem directly if already solved on coarsest level.
        if len(levels) == 1 and coarse_lu is not None:
            u_64_host[coarse_free_idx] = coarse_lu.solve(f_64[coarse_free_idx])
            wp.copy(u64_dev, wp.from_numpy(u_64_host, dtype=wp.float64))
            _launch_matvec_f64(
                lvl0.color_cells_dev,
                lvl0.brick_nodes_dev,
                lvl0.brick_payload_dev,
                ke_l0_f64_dev,
                flag_lut_f64_dev,
                lvl0.fixed_mask_dev,
                n_nodes,
                u64_dev,
                y64_dev,
                used_device,
                apply_fixed_mask=False,
            )
            y_host = y64_dev.numpy()
            rel_res = float(np.linalg.norm(y_host[free_mask_np] - f_64[free_mask_np]) / f_norm)
            residual_history.append(rel_res)
            outer_iters = 1
        else:
            m_trunc = 4 if n_dofs <= 8_000_000 else 2
            f64_dev = wp.array(f_64, dtype=wp.float64, device=used_device)
            r64_dev = wp.array(f_64, dtype=wp.float64, device=used_device)
            wp.launch(_apply_fixed_mask_f64_kernel, dim=n_nodes, inputs=[lvl0.fixed_mask_dev, r64_dev], device=used_device)
            fcg_hist = []
            inner_cap = min(max_inner_iter, 28 if len(levels) > 1 else 60)
            cg_v_r = wp.zeros(n_dofs, dtype=wp.float32, device=used_device)

            def _apply_precond(rhs_dev, z_out_dev):
                if len(levels) > 1 and not used_jacobi_fallback:
                    _vcycle(0, rhs_dev, z_out_dev, cg_v_r, cg_d_tmp, cg_ap)
                else:
                    wp.launch(_diag_precond_f32_kernel, dim=n_dofs, inputs=[lvl0.fixed_mask_dev, lvl0.diag_inv_dev, rhs_dev, z_out_dev], device=used_device)

            r_norm = math.sqrt(max(_gpu_dot_f64(r64_dev, r64_dev, scratch_256, used_device), 0.0))
            rel_res = r_norm / f_norm
            residual_history.append(rel_res)

            for outer in range(max_outer_ir):
                if rel_res <= rtol:
                    _launch_matvec_f64(
                        lvl0.color_cells_dev,
                        lvl0.brick_nodes_dev,
                        lvl0.brick_payload_dev,
                        ke_l0_f64_dev,
                        flag_lut_f64_dev,
                        lvl0.fixed_mask_dev,
                        n_nodes,
                        u64_dev,
                        y64_dev,
                        used_device,
                        apply_fixed_mask=True,
                    )
                    wp.launch(_axpby_f64_kernel, dim=n_dofs, inputs=[np.float64(1.0), f64_dev, np.float64(-1.0), y64_dev, r64_dev], device=used_device)
                    wp.launch(_apply_fixed_mask_f64_kernel, dim=n_nodes, inputs=[lvl0.fixed_mask_dev, r64_dev], device=used_device)
                    r_norm = math.sqrt(max(_gpu_dot_f64(r64_dev, r64_dev, scratch_256, used_device), 0.0))
                    rel_res = r_norm / f_norm
                    if rel_res <= rtol:
                        break

                outer_iters = outer + 1
                cg_x.zero_()
                wp.launch(_scale_f64_to_f32_kernel, dim=n_dofs, inputs=[np.float64(1.0 / r_norm), r64_dev, cg_r], device=used_device)
                r0_norm_f32 = math.sqrt(max(_gpu_dot_f32(cg_r, cg_r, scratch_256, used_device), 1e-30))
                inner_target = 1.5e-2

                _apply_precond(cg_r, cg_z)
                rz_old = _gpu_dot_f32(cg_r, cg_z, scratch_256, used_device)
                if rz_old <= 0.0:
                    used_jacobi_fallback = True
                    _apply_precond(cg_r, cg_z)
                    rz_old = _gpu_dot_f32(cg_r, cg_z, scratch_256, used_device)
                wp.copy(cg_p, cg_z)

                for _it in range(1, inner_cap + 1):
                    total_inner_iters += 1
                    _launch_matvec_f32(lvl0, cg_p, cg_ap, used_device)
                    pAp = _gpu_dot_f32(cg_p, cg_ap, scratch_256, used_device)
                    if pAp <= 0.0:
                        break
                    alpha_f32 = np.float32(rz_old / pAp)
                    wp.launch(_axpby_f32_kernel, dim=n_dofs, inputs=[np.float32(1.0), cg_x, alpha_f32, cg_p, cg_x], device=used_device)
                    wp.launch(_axpby_f32_kernel, dim=n_dofs, inputs=[np.float32(1.0), cg_r, np.float32(-alpha_f32), cg_ap, cg_r], device=used_device)
                    r_cur_norm = math.sqrt(max(_gpu_dot_f32(cg_r, cg_r, scratch_256, used_device), 0.0))
                    if r_cur_norm / r0_norm_f32 <= inner_target:
                        break
                    _apply_precond(cg_r, cg_z)
                    rz_new = _gpu_dot_f32(cg_r, cg_z, scratch_256, used_device)
                    if rz_new <= 0.0:
                        break
                    beta_f32 = np.float32(rz_new / rz_old)
                    wp.launch(_axpby_f32_kernel, dim=n_dofs, inputs=[np.float32(1.0), cg_z, beta_f32, cg_p, cg_p], device=used_device)
                    rz_old = rz_new

                if len(fcg_hist) >= m_trunc:
                    p_dev, q_dev, _ = fcg_hist.pop(0)
                else:
                    p_dev = wp.zeros(n_dofs, dtype=wp.float64, device=used_device)
                    q_dev = wp.zeros(n_dofs, dtype=wp.float64, device=used_device)

                wp.launch(
                    _scale_f32_to_f64_masked_kernel,
                    dim=n_dofs,
                    inputs=[np.float64(r_norm), lvl0.fixed_mask_dev, cg_x, p_dev],
                    device=used_device,
                )
                for pj, qj, dj in fcg_hist:
                    beta_j = _gpu_dot_f64(p_dev, qj, scratch_256, used_device) / dj
                    wp.launch(_axpby_f64_kernel, dim=n_dofs, inputs=[np.float64(1.0), p_dev, np.float64(-beta_j), pj, p_dev], device=used_device)
                wp.launch(_apply_fixed_mask_f64_kernel, dim=n_nodes, inputs=[lvl0.fixed_mask_dev, p_dev], device=used_device)

                _launch_matvec_f64(
                    lvl0.color_cells_dev,
                    lvl0.brick_nodes_dev,
                    lvl0.brick_payload_dev,
                    ke_l0_f64_dev,
                    flag_lut_f64_dev,
                    lvl0.fixed_mask_dev,
                    n_nodes,
                    p_dev,
                    q_dev,
                    used_device,
                    apply_fixed_mask=True,
                )
                delta_i = _gpu_dot_f64(p_dev, q_dev, scratch_256, used_device)
                pr = _gpu_dot_f64(p_dev, r64_dev, scratch_256, used_device)
                if not math.isfinite(delta_i) or delta_i <= 0.0 or pr <= 0.0:
                    fcg_hist.clear()
                    wp.launch(
                        _scale_f32_to_f64_masked_kernel,
                        dim=n_dofs,
                        inputs=[np.float64(r_norm), lvl0.fixed_mask_dev, cg_x, p_dev],
                        device=used_device,
                    )
                    _launch_matvec_f64(
                        lvl0.color_cells_dev,
                        lvl0.brick_nodes_dev,
                        lvl0.brick_payload_dev,
                        ke_l0_f64_dev,
                        flag_lut_f64_dev,
                        lvl0.fixed_mask_dev,
                        n_nodes,
                        p_dev,
                        q_dev,
                        used_device,
                        apply_fixed_mask=True,
                    )
                    delta_i = _gpu_dot_f64(p_dev, q_dev, scratch_256, used_device)
                    pr = _gpu_dot_f64(p_dev, r64_dev, scratch_256, used_device)
                    if not math.isfinite(delta_i) or delta_i <= 0.0 or pr <= 0.0:
                        del p_dev, q_dev
                        break

                alpha64 = np.float64(pr / delta_i)
                wp.launch(_axpby_f64_kernel, dim=n_dofs, inputs=[np.float64(1.0), u64_dev, alpha64, p_dev, u64_dev], device=used_device)
                wp.launch(_axpby_f64_kernel, dim=n_dofs, inputs=[np.float64(1.0), r64_dev, np.float64(-alpha64), q_dev, r64_dev], device=used_device)
                fcg_hist.append((p_dev, q_dev, float(delta_i)))

                r_norm = math.sqrt(max(_gpu_dot_f64(r64_dev, r64_dev, scratch_256, used_device), 0.0))
                rel_res = r_norm / f_norm
                if (outer_iters % 6 == 0) or (rel_res <= rtol):
                    _launch_matvec_f64(
                        lvl0.color_cells_dev,
                        lvl0.brick_nodes_dev,
                        lvl0.brick_payload_dev,
                        ke_l0_f64_dev,
                        flag_lut_f64_dev,
                        lvl0.fixed_mask_dev,
                        n_nodes,
                        u64_dev,
                        y64_dev,
                        used_device,
                        apply_fixed_mask=True,
                    )
                    wp.launch(_axpby_f64_kernel, dim=n_dofs, inputs=[np.float64(1.0), f64_dev, np.float64(-1.0), y64_dev, r64_dev], device=used_device)
                    wp.launch(_apply_fixed_mask_f64_kernel, dim=n_nodes, inputs=[lvl0.fixed_mask_dev, r64_dev], device=used_device)
                    r_norm = math.sqrt(max(_gpu_dot_f64(r64_dev, r64_dev, scratch_256, used_device), 0.0))
                    rel_res = r_norm / f_norm
                residual_history.append(rel_res)
                if rel_res <= rtol:
                    break

            u_64_host = u64_dev.numpy()
            del fcg_hist, f64_dev, r64_dev

    solve_seconds = time.perf_counter() - t_solve

    # Final unmasked fp64 audit: computes exact unscaled free-DOF residual AND fixed-node reactions in 1 matvec!
    wp.copy(u64_dev, wp.from_numpy(u_64_host, dtype=wp.float64))
    _launch_matvec_f64(
        lvl0.color_cells_dev,
        lvl0.brick_nodes_dev,
        lvl0.brick_payload_dev,
        ke_l0_f64_dev,
        flag_lut_f64_dev,
        lvl0.fixed_mask_dev,
        n_nodes,
        u64_dev,
        y64_dev,
        used_device,
        apply_fixed_mask=False,
    )
    y_final = y64_dev.numpy()
    residual_vec = y_final - f_64
    final_rel_res = float(np.linalg.norm(residual_vec[free_mask_np]) / max(f_norm, 1e-20))
    if not np.isfinite(u_64_host).all() or final_rel_res > 1e-6:
        raise ValueError(f"GPU H8 求解未通过原始系统 fp64 残差校核：relative_residual={final_rel_res:.3g}")

    reactions = residual_vec.reshape(-1, 3)[fixed_nodes]
    reaction_sum = reactions.sum(axis=0)
    rel_force_err = float(np.linalg.norm(reaction_sum + force) / max(np.linalg.norm(force), 1.0))
    moment_vec = np.cross(loaded_xyz - node_mean, loaded_forces_compact).sum(axis=0) + np.cross(
        fixed_xyz - node_mean, reactions
    ).sum(axis=0)
    rel_moment_err = float(np.linalg.norm(moment_vec) / max(np.linalg.norm(force) * float(node_extent.max()), 1.0))
    if rel_force_err >= 1e-6 or rel_moment_err >= 1e-6:
        raise ValueError(
            f"GPU H8 求解未通过平衡校核：force_err={rel_force_err:.3g}, moment_err={rel_moment_err:.3g}"
        )

    u_nodes = u_64_host.reshape(-1, 3)
    disp_mag = np.linalg.norm(u_nodes, axis=1)
    max_disp = float(disp_mag.max())
    strain_energy = float(0.5 * np.dot(u_64_host, f_64))
    force_norm_val = float(np.linalg.norm(force))
    work_conj_disp = float(2.0 * strain_energy / force_norm_val) if force_norm_val > 0 else None
    compliance = float(2.0 * strain_energy / (force_norm_val ** 2)) if force_norm_val > 0 else None

    vram_used_mib = None
    vram_peak_mib = None
    vram_free_mib = None
    if used_device.startswith("cuda"):
        try:
            vram_used_mib = float(wp.get_mempool_used_mem_current(used_device) / 1024**2)
            vram_peak_mib = float(wp.get_mempool_used_mem_high(used_device) / 1024**2)
            vram_free_mib = float(wp.get_device(used_device).free_memory / 1024**2)
        except Exception:
            pass

    level_summary = [
        {
            "level": idx,
            "pitch_mm": [float(v) for v in lvl.grid.pitch],
            "bricks": lvl.grid.num_bricks,
            "voxels": lvl.grid.num_voxels,
            "nodes": lvl.n_nodes,
            "free_dofs": int(3 * (lvl.fixed_mask_np == 0).sum()),
            "lambda_max": float(lvl.lambda_max),
        }
        for idx, lvl in enumerate(levels)
    ]

    # Free GPU solver buffers before streaming stress recovery so VRAM stays minimal.
    del levels, lvl0, cg_x, cg_r, cg_p, cg_ap, cg_z, cg_d_tmp, scratch_256, u64_dev, y64_dev, ke_l0_f64_dev, flag_lut_f64_dev
    if "cg_v_r" in locals():
        del cg_v_r
    if used_device.startswith("cuda"):
        try:
            wp.synchronize_device(used_device)
        except Exception:
            pass

    rss_mib = float(psutil.Process().memory_info().rss / 1024**2)

    solver_info = {
        "device": used_device,
        "nodes": n_nodes,
        "elements": int(grid.num_voxels),
        "free_dofs": int(free_mask_np.sum()),
        "levels": level_summary,
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
        "reaction_N": reaction_sum.tolist(),
        "regions": quad_area_stats,
        "bounds_mm": [bbox_min, bbox_max],
        "interface_config": interface_cfg.to_dict(),
        "process_rss_mib": rss_mib,
        "vram_reported_mib": vram_used_mib,
        "vram_peak_mib": vram_peak_mib,
        "vram_free_after_solve_mib": vram_free_mib,
    }
    return {
        "displacement": u_nodes,
        "brick_nodes": brick_nodes,
        "node_ijk": node_ijk,
        "fixed_nodes": fixed_nodes,
        "loaded_nodes": loaded_nodes,
        "reactions": reactions,
        "forces": f_64.reshape(-1, 3),
        "material": material,
        "build_axis": build_axis,
        "solver_info": solver_info,
    }


def recover_stress_streaming(
    grid: BrickGrid,
    brick_nodes,
    displacement,
    material=None,
    interface_cfg: HexInterfaceConfig = None,
    build_axis="Z",
    chunk_bricks=8192,
    spill_path=None,
):
    """Stream active bricks in bounded chunks to compute centroid orthotropic stresses and hotspots."""
    t0 = time.perf_counter()
    material = material or Material()
    interface_cfg = interface_cfg or HexInterfaceConfig()
    eff_material = interface_cfg.effective_material(material)
    flag_lut = interface_cfg.flag_scale_lut()
    D_local = local_stiffness(eff_material)
    dx, dy, dz = (float(x) for x in grid.pitch)
    _, B0 = hex_B_matrices(dx, dy, dz)
    n_bins = int(grid.meta.get("n_angle_bins", 360))
    corner_offs = np.array([0, 1, 5, 6, 25, 26, 30, 31], dtype=np.int32)

    n_vox = grid.num_voxels
    vm_all = np.empty(n_vox, dtype=np.float32)
    vm_unscaled_all = np.empty(n_vox, dtype=np.float32)
    s_along_all = np.empty(n_vox, dtype=np.float32)
    s_layer_all = np.empty(n_vox, dtype=np.float32)
    shear_z_all = np.empty(n_vox, dtype=np.float32)
    feat_all = np.empty(n_vox, dtype=np.uint8)
    flags_all = np.empty(n_vox, dtype=np.uint8)

    # Track global top-20 von Mises & top-20 interlayer normal stress hotspots
    top_k = 20
    top_vm = []
    top_sz = []

    spill_fh = None
    if spill_path is not None:
        spill_path = Path(spill_path)
        spill_path.parent.mkdir(parents=True, exist_ok=True)
        spill_fh = spill_path.open("wb")
        # Header: magic + voxel count; each voxel writes 5 float32s [vm, s_along, s_trans, s_layer, shear_z]
        spill_fh.write(b"TPSTRS1\x00" + struct.pack("<Q", n_vox))

    write_pos = 0
    n_bricks = len(grid.brick_coords)
    try:
        for b_start in range(0, n_bricks, chunk_bricks):
            b_stop = min(n_bricks, b_start + chunk_bricks)
            coords_ch = grid.brick_coords[b_start:b_stop]
            masks_ch = grid.brick_masks[b_start:b_stop]
            payload_ch = grid.brick_payload[b_start:b_stop]
            bn_ch = brick_nodes[b_start:b_stop]

            # Gather active cells in canonical (brick, cell) order for this chunk.
            occ_2d = ((masks_ch[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)) != 0
            ab, ac = np.nonzero(occ_2d)
            if not len(ab):
                continue
            ab = ab.astype(np.int32)
            ac = ac.astype(np.int32)
            m = len(ab)

            lx = ac & 3
            ly = (ac >> 2) & 3
            lz = ac >> 4
            bases = lx + 5 * ly + 25 * lz
            nids = bn_ch[ab[:, None], bases[:, None] + corner_offs[None, :]]
            u_e = displacement[nids].reshape(m, 24)
            p_ch = payload_ch[ab, ac]
            abin, _, flags = unpack_payload(p_ch)
            angles = bin_to_angle(abin, n_bins)
            feats = decode_feature(flags).astype(np.uint8)
            fscale = flag_lut[(flags >> 5) & 7]

            local_s, global_s = hex_centroid_stress(u_e, D_local, B0, angles, build_axis=build_axis)
            sx, sy, sz, txy, tyz, txz = global_s.T
            vm_raw = np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3.0 * (txy * txy + tyz * tyz + txz * txz))
            shear_z = np.hypot(local_s[:, 4], local_s[:, 5])

            vm_32 = (vm_raw * fscale).astype(np.float32)
            sa_32 = (local_s[:, 0] * fscale).astype(np.float32)
            st_32 = (local_s[:, 1] * fscale).astype(np.float32)
            sl_32 = (local_s[:, 2] * fscale).astype(np.float32)
            sh_32 = (shear_z * fscale).astype(np.float32)

            vm_all[write_pos : write_pos + m] = vm_32
            vm_unscaled_all[write_pos : write_pos + m] = vm_raw.astype(np.float32)
            s_along_all[write_pos : write_pos + m] = sa_32
            s_layer_all[write_pos : write_pos + m] = sl_32
            shear_z_all[write_pos : write_pos + m] = sh_32
            feat_all[write_pos : write_pos + m] = feats
            flags_all[write_pos : write_pos + m] = flags.astype(np.uint8)
            write_pos += m

            if spill_fh is not None:
                packed = np.column_stack([vm_32, sa_32, st_32, sl_32, sh_32]).astype("<f4", copy=False)
                spill_fh.write(packed.tobytes())

            # Update top-K hotspots with exact physical center coordinates
            centers_x = grid.origin[0] + (4 * coords_ch[ab, 0] + lx + 0.5) * dx
            centers_y = grid.origin[1] + (4 * coords_ch[ab, 1] + ly + 0.5) * dy
            centers_z = grid.origin[2] + (4 * coords_ch[ab, 2] + lz + 0.5) * dz

            k_loc = min(top_k, m)
            idx_vm = np.argpartition(vm_32, -k_loc)[-k_loc:]
            for idx in idx_vm:
                top_vm.append((float(vm_32[idx]), [float(centers_x[idx]), float(centers_y[idx]), float(centers_z[idx])], FEATURE_NAMES.get(int(feats[idx]), "unknown")))
            top_vm.sort(key=lambda t: t[0], reverse=True)
            top_vm = top_vm[:top_k]

            idx_sz = np.argpartition(sl_32, -k_loc)[-k_loc:]
            for idx in idx_sz:
                top_sz.append((float(sl_32[idx]), [float(centers_x[idx]), float(centers_y[idx]), float(centers_z[idx])], FEATURE_NAMES.get(int(feats[idx]), "unknown")))
            top_sz.sort(key=lambda t: t[0], reverse=True)
            top_sz = top_sz[:top_k]
    finally:
        if spill_fh is not None:
            spill_fh.close()

    by_feature = {}
    for code, fname in FEATURE_NAMES.items():
        msk = feat_all == code
        if not msk.any():
            continue
        by_feature[fname] = {
            "voxels": int(msk.sum()),
            "von_mises_max_MPa": float(vm_all[msk].max()),
            "von_mises_p95_MPa": float(np.percentile(vm_all[msk], 95)),
            "sigma_layer_max_MPa": float(s_layer_all[msk].max()),
            "sigma_along_max_abs_MPa": float(np.abs(s_along_all[msk]).max()),
            "interlayer_shear_max_MPa": float(shear_z_all[msk].max()),
        }

    m_seam = (flags_all & FLAG_SEAM_LOOP) != 0
    m_ss = (flags_all & FLAG_START_STOP_ZONE) != 0
    m_iface = (flags_all & FLAG_BEAD_INTERFACE) != 0
    m_interior = ~(m_seam | m_ss | m_iface)
    by_zone = {}
    for zname, zmask in (
        ("seam_loop", m_seam),
        ("start_stop_zone", m_ss),
        ("bead_interface", m_iface),
        ("interior", m_interior),
    ):
        if zmask.any():
            by_zone[zname] = {
                "voxels": int(zmask.sum()),
                "von_mises_max_MPa": float(vm_all[zmask].max()),
                "von_mises_p95_MPa": float(np.percentile(vm_all[zmask], 95)),
                "strain_equiv_vm_max_MPa": float(vm_unscaled_all[zmask].max()),
            }

    return {
        "voxel_count": int(write_pos),
        "von_mises_max_MPa": float(vm_all.max()),
        "von_mises_p95_MPa": float(np.percentile(vm_all, 95)),
        "sigma_along_max_MPa": float(s_along_all.max()),
        "sigma_along_min_MPa": float(s_along_all.min()),
        "sigma_layer_max_MPa": float(s_layer_all.max()),
        "sigma_layer_min_MPa": float(s_layer_all.min()),
        "interlayer_shear_max_MPa": float(shear_z_all.max()),
        "interlayer_shear_p95_MPa": float(np.percentile(shear_z_all, 95)),
        "by_feature": by_feature,
        "by_zone": by_zone,
        "top_von_mises_hotspots": [{"stress_MPa": v, "xyz_mm": c, "feature": f} for v, c, f in top_vm],
        "top_interlayer_tension_hotspots": [{"sigma_layer_MPa": v, "xyz_mm": c, "feature": f} for v, c, f in top_sz],
        "recovery_seconds": time.perf_counter() - t0,
        "spill_path": str(spill_path.resolve()) if spill_path is not None else None,
        # Keep float32 summary vectors for compact 2D projection preview
        "_vm_f32": vm_all,
        "_sl_f32": s_layer_all,
    }
