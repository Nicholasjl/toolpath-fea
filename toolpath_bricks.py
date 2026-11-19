"""Streaming G-code rasterization to 4x4x4 sparse bricks with atomic disk spill."""
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time
import uuid

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from hex_fem import angle_to_bin


BRICK_SIZE = 4
BRICK_VOLUME = 64
MAGIC_HEADER = b"TPBRICK1"

FEATURE_UNKNOWN = 0
FEATURE_OUTER_WALL = 1
FEATURE_INNER_WALL = 2
FEATURE_SOLID_INFILL = 3
FEATURE_SPARSE_INFILL = 4
FEATURE_BRIDGE = 5

FEATURE_NAMES = {
    FEATURE_UNKNOWN: "unknown",
    FEATURE_OUTER_WALL: "outer_wall",
    FEATURE_INNER_WALL: "inner_wall",
    FEATURE_SOLID_INFILL: "solid_infill",
    FEATURE_SPARSE_INFILL: "sparse_infill",
    FEATURE_BRIDGE: "bridge",
}

FEATURE_PRIORITY = {
    FEATURE_UNKNOWN: 0,
    FEATURE_SPARSE_INFILL: 1,
    FEATURE_SOLID_INFILL: 2,
    FEATURE_BRIDGE: 3,
    FEATURE_INNER_WALL: 4,
    FEATURE_OUTER_WALL: 5,
}

FLAG_BEAD_OVERLAP = 1 << 0
FLAG_MULTI_LAYER = 1 << 4

_SKIP_FEATURE_KEYWORDS = (
    "skirt",
    "brim",
    "prime tower",
    "prime_tower",
    "wipe tower",
    "wipe_tower",
    "support",
    "custom",
)

_WORD_RE = re.compile(r"([A-Z])\s*([-+]?(?:\d*\.\d+|\d+\.?\d*))")
_PAREN_RE = re.compile(r"\([^)]*\)")

# Intra-brick bit masks for 4x4x4 local index c = lx + 4*ly + 16*lz.
_ALL_64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_MASK_X0 = np.uint64(0)
_MASK_X3 = np.uint64(0)
_MASK_Y0 = np.uint64(0)
_MASK_Y3 = np.uint64(0)
_MASK_Z0 = np.uint64(0)
_MASK_Z3 = np.uint64(0)
for _lz in range(4):
    for _ly in range(4):
        for _lx in range(4):
            _bit = np.uint64(1) << np.uint64(_lx + 4 * _ly + 16 * _lz)
            if _lx == 0:
                _MASK_X0 |= _bit
            if _lx == 3:
                _MASK_X3 |= _bit
            if _ly == 0:
                _MASK_Y0 |= _bit
            if _ly == 3:
                _MASK_Y3 |= _bit
            if _lz == 0:
                _MASK_Z0 |= _bit
            if _lz == 3:
                _MASK_Z3 |= _bit
_NOT_X0 = _ALL_64 ^ _MASK_X0
_NOT_X3 = _ALL_64 ^ _MASK_X3
_NOT_Y0 = _ALL_64 ^ _MASK_Y0
_NOT_Y3 = _ALL_64 ^ _MASK_Y3
_NOT_Z0 = _ALL_64 ^ _MASK_Z0
_NOT_Z3 = _ALL_64 ^ _MASK_Z3

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)

_WARP_INITIALIZED = False


def _ensure_warp(device="auto"):
    import warp as wp

    global _WARP_INITIALIZED
    if not _WARP_INITIALIZED:
        wp.init()
        _WARP_INITIALIZED = True
    if device == "auto":
        return "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    return device


try:
    import warp as wp

    @wp.func
    def _segment_cell_mask16(
        x0: wp.float32,
        y0: wp.float32,
        vx: wp.float32,
        vy: wp.float32,
        inv_len2: wp.float32,
        r2: wp.float32,
        ix: wp.int32,
        iy: wp.int32,
        dx: wp.float32,
        dy: wp.float32,
    ) -> wp.uint32:
        m16 = wp.uint32(0)
        bx = wp.float32(ix) * dx
        by = wp.float32(iy) * dy
        for sy in range(4):
            py = by + (wp.float32(sy) + wp.float32(0.5)) * wp.float32(0.25) * dy
            dy_p = py - y0
            for sx in range(4):
                px = bx + (wp.float32(sx) + wp.float32(0.5)) * wp.float32(0.25) * dx
                dx_p = px - x0
                t = wp.clamp((dx_p * vx + dy_p * vy) * inv_len2, wp.float32(0.0), wp.float32(1.0))
                qx = t * vx - dx_p
                qy = t * vy - dy_p
                if qx * qx + qy * qy <= r2:
                    bit = sy * 4 + sx
                    m16 = m16 | (wp.uint32(1) << wp.uint32(bit))
        return m16

    @wp.kernel
    def _rasterize_segments_2d_kernel(
        segments: wp.array(dtype=wp.float32, ndim=2),
        nx: wp.int32,
        ny: wp.int32,
        dx: wp.float32,
        dy: wp.float32,
        submask: wp.array(dtype=wp.uint32, ndim=1),
        cos2_sum: wp.array(dtype=wp.float32, ndim=1),
        sin2_sum: wp.array(dtype=wp.float32, ndim=1),
        cov_sum: wp.array(dtype=wp.int32, ndim=1),
        feat_arr: wp.array(dtype=wp.int32, ndim=1),
    ):
        tid = wp.tid()
        x0 = segments[tid, 0]
        y0 = segments[tid, 1]
        x1 = segments[tid, 2]
        y1 = segments[tid, 3]
        r = segments[tid, 4]
        feat_prio = wp.int32(segments[tid, 5])
        vx = x1 - x0
        vy = y1 - y0
        len2 = vx * vx + vy * vy
        if len2 <= wp.float32(1e-16):
            return
        theta = wp.atan2(vy, vx)
        cos2 = wp.cos(wp.float32(2.0) * theta)
        sin2 = wp.sin(wp.float32(2.0) * theta)
        r2 = r * r
        inv_len2 = wp.float32(1.0) / len2

        if wp.abs(vx) >= wp.abs(vy):
            ix_min = wp.max(0, wp.int32(wp.floor((wp.min(x0, x1) - r) / dx)))
            ix_max = wp.min(nx - 1, wp.int32(wp.floor((wp.max(x0, x1) + r) / dx)))
            for ix in range(ix_min, ix_max + 1):
                cx0 = wp.float32(ix) * dx - r
                cx1 = wp.float32(ix + 1) * dx + r
                if wp.abs(vx) > wp.float32(1e-12):
                    t0 = (cx0 - x0) / vx
                    t1 = (cx1 - x0) / vx
                    ta = wp.clamp(wp.min(t0, t1), wp.float32(0.0), wp.float32(1.0))
                    tb = wp.clamp(wp.max(t0, t1), wp.float32(0.0), wp.float32(1.0))
                else:
                    ta = wp.float32(0.0)
                    tb = wp.float32(1.0)
                ya = y0 + ta * vy
                yb = y0 + tb * vy
                iy_min = wp.max(0, wp.int32(wp.floor((wp.min(ya, yb) - r) / dy)))
                iy_max = wp.min(ny - 1, wp.int32(wp.floor((wp.max(ya, yb) + r) / dy)))
                for iy in range(iy_min, iy_max + 1):
                    m16 = _segment_cell_mask16(x0, y0, vx, vy, inv_len2, r2, ix, iy, dx, dy)
                    if m16 != wp.uint32(0):
                        pop = wp.bit_count(wp.int32(m16))
                        idx = iy * nx + ix
                        wp.atomic_or(submask, idx, m16)
                        w = wp.float32(pop)
                        wp.atomic_add(cos2_sum, idx, w * cos2)
                        wp.atomic_add(sin2_sum, idx, w * sin2)
                        wp.atomic_add(cov_sum, idx, pop)
                        wp.atomic_max(feat_arr, idx, feat_prio)
        else:
            iy_min = wp.max(0, wp.int32(wp.floor((wp.min(y0, y1) - r) / dy)))
            iy_max = wp.min(ny - 1, wp.int32(wp.floor((wp.max(y0, y1) + r) / dy)))
            for iy in range(iy_min, iy_max + 1):
                cy0 = wp.float32(iy) * dy - r
                cy1 = wp.float32(iy + 1) * dy + r
                if wp.abs(vy) > wp.float32(1e-12):
                    t0 = (cy0 - y0) / vy
                    t1 = (cy1 - y0) / vy
                    ta = wp.clamp(wp.min(t0, t1), wp.float32(0.0), wp.float32(1.0))
                    tb = wp.clamp(wp.max(t0, t1), wp.float32(0.0), wp.float32(1.0))
                else:
                    ta = wp.float32(0.0)
                    tb = wp.float32(1.0)
                xa = x0 + ta * vx
                xb = x0 + tb * vx
                ix_min = wp.max(0, wp.int32(wp.floor((wp.min(xa, xb) - r) / dx)))
                ix_max = wp.min(nx - 1, wp.int32(wp.floor((wp.max(xa, xb) + r) / dx)))
                for ix in range(ix_min, ix_max + 1):
                    m16 = _segment_cell_mask16(x0, y0, vx, vy, inv_len2, r2, ix, iy, dx, dy)
                    if m16 != wp.uint32(0):
                        pop = wp.bit_count(wp.int32(m16))
                        idx = iy * nx + ix
                        wp.atomic_or(submask, idx, m16)
                        w = wp.float32(pop)
                        wp.atomic_add(cos2_sum, idx, w * cos2)
                        wp.atomic_add(sin2_sum, idx, w * sin2)
                        wp.atomic_add(cov_sum, idx, pop)
                        wp.atomic_max(feat_arr, idx, feat_prio)

    @wp.kernel
    def _pack_brick_slab_kernel(
        nx: wp.int32,
        ny: wp.int32,
        nbx: wp.int32,
        n_bins: wp.int32,
        submask: wp.array(dtype=wp.uint32, ndim=2),
        cos2_sum: wp.array(dtype=wp.float32, ndim=2),
        sin2_sum: wp.array(dtype=wp.float32, ndim=2),
        cov_sum: wp.array(dtype=wp.int32, ndim=2),
        feat_arr: wp.array(dtype=wp.int32, ndim=2),
        multi_arr: wp.array(dtype=wp.int32, ndim=2),
        slab_masks: wp.array(dtype=wp.uint64, ndim=1),
        slab_payload: wp.array(dtype=wp.uint32, ndim=2),
    ):
        bid = wp.tid()
        by = bid // nbx
        bx = bid - by * nbx
        base_x = bx * 4
        base_y = by * 4
        m64 = wp.uint64(0)
        pi = wp.float32(3.141592653589793)
        for lz in range(4):
            for ly in range(4):
                iy = base_y + ly
                row_off = iy * nx + base_x
                for lx in range(4):
                    idx = row_off + lx
                    sm = submask[lz, idx]
                    c = lz * 16 + ly * 4 + lx
                    if sm != wp.uint32(0):
                        m64 = m64 | (wp.uint64(1) << wp.uint64(c))
                        pop = wp.bit_count(wp.int32(sm & wp.uint32(0xFFFF)))
                        rho_q = (pop * 255 + 8) // 16
                        if rho_q > 255:
                            rho_q = 255
                        c2 = cos2_sum[lz, idx]
                        s2 = sin2_sum[lz, idx]
                        theta = wp.float32(0.5) * wp.atan2(s2, c2)
                        if theta < wp.float32(0.0):
                            theta = theta + pi
                        bin_f = wp.rint(theta * (wp.float32(n_bins) / pi))
                        abin = wp.int32(bin_f) % n_bins
                        if abin < 0:
                            abin = abin + n_bins
                        feat_code = feat_arr[lz, idx] & 0x07
                        flags = (feat_code << 1)
                        if cov_sum[lz, idx] > pop:
                            flags = flags | 1
                        if multi_arr[lz, idx] > 1:
                            flags = flags | 16
                        packed = (
                            wp.uint32(abin)
                            | (wp.uint32(rho_q) << wp.uint32(16))
                            | (wp.uint32(flags) << wp.uint32(24))
                        )
                        slab_payload[bid, c] = packed
                    else:
                        slab_payload[bid, c] = wp.uint32(0)
        slab_masks[bid] = m64

except Exception:  # pragma: no cover - optional runtime check if warp unavailable
    wp = None


def popcount_uint64(masks):
    """Exact popcount for uint64 NumPy arrays."""
    arr = np.ascontiguousarray(masks, dtype=np.uint64)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.int32)
    raw = arr.view(np.uint8).reshape(arr.shape + (8,))
    return _POPCOUNT_LUT[raw].sum(axis=-1, dtype=np.int32)


def pack_payload(angle_bin, rho_q, flags=0):
    """Pack (angle_bin[0..65535], rho_q[0..255], flags[0..255]) into uint32."""
    return (
        (np.asarray(angle_bin, dtype=np.uint32) & np.uint32(0xFFFF))
        | ((np.asarray(rho_q, dtype=np.uint32) & np.uint32(0xFF)) << np.uint32(16))
        | ((np.asarray(flags, dtype=np.uint32) & np.uint32(0xFF)) << np.uint32(24))
    )


def unpack_payload(payload):
    """Unpack uint32 payload into (angle_bin, rho_q, flags)."""
    p = np.asarray(payload, dtype=np.uint32)
    angle_bin = (p & np.uint32(0xFFFF)).astype(np.int32)
    rho_q = ((p >> np.uint32(16)) & np.uint32(0xFF)).astype(np.int32)
    flags = ((p >> np.uint32(24)) & np.uint32(0xFF)).astype(np.int32)
    return angle_bin, rho_q, flags


def decode_feature(flags):
    """Extract feature code (0..5) from packed flags."""
    return (np.asarray(flags, dtype=np.int32) >> 1) & 0x07


@dataclass
class BrickGrid:
    """Compact 4x4x4 sparse brick representation of toolpath voxels."""

    origin: np.ndarray
    pitch: np.ndarray
    shape: tuple
    brick_coords: np.ndarray
    brick_masks: np.ndarray
    brick_payload: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def num_bricks(self):
        return int(len(self.brick_coords))

    @property
    def num_voxels(self):
        return int(popcount_uint64(self.brick_masks).sum())

    @property
    def brick_shape(self):
        return tuple(int(s // BRICK_SIZE) for s in self.shape)


def _classify_feature(label):
    low = label.strip().lower()
    if not low:
        return FEATURE_UNKNOWN, False
    is_skip = any(key in low for key in _SKIP_FEATURE_KEYWORDS)
    if "outer" in low or "external" in low:
        return FEATURE_OUTER_WALL, is_skip
    if "inner" in low or "perimeter" in low or "wall" in low:
        return FEATURE_INNER_WALL, is_skip
    if "bridge" in low or "overhang" in low or "floating vertical shell" in low:
        return FEATURE_BRIDGE, is_skip
    if "solid" in low or "top" in low or "bottom" in low or "gap" in low or "ironing" in low:
        return FEATURE_SOLID_INFILL, is_skip
    if "infill" in low:
        return FEATURE_SPARSE_INFILL, is_skip
    return FEATURE_UNKNOWN, is_skip


def iter_gcode_layers(
    path,
    offset=(0.0, 0.0, 0.0),
    default_width_mm=0.42,
    default_layer_height_mm=0.2,
    filament_diameter_mm=1.75,
    skip_features=True,
):
    """Stream G-code line-by-line and yield (layer_z, layer_h, segments_np) per layer.

    Each row of segments_np is [x0, y0, x1, y1, width, feature_code, bead_id].
    Non-extruding Z-hops do not trigger layer switches or non-planar rejections.
    """
    off = np.asarray(offset, dtype=np.float64)
    if off.shape != (3,) or not np.isfinite(off).all():
        raise ValueError("offset must be a finite 3-vector")
    if default_width_mm <= 0 or default_layer_height_mm <= 0 or filament_diameter_mm <= 0:
        raise ValueError("default_width_mm, default_layer_height_mm, filament_diameter_mm must be positive")

    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0, "E": 0.0}
    absolute = True
    e_absolute = True
    explicit_width = None
    current_layer_h = float(default_layer_height_mm)
    filament_area = math.pi * (0.5 * float(filament_diameter_mm)) ** 2
    feature_code = FEATURE_UNKNOWN
    feature_skipped = False
    bead_id = 0
    in_bead = False
    current_layer_z = None
    layer_h_used = current_layer_h
    layer_segs = []
    yielded_any = False

    with open(path, "r", encoding="utf-8", errors="replace") as source:
        for raw_line in source:
            semi = raw_line.find(";")
            if semi != -1:
                comment = raw_line[semi + 1 :].strip()
                code_part = raw_line[:semi]
                if comment:
                    upper_c = comment.upper()
                    if upper_c.startswith("LAYER_HEIGHT:"):
                        try:
                            val = float(comment.split(":", 1)[1].strip())
                            if 0.01 <= val <= 2.0:
                                current_layer_h = val
                        except ValueError:
                            pass
                    elif upper_c.startswith("LINE_WIDTH:") or upper_c.startswith("WIDTH:"):
                        try:
                            val = float(comment.split(":", 1)[1].strip())
                            explicit_width = val if val > 0 else None
                        except ValueError:
                            explicit_width = None
                    elif upper_c.startswith("FEATURE:") or upper_c.startswith("TYPE:"):
                        new_code, new_skip = _classify_feature(comment.split(":", 1)[1])
                        if new_code != feature_code or new_skip != feature_skipped:
                            in_bead = False
                        feature_code = new_code
                        feature_skipped = new_skip
                    elif upper_c.startswith("FILAMENT_DIAMETER:"):
                        try:
                            val = float(comment.split(":", 1)[1].strip())
                            if 0.5 <= val <= 5.0:
                                filament_area = math.pi * (0.5 * val) ** 2
                        except ValueError:
                            pass
            else:
                code_part = raw_line

            if "(" in code_part:
                code_part = _PAREN_RE.sub("", code_part)
            code_part = code_part.strip()
            if not code_part:
                continue
            words = _WORD_RE.findall(code_part.upper())
            if not words:
                continue
            commands = [(k, int(float(v))) for k, v in words if k in ("G", "M")]
            vals = {k: float(v) for k, v in words if k in pos}
            for kind, cmd in commands:
                if kind == "M":
                    if cmd == 82:
                        e_absolute = True
                    elif cmd == 83:
                        e_absolute = False
                    continue
                if cmd == 20:
                    raise ValueError("仅支持毫米 G-code (G21)，不支持英寸 G20")
                if cmd == 90:
                    absolute = True
                    e_absolute = True
                elif cmd == 91:
                    absolute = False
                    e_absolute = False
                elif cmd == 92:
                    pos.update(vals)
                    in_bead = False
                elif cmd in (2, 3):
                    if not (skip_features and feature_skipped):
                        raise ValueError("检测到圆弧 G2/G3，请在切片器中禁用圆弧拟合后重新导出")
                elif cmd in (0, 1):
                    ox, oy, oz, oe = pos["X"], pos["Y"], pos["Z"], pos["E"]
                    for axis, value in vals.items():
                        use_abs = e_absolute if axis == "E" else absolute
                        pos[axis] = value if use_abs else pos[axis] + value
                    dx_m = pos["X"] - ox
                    dy_m = pos["Y"] - oy
                    dL = math.hypot(dx_m, dy_m)
                    dE = pos["E"] - oe
                    if dE <= 1e-8 or dL <= 1e-8:
                        if dL > 1e-6 or dE < -1e-6:
                            in_bead = False
                        continue
                    if skip_features and feature_skipped:
                        in_bead = False
                        continue
                    if abs(pos["Z"] - oz) > 1e-5:
                        raise ValueError("检测到非平面挤出路径；此版本仅支持平面层")
                    z_world = pos["Z"] + off[2]
                    if current_layer_z is None:
                        current_layer_z = z_world
                        layer_h_used = current_layer_h
                    elif abs(z_world - current_layer_z) > 1e-4:
                        if layer_segs:
                            yielded_any = True
                            yield current_layer_z, layer_h_used, np.asarray(layer_segs, dtype=np.float64)
                            layer_segs = []
                        current_layer_z = z_world
                        layer_h_used = current_layer_h
                        in_bead = False
                    if explicit_width is not None and explicit_width > 0:
                        w = explicit_width
                    else:
                        w_est = (dE * filament_area) / (dL * max(current_layer_h, 1e-6))
                        w = w_est if 0.15 <= w_est <= 1.5 else float(default_width_mm)
                    if not in_bead:
                        bead_id += 1
                        in_bead = True
                    layer_segs.append(
                        [
                            ox + off[0],
                            oy + off[1],
                            pos["X"] + off[0],
                            pos["Y"] + off[1],
                            w,
                            float(feature_code),
                            float(bead_id),
                        ]
                    )
    if layer_segs:
        yielded_any = True
        yield current_layer_z, layer_h_used, np.asarray(layer_segs, dtype=np.float64)
    if not yielded_any:
        raise ValueError("未找到有效的 G0/G1 挤出线段")


def _rasterize_layer_cpu(segs_local, nx, ny, dx, dy, submask, cos2_sum, sin2_sum, cov_sum, feat_arr):
    """Pure-NumPy vectorized fallback if Warp is not used."""
    sub_offsets = (np.arange(4, dtype=np.float64) + 0.5) * 0.25
    gx, gy = np.meshgrid(sub_offsets * dx, sub_offsets * dy)
    sub_dx = gx.ravel()[None, :]  # (1, 16)
    sub_dy = gy.ravel()[None, :]  # (1, 16)
    bits = (np.uint32(1) << np.arange(16, dtype=np.uint32))[None, :]
    for x0, y0, x1, y1, r, fprio in segs_local:
        vx = x1 - x0
        vy = y1 - y0
        len2 = vx * vx + vy * vy
        if len2 <= 1e-16:
            continue
        theta = math.atan2(vy, vx)
        cos2 = math.cos(2.0 * theta)
        sin2 = math.sin(2.0 * theta)
        r2 = r * r
        ix0 = max(0, int(math.floor((min(x0, x1) - r) / dx)))
        ix1 = min(nx - 1, int(math.floor((max(x0, x1) + r) / dx)))
        iy0 = max(0, int(math.floor((min(y0, y1) - r) / dy)))
        iy1 = min(ny - 1, int(math.floor((max(y0, y1) + r) / dy)))
        if ix1 < ix0 or iy1 < iy0:
            continue
        xs = np.arange(ix0, ix1 + 1, dtype=np.float64) * dx
        ys = np.arange(iy0, iy1 + 1, dtype=np.float64) * dy
        cx, cy = np.meshgrid(xs, ys)
        cx = cx.ravel()[:, None] + sub_dx
        cy = cy.ravel()[:, None] + sub_dy
        dxp = cx - x0
        dyp = cy - y0
        t = np.clip((dxp * vx + dyp * vy) / len2, 0.0, 1.0)
        qx = t * vx - dxp
        qy = t * vy - dyp
        inside = (qx * qx + qy * qy) <= r2
        m16 = (inside.astype(np.uint32) * bits).sum(axis=1)
        hit = m16 != 0
        if not hit.any():
            continue
        ix_flat, iy_flat = np.meshgrid(np.arange(ix0, ix1 + 1), np.arange(iy0, iy1 + 1))
        idx = (iy_flat.ravel()[hit] * nx + ix_flat.ravel()[hit]).astype(np.int64)
        mh = m16[hit]
        pop = (_POPCOUNT_LUT[mh & 0xFF] + _POPCOUNT_LUT[(mh >> 8) & 0xFF]).astype(np.int32)
        submask[idx] |= mh
        cos2_sum[idx] += pop.astype(np.float32) * np.float32(cos2)
        sin2_sum[idx] += pop.astype(np.float32) * np.float32(sin2)
        cov_sum[idx] += pop
        feat_arr[idx] = np.maximum(feat_arr[idx], int(fprio))


def _pack_slab_numpy(nx, ny, nbx, n_bins, submask, cos2_sum, sin2_sum, cov_sum, feat_arr, multi_arr):
    """NumPy fallback to pack 4 Z-slices into non-empty 4x4x4 bricks."""
    nby = ny // 4
    sm4 = submask.reshape(4, nby, 4, nbx, 4).transpose(1, 3, 0, 2, 4).reshape(nby * nbx, 64)
    active_b = np.flatnonzero((sm4 != 0).any(axis=1))
    if len(active_b) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.uint64),
            np.empty((0, 64), dtype=np.uint32),
        )
    sm_a = sm4[active_b] & np.uint32(0xFFFF)
    c2_a = cos2_sum.reshape(4, nby, 4, nbx, 4).transpose(1, 3, 0, 2, 4).reshape(nby * nbx, 64)[active_b]
    s2_a = sin2_sum.reshape(4, nby, 4, nbx, 4).transpose(1, 3, 0, 2, 4).reshape(nby * nbx, 64)[active_b]
    cov_a = cov_sum.reshape(4, nby, 4, nbx, 4).transpose(1, 3, 0, 2, 4).reshape(nby * nbx, 64)[active_b]
    ft_a = feat_arr.reshape(4, nby, 4, nbx, 4).transpose(1, 3, 0, 2, 4).reshape(nby * nbx, 64)[active_b]
    mu_a = multi_arr.reshape(4, nby, 4, nbx, 4).transpose(1, 3, 0, 2, 4).reshape(nby * nbx, 64)[active_b]

    occ = sm_a != 0
    bit_weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    masks = (occ.astype(np.uint64) * bit_weights[None, :]).sum(axis=1, dtype=np.uint64)
    pop = _POPCOUNT_LUT[sm_a & np.uint32(0xFF)] + _POPCOUNT_LUT[(sm_a >> np.uint32(8)) & np.uint32(0xFF)]
    rho_q = np.clip((pop * 255 + 8) // 16, 0, 255)
    theta = 0.5 * np.arctan2(s2_a, c2_a)
    abin = angle_to_bin(theta, n_bins)
    fcode = ft_a & 0x07
    flags = (fcode << 1) | (cov_a > pop).astype(np.int32) | ((mu_a > 1).astype(np.int32) << 4)
    payload = np.where(occ, pack_payload(abin, rho_q, flags), np.uint32(0)).astype(np.uint32)
    by = (active_b // nbx).astype(np.int32)
    bx = (active_b % nbx).astype(np.int32)
    coords = np.column_stack([bx, by, np.zeros_like(bx)])
    return coords, masks, payload


def _local_brick_components(masks):
    """Exact 6-connected components inside each 4x4x4 uint64 brick mask.

    Returns (super_counts[B], cell_local_id[B, 64], boundary_faces_masks).
    Fast path handles single-component bricks (~98%) with 9 vectorized bitwise flood steps.
    """
    n_bricks = len(masks)
    cell_local = np.full((n_bricks, 64), -1, dtype=np.int8)
    super_counts = np.zeros(n_bricks, dtype=np.int32)
    if n_bricks == 0:
        return super_counts, cell_local

    remaining = masks.copy()
    comp_idx = 0
    active = np.flatnonzero(remaining != 0)
    while len(active):
        rem_a = remaining[active]
        # Lowest set bit in uint64: m & (~m + 1)
        seed = rem_a & (~rem_a + np.uint64(1))
        flood = seed.copy()
        for _ in range(9):
            nxt = (
                flood
                | ((flood & _NOT_X3) << np.uint64(1))
                | ((flood & _NOT_X0) >> np.uint64(1))
                | ((flood & _NOT_Y3) << np.uint64(4))
                | ((flood & _NOT_Y0) >> np.uint64(4))
                | ((flood & _NOT_Z3) << np.uint64(16))
                | ((flood & _NOT_Z0) >> np.uint64(16))
            ) & rem_a
            if np.array_equal(nxt, flood):
                break
            flood = nxt
        for c in range(64):
            hit = ((flood >> np.uint64(c)) & np.uint64(1)) != 0
            if hit.any():
                cell_local[active[hit], c] = comp_idx
        super_counts[active] += 1
        remaining[active] &= ~flood
        active = active[remaining[active] != 0]
        comp_idx += 1
    return super_counts, cell_local


def filter_connected_bricks(brick_coords, brick_masks, brick_payload, brick_shape, keep_largest_component=True):
    """Exact 6-neighbor face connectivity across sparse 4x4x4 bricks via supernode contraction."""
    n_bricks = len(brick_coords)
    if n_bricks == 0:
        raise ValueError("体素砖块为空")
    nbx, nby, nbz = (int(x) for x in brick_shape)
    super_counts, cell_local = _local_brick_components(brick_masks)
    super_offsets = np.zeros(n_bricks + 1, dtype=np.int64)
    np.cumsum(super_counts, out=super_offsets[1:])
    n_super = int(super_offsets[-1])

    # Sort brick keys for O(B log B) neighbor brick lookup.
    bkeys = (
        brick_coords[:, 2].astype(np.int64) * (nby * nbx)
        + brick_coords[:, 1].astype(np.int64) * nbx
        + brick_coords[:, 0].astype(np.int64)
    )
    order = np.argsort(bkeys)
    sorted_keys = bkeys[order]

    def _lookup(target_keys, valid_mask):
        idx = np.full(len(target_keys), -1, dtype=np.int32)
        if not valid_mask.any():
            return idx
        tk = target_keys[valid_mask]
        pos = np.searchsorted(sorted_keys, tk)
        in_bounds = pos < len(sorted_keys)
        pos_safe = np.where(in_bounds, pos, 0)
        matched = in_bounds & (sorted_keys[pos_safe] == tk)
        sub = np.flatnonzero(valid_mask)[matched]
        idx[sub] = order[pos_safe[matched]]
        return idx

    # Face cell indices for +X/-X, +Y/-Y, +Z/-Z boundaries
    fx_hi = np.array([3 + 4 * ly + 16 * lz for lz in range(4) for ly in range(4)], dtype=np.int32)
    fx_lo = fx_hi - 3
    fy_hi = np.array([lx + 4 * 3 + 16 * lz for lz in range(4) for lx in range(4)], dtype=np.int32)
    fy_lo = fy_hi - 12
    fz_hi = np.array([lx + 4 * ly + 16 * 3 for ly in range(4) for lx in range(4)], dtype=np.int32)
    fz_lo = fz_hi - 48

    edges_u = []
    edges_v = []
    for axis, stride, f_hi, f_lo in (
        (0, 1, fx_hi, fx_lo),
        (1, nbx, fy_hi, fy_lo),
        (2, nbx * nby, fz_hi, fz_lo),
    ):
        limit = (nbx, nby, nbz)[axis] - 1
        valid = brick_coords[:, axis] < limit
        nbr = _lookup(bkeys + stride, valid)
        has_nbr = np.flatnonzero(nbr >= 0)
        if not len(has_nbr):
            continue
        nb = nbr[has_nbr]
        la = cell_local[has_nbr][:, f_hi]
        lb = cell_local[nb][:, f_lo]
        both = (la >= 0) & (lb >= 0)
        if not both.any():
            continue
        b_idx, f_idx = np.nonzero(both)
        ua = super_offsets[has_nbr[b_idx]] + la[b_idx, f_idx]
        vb = super_offsets[nb[b_idx]] + lb[b_idx, f_idx]
        edges_u.append(ua)
        edges_v.append(vb)

    if edges_u:
        eu = np.concatenate(edges_u)
        ev = np.concatenate(edges_v)
        graph = coo_matrix(
            (np.ones(len(eu) * 2, dtype=np.int8), (np.r_[eu, ev], np.r_[ev, eu])),
            shape=(n_super, n_super),
        ).tocsr()
        n_comp, labels = connected_components(graph, directed=False, return_labels=True)
    else:
        n_comp = n_super
        labels = np.arange(n_super, dtype=np.int32)

    if n_comp == 1:
        return (
            brick_coords,
            brick_masks,
            brick_payload,
            dict(components=1, kept_component=0, dropped_components=0, dropped_voxels=0, dropped_volume_fraction=0.0),
        )

    # Compute exact rasterized volume per connected component.
    _, rho_q_all, _ = unpack_payload(brick_payload)
    comp_vox = np.zeros(n_comp, dtype=np.int64)
    comp_rho = np.zeros(n_comp, dtype=np.float64)
    for c in range(64):
        loc = cell_local[:, c]
        act = loc >= 0
        if not act.any():
            continue
        b_act = np.flatnonzero(act)
        s_ids = super_offsets[b_act] + loc[act]
        c_ids = labels[s_ids]
        np.add.at(comp_vox, c_ids, 1)
        np.add.at(comp_rho, c_ids, rho_q_all[b_act, c] / 255.0)

    total_vox = int(comp_vox.sum())
    total_rho = float(comp_rho.sum())
    if not keep_largest_component:
        raise ValueError(f"检测到 {n_comp} 个不连通体素分量（仅边角接触也不允许）；请清理路径或启用 keep_largest_component")

    best_comp = int(np.argmax(comp_vox))
    dropped_voxels = total_vox - int(comp_vox[best_comp])
    dropped_vol_frac = float((total_rho - comp_rho[best_comp]) / max(total_rho, 1e-20))

    new_masks = np.zeros_like(brick_masks)
    new_payload = np.zeros_like(brick_payload)
    for c in range(64):
        loc = cell_local[:, c]
        act = loc >= 0
        if not act.any():
            continue
        b_act = np.flatnonzero(act)
        s_ids = super_offsets[b_act] + loc[act]
        keep_b = b_act[labels[s_ids] == best_comp]
        if len(keep_b):
            new_masks[keep_b] |= np.uint64(1) << np.uint64(c)
            new_payload[keep_b, c] = brick_payload[keep_b, c]

    non_empty = new_masks != 0
    return (
        np.ascontiguousarray(brick_coords[non_empty], dtype=np.int32),
        np.ascontiguousarray(new_masks[non_empty], dtype=np.uint64),
        np.ascontiguousarray(new_payload[non_empty], dtype=np.uint32),
        dict(
            components=int(n_comp),
            kept_component=best_comp,
            dropped_components=int(n_comp - 1),
            dropped_voxels=int(dropped_voxels),
            dropped_volume_fraction=dropped_vol_frac,
        ),
    )


def gcode_to_bricks(
    gcode_path,
    out_bricks_path=None,
    pitch=0.2,
    dz=None,
    offset=(0.0, 0.0, 0.0),
    default_width_mm=0.42,
    default_layer_height_mm=0.2,
    filament_diameter_mm=1.75,
    skip_features=True,
    keep_largest_component=True,
    n_bins=360,
    device="auto",
):
    """Stream G-code in two passes to 4x4x4 sparse bricks with slab-level disk spill."""
    t0 = time.perf_counter()
    gcode_path = Path(gcode_path)
    if not gcode_path.is_file():
        raise FileNotFoundError(f"G-code not found: {gcode_path}")
    dx = dy = float(pitch)
    dz = float(pitch if dz is None else dz)
    if not np.isfinite([dx, dy, dz]).all() or min(dx, dy, dz) <= 0:
        raise ValueError("pitch and dz must be positive finite numbers")

    src_hash = hashlib.sha256()
    with gcode_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            src_hash.update(chunk)

    # Pass 1: stream layers to determine exact bounding box, layer schedule, and Z alignment.
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    max_half_w = 0.0
    layer_records = []
    prev_z = None
    total_segs = 0
    for lz, lh, segs in iter_gcode_layers(
        gcode_path,
        offset=offset,
        default_width_mm=default_width_mm,
        default_layer_height_mm=default_layer_height_mm,
        filament_diameter_mm=filament_diameter_mm,
        skip_features=skip_features,
    ):
        if prev_z is not None and lz < prev_z - 1e-4:
            raise ValueError("检测到非单调 Z 层序（例如逐个打印模式）；请使用按层切片 G-code")
        prev_z = lz
        w = segs[:, 4]
        hw = 0.5 * float(w.max())
        max_half_w = max(max_half_w, hw)
        xmin = min(xmin, float(np.minimum(segs[:, 0], segs[:, 2]).min()) - hw)
        xmax = max(xmax, float(np.maximum(segs[:, 0], segs[:, 2]).max()) + hw)
        ymin = min(ymin, float(np.minimum(segs[:, 1], segs[:, 3]).min()) - hw)
        ymax = max(ymax, float(np.maximum(segs[:, 1], segs[:, 3]).max()) + hw)
        layer_records.append((lz, lh, len(segs)))
        total_segs += len(segs)

    first_z, first_h, _ = layer_records[0]
    origin_z = float(first_z - first_h)
    max_z = max(z for z, _, _ in layer_records)
    pad_xy = max(max_half_w * 0.25, dx)
    origin_x = math.floor((xmin - pad_xy) / dx) * dx
    origin_y = math.floor((ymin - pad_xy) / dy) * dy
    nx_raw = max(1, int(math.ceil((xmax + pad_xy - origin_x) / dx)))
    ny_raw = max(1, int(math.ceil((ymax + pad_xy - origin_y) / dy)))
    nz_raw = max(1, int(math.ceil((max_z - origin_z - 1e-7) / dz)))
    nx = ((nx_raw + 3) // 4) * 4
    ny = ((ny_raw + 3) // 4) * 4
    nz = ((nz_raw + 3) // 4) * 4
    nbx, nby, nbz = nx // 4, ny // 4, nz // 4

    used_device = _ensure_warp(device) if wp is not None else "cpu"
    use_warp = wp is not None

    # Scratch spill directory for finished 4-slice brick slabs (never holds full 3D grid in RAM).
    if out_bricks_path is not None:
        out_p = Path(out_bricks_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        scratch_dir = out_p.parent / f"._{out_p.stem}_{uuid.uuid4().hex}_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        cleanup_scratch = lambda: shutil.rmtree(scratch_dir, ignore_errors=True)
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="tpbricks_")
        scratch_dir = Path(tmp_obj.name)
        cleanup_scratch = tmp_obj.cleanup

    slab_files = []
    try:
        if use_warp:
            submask_dev = wp.zeros((4, ny * nx), dtype=wp.uint32, device=used_device)
            cos2_dev = wp.zeros((4, ny * nx), dtype=wp.float32, device=used_device)
            sin2_dev = wp.zeros((4, ny * nx), dtype=wp.float32, device=used_device)
            cov_dev = wp.zeros((4, ny * nx), dtype=wp.int32, device=used_device)
            feat_dev = wp.zeros((4, ny * nx), dtype=wp.int32, device=used_device)
            multi_dev = wp.zeros((4, ny * nx), dtype=wp.int32, device=used_device)
            slab_masks_dev = wp.zeros(nby * nbx, dtype=wp.uint64, device=used_device)
            slab_payload_dev = wp.zeros((nby * nbx, 64), dtype=wp.uint32, device=used_device)
        else:
            submask_np = np.zeros((4, ny * nx), dtype=np.uint32)
            cos2_np = np.zeros((4, ny * nx), dtype=np.float32)
            sin2_np = np.zeros((4, ny * nx), dtype=np.float32)
            cov_np = np.zeros((4, ny * nx), dtype=np.int32)
            feat_np = np.zeros((4, ny * nx), dtype=np.int32)
            multi_np = np.zeros((4, ny * nx), dtype=np.int32)

        slab_start = 0
        slab_dirty = False

        @wp.kernel
        def _mark_slice_hit(submask: wp.array(dtype=wp.uint32, ndim=2), sz: wp.int32, multi_arr: wp.array(dtype=wp.int32, ndim=2)):
            i = wp.tid()
            if submask[sz, i] != wp.uint32(0):
                multi_arr[sz, i] = multi_arr[sz, i] + 1

        def _flush_slab(bz):
            nonlocal slab_dirty
            if not slab_dirty:
                return
            if use_warp:
                wp.launch(
                    _pack_brick_slab_kernel,
                    dim=nby * nbx,
                    inputs=[
                        nx,
                        ny,
                        nbx,
                        n_bins,
                        submask_dev,
                        cos2_dev,
                        sin2_dev,
                        cov_dev,
                        feat_dev,
                        multi_dev,
                        slab_masks_dev,
                        slab_payload_dev,
                    ],
                    device=used_device,
                )
                masks_host = slab_masks_dev.numpy()
                active_b = np.flatnonzero(masks_host != 0)
                if len(active_b):
                    payload_host = slab_payload_dev.numpy()[active_b]
                    by = (active_b // nbx).astype(np.int32)
                    bx = (active_b % nbx).astype(np.int32)
                    coords = np.column_stack([bx, by, np.full(len(active_b), bz, dtype=np.int32)])
                    masks = np.ascontiguousarray(masks_host[active_b], dtype=np.uint64)
                    payload = np.ascontiguousarray(payload_host, dtype=np.uint32)
                else:
                    coords = np.empty((0, 3), dtype=np.int32)
                    masks = np.empty((0,), dtype=np.uint64)
                    payload = np.empty((0, 64), dtype=np.uint32)
                for buf in (submask_dev, cos2_dev, sin2_dev, cov_dev, feat_dev, multi_dev):
                    buf.zero_()
            else:
                coords, masks, payload = _pack_slab_numpy(
                    nx, ny, nbx, n_bins, submask_np, cos2_np, sin2_np, cov_np, feat_np, multi_np
                )
                coords[:, 2] = bz
                for buf in (submask_np, cos2_np, sin2_np, cov_np, feat_np, multi_np):
                    buf.fill(0)

            if len(coords):
                sf = scratch_dir / f"slab_{bz:05d}.bin"
                with sf.open("wb") as fh:
                    fh.write(struct.pack("<Q", len(coords)))
                    fh.write(coords.tobytes())
                    fh.write(masks.tobytes())
                    fh.write(payload.tobytes())
                slab_files.append(sf)
            slab_dirty = False

        # Pass 2: stream layers, rasterize on GPU/CPU, flush 4-slice slabs to disk.
        prio_lut = np.array([FEATURE_PRIORITY.get(c, 0) for c in range(8)], dtype=np.float32)
        for lz, lh, segs in iter_gcode_layers(
            gcode_path,
            offset=offset,
            default_width_mm=default_width_mm,
            default_layer_height_mm=default_layer_height_mm,
            filament_diameter_mm=filament_diameter_mm,
            skip_features=skip_features,
        ):
            z_lo = lz - lh
            z_hi = lz
            iz_lo = max(0, int(math.floor((z_lo + 1e-6 - origin_z) / dz)))
            iz_hi = min(nz_raw - 1, int(math.floor((z_hi - 1e-6 - origin_z) / dz)))
            if iz_hi < iz_lo:
                z_mid = 0.5 * (z_lo + z_hi)
                iz_single = min(nz_raw - 1, max(0, int(math.floor((z_mid - origin_z) / dz))))
                iz_list = [iz_single]
            else:
                iz_list = list(range(iz_lo, iz_hi + 1))

            # Build local-origin segment array [x0, y0, x1, y1, radius, packed_feat_prio]
            seg_local = np.empty((len(segs), 6), dtype=np.float32)
            seg_local[:, 0] = (segs[:, 0] - origin_x).astype(np.float32)
            seg_local[:, 1] = (segs[:, 1] - origin_y).astype(np.float32)
            seg_local[:, 2] = (segs[:, 2] - origin_x).astype(np.float32)
            seg_local[:, 3] = (segs[:, 3] - origin_y).astype(np.float32)
            seg_local[:, 4] = (0.5 * segs[:, 4]).astype(np.float32)
            fc = segs[:, 5].astype(np.int32)
            seg_local[:, 5] = (prio_lut[np.clip(fc, 0, 7)].astype(np.int32) << 4) | (fc & 0x07)

            if use_warp:
                seg_dev = wp.array(seg_local, dtype=wp.float32, device=used_device)

            for iz in iz_list:
                while iz >= slab_start + 4:
                    _flush_slab(slab_start // 4)
                    slab_start += 4
                sz = iz - slab_start
                slab_dirty = True
                if use_warp:
                    wp.launch(
                        _rasterize_segments_2d_kernel,
                        dim=len(seg_local),
                        inputs=[
                            seg_dev,
                            nx,
                            ny,
                            np.float32(dx),
                            np.float32(dy),
                            submask_dev[sz],
                            cos2_dev[sz],
                            sin2_dev[sz],
                            cov_dev[sz],
                            feat_dev[sz],
                        ],
                        device=used_device,
                    )
                    wp.launch(_mark_slice_hit, dim=ny * nx, inputs=[submask_dev, sz, multi_dev], device=used_device)
                else:
                    _rasterize_layer_cpu(
                        seg_local,
                        nx,
                        ny,
                        dx,
                        dy,
                        submask_np[sz],
                        cos2_np[sz],
                        sin2_np[sz],
                        cov_np[sz],
                        feat_np[sz],
                    )
                    multi_np[sz][submask_np[sz] != 0] += 1

        _flush_slab(slab_start // 4)

        # Free device scratch buffers immediately before connectivity assembly.
        if use_warp:
            del submask_dev, cos2_dev, sin2_dev, cov_dev, feat_dev, multi_dev, slab_masks_dev, slab_payload_dev

        # Stream-read spilled slab files into compact brick arrays.
        coords_list, masks_list, payload_list = [], [], []
        for sf in slab_files:
            with sf.open("rb") as fh:
                (count,) = struct.unpack("<Q", fh.read(8))
                c_arr = np.frombuffer(fh.read(count * 12), dtype=np.int32).reshape(count, 3).copy()
                m_arr = np.frombuffer(fh.read(count * 8), dtype=np.uint64).copy()
                p_arr = np.frombuffer(fh.read(count * 256), dtype=np.uint32).reshape(count, 64).copy()
            coords_list.append(c_arr)
            masks_list.append(m_arr)
            payload_list.append(p_arr)

        if not coords_list:
            raise ValueError("光栅化后未生成任何有效实体体素")
        brick_coords = np.concatenate(coords_list, axis=0)
        brick_masks = np.concatenate(masks_list, axis=0)
        brick_payload = np.concatenate(payload_list, axis=0)
        del coords_list, masks_list, payload_list
    finally:
        cleanup_scratch()

    brick_coords, brick_masks, brick_payload, conn_info = filter_connected_bricks(
        brick_coords,
        brick_masks,
        brick_payload,
        (nbx, nby, nbz),
        keep_largest_component=keep_largest_component,
    )

    # Audit final active voxels, volume, features, and overlaps.
    active_bit = ((brick_masks[:, None] >> np.arange(64, dtype=np.uint64)[None, :]) & np.uint64(1)) != 0
    _, rho_q, flags = unpack_payload(brick_payload)
    rho_active = rho_q[active_bit].astype(np.float64) / 255.0
    flags_active = flags[active_bit]
    feats_active = decode_feature(flags_active)
    voxel_vol = dx * dy * dz
    rasterized_volume = float(rho_active.sum() * voxel_vol)
    full_voxel_volume = float(len(rho_active) * voxel_vol)
    feature_counts = {
        FEATURE_NAMES[k]: int((feats_active == k).sum()) for k in sorted(FEATURE_NAMES) if (feats_active == k).any()
    }
    overlap_fraction = float(np.mean((flags_active & FLAG_BEAD_OVERLAP) != 0))
    elapsed = time.perf_counter() - t0

    meta = {
        "source_gcode": str(gcode_path.resolve()),
        "source_sha256": src_hash.hexdigest(),
        "layer_count": len(layer_records),
        "segment_count": int(total_segs),
        "layer_z_min_mm": float(min(z for z, _, _ in layer_records)),
        "layer_z_max_mm": float(max_z),
        "median_layer_height_mm": float(np.median([h for _, h, _ in layer_records])),
        "pitch_mm": [dx, dy, dz],
        "origin_mm": [origin_x, origin_y, origin_z],
        "grid_shape": [nx, ny, nz],
        "brick_shape": [nbx, nby, nbz],
        "num_bricks": int(len(brick_coords)),
        "solid_voxel_count": int(len(rho_active)),
        "rasterized_volume_mm3": rasterized_volume,
        "full_voxel_volume_mm3": full_voxel_volume,
        "mean_rho": float(rho_active.mean()),
        "overlap_voxel_fraction": overlap_fraction,
        "feature_voxel_counts": feature_counts,
        "connectivity": conn_info,
        "rasterize_device": used_device if use_warp else "cpu_numpy",
        "rasterize_seconds": elapsed,
        "n_angle_bins": int(n_bins),
        "subsamples": 4,
    }

    grid = BrickGrid(
        origin=np.array([origin_x, origin_y, origin_z], dtype=np.float64),
        pitch=np.array([dx, dy, dz], dtype=np.float64),
        shape=(nx, ny, nz),
        brick_coords=brick_coords,
        brick_masks=brick_masks,
        brick_payload=brick_payload,
        meta=meta,
    )
    if out_bricks_path is not None:
        save_bricks(out_bricks_path, grid)
    return grid


def mask_to_bricks(occupied_ijk, pitch, origin=(0.0, 0.0, 0.0), angles_rad=None, rho=None, feature_codes=None, flags=None, n_bins=360):
    """Build a BrickGrid directly from active voxel integer coordinates (V, 3)."""
    ijk = np.asarray(occupied_ijk, dtype=np.int32)
    if ijk.ndim != 2 or ijk.shape[1] != 3 or len(ijk) == 0:
        raise ValueError("occupied_ijk must be a non-empty (V, 3) integer array")
    if np.any(ijk < 0):
        raise ValueError("occupied_ijk indices must be non-negative")
    p = np.broadcast_to(np.asarray(pitch, dtype=np.float64), (3,)).copy()
    o = np.asarray(origin, dtype=np.float64)
    if p.shape != (3,) or np.any(p <= 0) or not np.isfinite(p).all() or o.shape != (3,) or not np.isfinite(o).all():
        raise ValueError("invalid pitch or origin")

    n_vox = len(ijk)
    ang = np.zeros(n_vox, dtype=np.float64) if angles_rad is None else np.broadcast_to(np.asarray(angles_rad, dtype=np.float64), (n_vox,))
    rho_arr = np.ones(n_vox, dtype=np.float64) if rho is None else np.broadcast_to(np.asarray(rho, dtype=np.float64), (n_vox,))
    fcode = np.full(n_vox, FEATURE_SOLID_INFILL, dtype=np.int32) if feature_codes is None else np.broadcast_to(np.asarray(feature_codes, dtype=np.int32), (n_vox,))
    fl_extra = np.zeros(n_vox, dtype=np.int32) if flags is None else np.broadcast_to(np.asarray(flags, dtype=np.int32), (n_vox,))

    nx = int(((ijk[:, 0].max() + 1 + 3) // 4) * 4)
    ny = int(((ijk[:, 1].max() + 1 + 3) // 4) * 4)
    nz = int(((ijk[:, 2].max() + 1 + 3) // 4) * 4)
    nbx, nby = nx // 4, ny // 4

    bijk = ijk // 4
    lijk = ijk % 4
    bkey = bijk[:, 2].astype(np.int64) * (nbx * nby) + bijk[:, 1].astype(np.int64) * nbx + bijk[:, 0].astype(np.int64)
    uniq_keys, inv = np.unique(bkey, return_inverse=True)
    n_bricks = len(uniq_keys)
    bz = (uniq_keys // (nbx * nby)).astype(np.int32)
    rem = uniq_keys % (nbx * nby)
    by = (rem // nbx).astype(np.int32)
    bx = (rem % nbx).astype(np.int32)
    brick_coords = np.column_stack([bx, by, bz])
    brick_masks = np.zeros(n_bricks, dtype=np.uint64)
    brick_payload = np.zeros((n_bricks, 64), dtype=np.uint32)

    local_c = lijk[:, 0] + 4 * lijk[:, 1] + 16 * lijk[:, 2]
    abin = angle_to_bin(ang, n_bins)
    rho_q = np.clip(np.rint(rho_arr * 255.0), 1, 255).astype(np.int32)
    packed_flags = (fl_extra & ~0x0E) | ((fcode & 0x07) << 1)
    payload_vals = pack_payload(abin, rho_q, packed_flags)

    np.bitwise_or.at(brick_masks, inv, np.uint64(1) << local_c.astype(np.uint64))
    brick_payload[inv, local_c] = payload_vals

    meta = {
        "pitch_mm": p.tolist(),
        "origin_mm": o.tolist(),
        "grid_shape": [nx, ny, nz],
        "brick_shape": [nx // 4, ny // 4, nz // 4],
        "num_bricks": n_bricks,
        "solid_voxel_count": n_vox,
        "rasterized_volume_mm3": float((rho_q / 255.0).sum() * np.prod(p)),
        "full_voxel_volume_mm3": float(n_vox * np.prod(p)),
        "n_angle_bins": int(n_bins),
    }
    return BrickGrid(o, p, (nx, ny, nz), brick_coords, brick_masks, brick_payload, meta)


def box_to_bricks(size=(40.0, 12.0, 8.0), pitch=0.2, origin=(0.0, 0.0, 0.0), angle_deg=0.0, rho=1.0, feature_code=FEATURE_SOLID_INFILL):
    """Construct a dense axis-aligned BrickGrid fitting `size` at `pitch`."""
    size_arr = np.asarray(size, dtype=np.float64)
    if size_arr.shape != (3,) or np.any(size_arr <= 0) or not np.isfinite(size_arr).all():
        raise ValueError("size must be 3 positive finite numbers")
    p_arr = np.broadcast_to(np.asarray(pitch, dtype=np.float64), (3,))
    counts = np.maximum(1, np.rint(size_arr / p_arr).astype(np.int32))
    cell_pitch = size_arr / counts
    ijk = np.indices(tuple(counts)).reshape(3, -1).T.astype(np.int32)
    return mask_to_bricks(
        ijk,
        cell_pitch,
        origin=origin,
        angles_rad=math.radians(angle_deg),
        rho=rho,
        feature_codes=feature_code,
    )


def _payload_sha256(brick_coords, brick_masks, brick_payload):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(brick_coords, dtype="<i4").tobytes())
    h.update(np.ascontiguousarray(brick_masks, dtype="<u8").tobytes())
    h.update(np.ascontiguousarray(brick_payload, dtype="<u4").tobytes())
    return h.hexdigest()


def save_bricks(path, grid):
    """Atomically write BrickGrid to binary .tbricks format with payload SHA-256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = np.ascontiguousarray(grid.brick_coords, dtype="<i4")
    masks = np.ascontiguousarray(grid.brick_masks, dtype="<u8")
    payload = np.ascontiguousarray(grid.brick_payload, dtype="<u4")
    header = {
        "format": MAGIC_HEADER.decode("ascii"),
        "brick_size": BRICK_SIZE,
        "num_bricks": int(len(coords)),
        "num_active_voxels": int(popcount_uint64(masks).sum()),
        "origin": [float(x) for x in grid.origin],
        "pitch": [float(x) for x in grid.pitch],
        "shape": [int(x) for x in grid.shape],
        "payload_sha256": _payload_sha256(coords, masks, payload),
        "meta": grid.meta,
    }
    header_bytes = json.dumps(header, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(MAGIC_HEADER)
            stream.write(struct.pack("<Q", len(header_bytes)))
            stream.write(header_bytes)
            for arr in (coords, masks, payload):
                mv = memoryview(arr).cast("B")
                step = 16 * 1024 * 1024
                for off in range(0, len(mv), step):
                    stream.write(mv[off : off + step])
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return path


def _read_exact_chunks(stream, total_bytes, dtype, shape):
    out = np.empty(shape, dtype=dtype)
    mv = memoryview(out).cast("B")
    read_pos = 0
    chunk = 16 * 1024 * 1024
    while read_pos < total_bytes:
        n = min(chunk, total_bytes - read_pos)
        got = stream.readinto(mv[read_pos : read_pos + n])
        if not got:
            raise ValueError("Unexpected EOF while reading .tbricks payload")
        read_pos += got
    return out


def load_bricks(path, verify_checksum=True):
    """Load BrickGrid from .tbricks using explicit chunked reads (never np.memmap)."""
    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.read(len(MAGIC_HEADER))
        if magic != MAGIC_HEADER:
            raise ValueError("Invalid .tbricks magic header")
        (json_len,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(json_len).decode("utf-8"))
        n_bricks = int(header["num_bricks"])
        coords = _read_exact_chunks(stream, n_bricks * 12, "<i4", (n_bricks, 3)).astype(np.int32, copy=False)
        masks = _read_exact_chunks(stream, n_bricks * 8, "<u8", (n_bricks,)).astype(np.uint64, copy=False)
        payload = _read_exact_chunks(stream, n_bricks * 256, "<u4", (n_bricks, 64)).astype(np.uint32, copy=False)
    if verify_checksum and _payload_sha256(coords, masks, payload) != header["payload_sha256"]:
        raise ValueError(".tbricks payload checksum mismatch")
    return BrickGrid(
        origin=np.asarray(header["origin"], dtype=np.float64),
        pitch=np.asarray(header["pitch"], dtype=np.float64),
        shape=tuple(int(x) for x in header["shape"]),
        brick_coords=coords,
        brick_masks=masks,
        brick_payload=payload,
        meta=header.get("meta", {}),
    )


def iter_brick_slabs(path, chunk_bricks=8192):
    """Stream .tbricks file in bounded brick chunks without loading full arrays."""
    path = Path(path)
    with path.open("rb") as stream:
        magic = stream.read(len(MAGIC_HEADER))
        if magic != MAGIC_HEADER:
            raise ValueError("Invalid .tbricks magic header")
        (json_len,) = struct.unpack("<Q", stream.read(8))
        header = json.loads(stream.read(json_len).decode("utf-8"))
        n_bricks = int(header["num_bricks"])
        base_off = len(MAGIC_HEADER) + 8 + json_len
        coords_off = base_off
        masks_off = coords_off + n_bricks * 12
        payload_off = masks_off + n_bricks * 8
        for start in range(0, n_bricks, chunk_bricks):
            cnt = min(chunk_bricks, n_bricks - start)
            stream.seek(coords_off + start * 12)
            coords = _read_exact_chunks(stream, cnt * 12, "<i4", (cnt, 3)).astype(np.int32, copy=False)
            stream.seek(masks_off + start * 8)
            masks = _read_exact_chunks(stream, cnt * 8, "<u8", (cnt,)).astype(np.uint64, copy=False)
            stream.seek(payload_off + start * 256)
            payload = _read_exact_chunks(stream, cnt * 256, "<u4", (cnt, 64)).astype(np.uint32, copy=False)
            yield header, coords, masks, payload
