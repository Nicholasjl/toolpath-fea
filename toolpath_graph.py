"""Streaming G-code to 6-DOF filament-bond network with explicit start/stop and seam modeling."""
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
from scipy.spatial import cKDTree

from toolpath_bricks import (
    FEATURE_BRIDGE,
    FEATURE_INNER_WALL,
    FEATURE_NAMES,
    FEATURE_OUTER_WALL,
    FEATURE_SOLID_INFILL,
    FEATURE_SPARSE_INFILL,
    FEATURE_UNKNOWN,
    _PAREN_RE,
    _WORD_RE,
    _classify_feature,
)


MAGIC_GRAPH = b"TPGRPH1\x00"

NODE_START = 1 << 0
NODE_STOP = 1 << 1
NODE_SEAM_LOOP = 1 << 2
NODE_SCARF_RAMP = 1 << 3
NODE_RETRACTION_START = 1 << 4
NODE_START_ZONE = 1 << 5
NODE_STOP_ZONE = 1 << 6

BOND_INTERLAYER = 1
BOND_LATERAL = 2
BOND_SEAM_CLOSED = 3
BOND_SEAM_SCARF = 4
BOND_START_ANCHOR = 5
BOND_STOP_ANCHOR = 6

BOND_TYPE_NAMES = {
    BOND_INTERLAYER: "interlayer",
    BOND_LATERAL: "lateral",
    BOND_SEAM_CLOSED: "seam_closed",
    BOND_SEAM_SCARF: "seam_scarf",
    BOND_START_ANCHOR: "start_anchor",
    BOND_STOP_ANCHOR: "stop_anchor",
}


@dataclass
class SeamConfig:
    """Explicit stiffness scaling factors for seams, start/stop zones, welds, and calibration."""

    seam_stiffness_factor: float = 0.5
    scarf_seam_stiffness_factor: float = 0.85
    start_anchor_factor: float = 0.7
    stop_anchor_factor: float = 0.6
    start_zone_beam_factor: float = 0.85
    stop_zone_beam_factor: float = 0.85
    lateral_weld_factor: float = 0.85
    interlayer_weld_factor: float = 0.85
    thermal_decay_tau_s: float = 0.0
    thermal_min_factor: float = 0.35
    beam_calibration_factor: float = 1.0
    bond_calibration_factor: float = 1.0

    @classmethod
    def ideal(cls, beam_calibration_factor: float = 1.0, bond_calibration_factor: float = 1.0):
        return cls(
            seam_stiffness_factor=1.0,
            scarf_seam_stiffness_factor=1.0,
            start_anchor_factor=1.0,
            stop_anchor_factor=1.0,
            start_zone_beam_factor=1.0,
            stop_zone_beam_factor=1.0,
            lateral_weld_factor=1.0,
            interlayer_weld_factor=1.0,
            thermal_decay_tau_s=0.0,
            beam_calibration_factor=float(beam_calibration_factor),
            bond_calibration_factor=float(bond_calibration_factor),
        )

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("seam_config must be a JSON object")
        d = dict(data)
        use_ideal = bool(d.pop("ideal", False))
        cfg = cls.ideal() if use_ideal else cls()
        valid = set(cls.__dataclass_fields__)
        unknown = set(d) - valid
        if unknown:
            raise ValueError(f"Unknown seam_config keys: {sorted(unknown)}")
        for k, v in d.items():
            fv = float(v)
            if not math.isfinite(fv):
                raise ValueError(f"seam_config.{k} must be finite")
            if k == "thermal_decay_tau_s":
                if fv < 0.0:
                    raise ValueError("seam_config.thermal_decay_tau_s must be >= 0")
            elif fv <= 0.0:
                raise ValueError(f"seam_config.{k} must be positive")
            setattr(cfg, k, fv)
        return cfg

    def to_dict(self):
        return {
            "seam_stiffness_factor": float(self.seam_stiffness_factor),
            "scarf_seam_stiffness_factor": float(self.scarf_seam_stiffness_factor),
            "start_anchor_factor": float(self.start_anchor_factor),
            "stop_anchor_factor": float(self.stop_anchor_factor),
            "start_zone_beam_factor": float(self.start_zone_beam_factor),
            "stop_zone_beam_factor": float(self.stop_zone_beam_factor),
            "lateral_weld_factor": float(self.lateral_weld_factor),
            "interlayer_weld_factor": float(self.interlayer_weld_factor),
            "thermal_decay_tau_s": float(self.thermal_decay_tau_s),
            "thermal_min_factor": float(self.thermal_min_factor),
            "beam_calibration_factor": float(self.beam_calibration_factor),
            "bond_calibration_factor": float(self.bond_calibration_factor),
        }


@dataclass
class StrengthConfig:
    """Reference allowables for screening failure index / FoS (uncalibrated reference values)."""

    sigma_beam_allow_MPa: float = 45.0
    sigma_interlayer_allow_MPa: float = 18.0
    tau_interlayer_allow_MPa: float = 14.0
    sigma_lateral_allow_MPa: float = 24.0
    tau_lateral_allow_MPa: float = 18.0
    strength_scale_exponent: float = 1.0

    @classmethod
    def from_dict(cls, data):
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("strength_config must be a JSON object")
        valid = set(cls.__dataclass_fields__)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown strength_config keys: {sorted(unknown)}")
        cfg = cls()
        for k, v in data.items():
            fv = float(v)
            if not math.isfinite(fv):
                raise ValueError(f"strength_config.{k} must be finite")
            if k == "strength_scale_exponent":
                if not 0.0 <= fv <= 3.0:
                    raise ValueError("strength_config.strength_scale_exponent must be in [0, 3]")
            elif fv <= 0.0:
                raise ValueError(f"strength_config.{k} must be positive")
            setattr(cfg, k, fv)
        return cfg

    def to_dict(self):
        return {
            "sigma_beam_allow_MPa": float(self.sigma_beam_allow_MPa),
            "sigma_interlayer_allow_MPa": float(self.sigma_interlayer_allow_MPa),
            "tau_interlayer_allow_MPa": float(self.tau_interlayer_allow_MPa),
            "sigma_lateral_allow_MPa": float(self.sigma_lateral_allow_MPa),
            "tau_lateral_allow_MPa": float(self.tau_lateral_allow_MPa),
            "strength_scale_exponent": float(self.strength_scale_exponent),
        }


@dataclass
class ToolpathGraph:
    """Compact 6-DOF beam + eccentric interface bond graph built from G-code beads."""

    node_xyz: np.ndarray        # (N, 3) float64
    node_w: np.ndarray          # (N,) float32
    node_h: np.ndarray          # (N,) float32
    node_t_sec: np.ndarray      # (N,) float32
    node_flags: np.ndarray      # (N,) uint8
    node_feature: np.ndarray    # (N,) uint8
    node_bead: np.ndarray       # (N,) int32
    beam_edges: np.ndarray      # (Mb, 2) int32
    beam_w: np.ndarray          # (Mb,) float32
    beam_h: np.ndarray          # (Mb,) float32
    beam_feature: np.ndarray    # (Mb,) uint8
    beam_flags: np.ndarray      # (Mb,) uint8
    bond_edges: np.ndarray      # (Mk, 2) int32
    bond_type: np.ndarray       # (Mk,) uint8
    bond_pc: np.ndarray         # (Mk, 3) float64
    bond_normal: np.ndarray     # (Mk, 3) float32
    bond_t1: np.ndarray         # (Mk, 3) float32
    bond_t2: np.ndarray         # (Mk, 3) float32
    bond_geom: np.ndarray       # (Mk, 4) float32 [area, l1, l2, d_eff]
    bond_dt: np.ndarray         # (Mk,) float32
    meta: dict = field(default_factory=dict)

    @property
    def num_nodes(self):
        return int(len(self.node_xyz))

    @property
    def num_beams(self):
        return int(len(self.beam_edges))

    @property
    def num_bonds(self):
        return int(len(self.bond_edges))


def iter_gcode_beads(
    path,
    offset=(0.0, 0.0, 0.0),
    default_width_mm=0.42,
    default_layer_height_mm=0.2,
    filament_diameter_mm=1.75,
    skip_features=True,
):
    """Stream G-code line-by-line and yield (layer_z, layer_h, beads) per layer.

    Each bead dict holds:
      bead_id: int
      retracted_start: bool
      segments: (S, 8) float64 [x0, y0, x1, y1, width, feature_code, t0_sec, t1_sec]
    """
    off = np.asarray(offset, dtype=np.float64)
    if off.shape != (3,) or not np.isfinite(off).all():
        raise ValueError("offset must be a finite 3-vector")

    pos = {"X": 0.0, "Y": 0.0, "Z": 0.0, "E": 0.0}
    absolute = True
    e_absolute = True
    explicit_width = None
    current_layer_h = float(default_layer_height_mm)
    filament_area = math.pi * (0.5 * float(filament_diameter_mm)) ** 2
    feature_code = FEATURE_UNKNOWN
    feature_skipped = False
    feed_mm_min = 3000.0
    t_sec = 0.0
    retracted_pending = False

    bead_id = 0
    in_bead = False
    cur_bead_segs = []
    cur_bead_retracted = False
    layer_beads = []
    current_layer_z = None
    layer_h_used = current_layer_h
    yielded_any = False

    def _close_bead():
        nonlocal in_bead, cur_bead_segs, bead_id
        if in_bead and cur_bead_segs:
            bead_id += 1
            layer_beads.append(
                {
                    "bead_id": bead_id,
                    "retracted_start": cur_bead_retracted,
                    "segments": np.asarray(cur_bead_segs, dtype=np.float64),
                }
            )
        in_bead = False
        cur_bead_segs = []

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
                            _close_bead()
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
            vals = {k: float(v) for k, v in words if k in ("X", "Y", "Z", "E", "F")}
            if "F" in vals and vals["F"] > 0:
                feed_mm_min = vals["F"]

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
                    for ax in ("X", "Y", "Z", "E"):
                        if ax in vals:
                            pos[ax] = vals[ax]
                    _close_bead()
                elif cmd in (2, 3):
                    if not (skip_features and feature_skipped):
                        raise ValueError("检测到圆弧 G2/G3，请在切片器中禁用圆弧拟合后重新导出")
                elif cmd in (0, 1):
                    ox, oy, oz, oe = pos["X"], pos["Y"], pos["Z"], pos["E"]
                    for ax in ("X", "Y", "Z", "E"):
                        if ax in vals:
                            use_abs = e_absolute if ax == "E" else absolute
                            pos[ax] = vals[ax] if use_abs else pos[ax] + vals[ax]
                    dx_m = pos["X"] - ox
                    dy_m = pos["Y"] - oy
                    dL = math.hypot(dx_m, dy_m)
                    dE = pos["E"] - oe
                    if dE < -1e-4:
                        retracted_pending = True
                    if dE <= 1e-8 or dL <= 1e-8:
                        if dL > 1e-6:
                            _close_bead()
                            t_sec += dL / max(feed_mm_min / 60.0, 1.0)
                        elif dE < -1e-6:
                            _close_bead()
                        continue
                    if skip_features and feature_skipped:
                        _close_bead()
                        t_sec += dL / max(feed_mm_min / 60.0, 1.0)
                        continue
                    if abs(pos["Z"] - oz) > 1e-5:
                        raise ValueError("检测到非平面挤出路径；此版本仅支持平面层")

                    z_world = pos["Z"] + off[2]
                    if current_layer_z is None:
                        current_layer_z = z_world
                        layer_h_used = current_layer_h
                    elif abs(z_world - current_layer_z) > 1e-4:
                        _close_bead()
                        if layer_beads:
                            yielded_any = True
                            yield current_layer_z, layer_h_used, layer_beads
                            layer_beads = []
                        current_layer_z = z_world
                        layer_h_used = current_layer_h

                    if explicit_width is not None and explicit_width > 0:
                        w = explicit_width
                    else:
                        w_est = (dE * filament_area) / (dL * max(current_layer_h, 1e-6))
                        w = w_est if 0.15 <= w_est <= 1.5 else float(default_width_mm)

                    t0 = t_sec
                    t_sec += dL / max(feed_mm_min / 60.0, 1.0)
                    t1 = t_sec

                    if not in_bead:
                        in_bead = True
                        cur_bead_retracted = retracted_pending
                        retracted_pending = False
                    cur_bead_segs.append(
                        [
                            ox + off[0],
                            oy + off[1],
                            pos["X"] + off[0],
                            pos["Y"] + off[1],
                            w,
                            float(feature_code),
                            t0,
                            t1,
                        ]
                    )
    _close_bead()
    if layer_beads:
        yielded_any = True
        yield current_layer_z, layer_h_used, layer_beads
    if not yielded_any:
        raise ValueError("未找到有效的 G0/G1 挤出线段")


def _discretize_layer_beads(layer_z, layer_h, beads, node_offset, max_seg_len, seam_zone_len, seam_zone_dist):
    """Discretize one layer's beads into nodes, beam edges, and intra-bead closed seams."""
    z_center = float(layer_z - 0.5 * layer_h)
    xyz_parts = []
    w_parts = []
    h_parts = []
    t_parts = []
    nflags_parts = []
    nfeat_parts = []
    nbead_parts = []
    poly_idx_parts = []
    trib_parts = []
    tang_parts = []

    b_edges_parts = []
    b_w_parts = []
    b_h_parts = []
    b_feat_parts = []
    b_flags_parts = []

    seam_edges = []
    seam_types = []
    seam_pc = []
    seam_n = []
    seam_t1 = []
    seam_t2 = []
    seam_geom = []

    stats = {
        "beads": 0,
        "closed_loops": 0,
        "scarf_seams": 0,
        "open_paths": 0,
        "retracted_starts": 0,
        "total_length_mm": 0.0,
    }

    cur_nid = int(node_offset)
    for bead in beads:
        segs = bead["segments"]
        bid = int(bead["bead_id"])
        retracted = bool(bead["retracted_start"])
        seg_dx = segs[:, 2] - segs[:, 0]
        seg_dy = segs[:, 3] - segs[:, 1]
        seg_lens = np.hypot(seg_dx, seg_dy)
        total_len = float(seg_lens.sum())
        if total_len <= 1e-8:
            continue

        stats["beads"] += 1
        stats["total_length_mm"] += total_len
        if retracted:
            stats["retracted_starts"] += 1

        w_start = float(segs[0, 4])
        w_end = float(segs[-1, 4])
        w_med = float(np.median(segs[:, 4]))
        p_start = segs[0, :2]
        p_end = segs[-1, 2:4]
        end_dist = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))
        is_closed = (end_dist <= max(w_start, w_end) * 1.25) and (total_len > 2.0 * max(w_start, w_end))
        is_scarf = min(w_start, w_end) < 0.75 * w_med

        # Subdivide each segment: finer near start/stop seam zones, coarser in interior.
        pts_xy = [p_start]
        node_w_list = [w_start]
        node_t_list = [float(segs[0, 6])]
        node_feat_list = [int(segs[0, 5])]
        node_s_list = [0.0]

        b_e0 = []
        b_e1 = []
        b_w_list = []
        b_feat_list = []
        b_flag_list = []

        s_acc = 0.0
        for m in range(len(segs)):
            x0, y0, x1, y1, w_m, f_m, t0_m, t1_m = segs[m]
            L_m = float(seg_lens[m])
            if L_m <= 1e-9:
                continue
            s_mid_start = s_acc
            s_mid_end = total_len - (s_acc + L_m)
            near_seam = (s_mid_start < seam_zone_dist) or (s_mid_end < seam_zone_dist)
            step = seam_zone_len if near_seam else max_seg_len
            n_sub = max(1, int(math.ceil(L_m / max(step, 0.05))))
            f_int = int(f_m)
            scarf_bit = NODE_SCARF_RAMP if (w_m < 0.75 * w_med) else 0

            for sub in range(1, n_sub + 1):
                frac = sub / n_sub
                px = x0 + frac * (x1 - x0)
                py = y0 + frac * (y1 - y0)
                s_cur = s_acc + frac * L_m
                t_cur = t0_m + frac * (t1_m - t0_m)
                pts_xy.append(np.array([px, py], dtype=np.float64))
                node_w_list.append(float(w_m))
                node_t_list.append(float(t_cur))
                node_feat_list.append(f_int)
                node_s_list.append(float(s_cur))

                idx_prev = cur_nid + len(pts_xy) - 2
                idx_curr = cur_nid + len(pts_xy) - 1
                s_beam_mid = s_cur - 0.5 * (L_m / n_sub)
                bf = scarf_bit
                if s_beam_mid < seam_zone_dist:
                    bf |= NODE_START_ZONE
                if (total_len - s_beam_mid) < seam_zone_dist:
                    bf |= NODE_STOP_ZONE
                b_e0.append(idx_prev)
                b_e1.append(idx_curr)
                b_w_list.append(float(w_m))
                b_feat_list.append(f_int)
                b_flag_list.append(bf)
            s_acc += L_m

        n_pts = len(pts_xy)
        if n_pts < 2 or not b_e0:
            continue

        pts_arr = np.asarray(pts_xy, dtype=np.float64)
        xyz_arr = np.column_stack([pts_arr, np.full(n_pts, z_center, dtype=np.float64)])
        s_arr = np.asarray(node_s_list, dtype=np.float64)

        # Node flags
        nf = np.zeros(n_pts, dtype=np.uint8)
        nf[s_arr <= seam_zone_dist] |= NODE_START_ZONE
        nf[(total_len - s_arr) <= seam_zone_dist] |= NODE_STOP_ZONE
        nf[np.asarray(node_w_list) < 0.75 * w_med] |= NODE_SCARF_RAMP
        nf[0] |= NODE_START
        nf[-1] |= NODE_STOP
        if retracted:
            nf[0] |= NODE_RETRACTION_START
        if is_closed:
            nf[0] |= NODE_SEAM_LOOP
            nf[-1] |= NODE_SEAM_LOOP
            stats["closed_loops"] += 1
            if is_scarf:
                stats["scarf_seams"] += 1
        else:
            stats["open_paths"] += 1

        # Tributary lengths and unit tangents at each node
        seg_vec = np.diff(pts_arr, axis=0)
        seg_l = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
        seg_u = seg_vec / np.maximum(seg_l[:, None], 1e-12)
        trib = np.empty(n_pts, dtype=np.float64)
        trib[0] = 0.5 * seg_l[0]
        trib[-1] = 0.5 * seg_l[-1]
        if n_pts > 2:
            trib[1:-1] = 0.5 * (seg_l[:-1] + seg_l[1:])
        tang = np.empty((n_pts, 2), dtype=np.float64)
        tang[0] = seg_u[0]
        tang[-1] = seg_u[-1]
        if n_pts > 2:
            t_mid = seg_u[:-1] + seg_u[1:]
            t_nrm = np.hypot(t_mid[:, 0], t_mid[:, 1])
            tang[1:-1] = np.where(t_nrm[:, None] > 1e-8, t_mid / np.maximum(t_nrm[:, None], 1e-12), seg_u[1:])

        # Emit explicit Z-seam / scarf-seam coupler closing the loop (never merge start & stop!)
        if is_closed:
            na = cur_nid + n_pts - 1
            nb = cur_nid
            t_seam = tang[0] + tang[-1]
            tn = math.hypot(t_seam[0], t_seam[1])
            if tn <= 1e-8:
                t_seam = tang[0]
            else:
                t_seam = t_seam / tn
            n_vec = np.array([t_seam[0], t_seam[1], 0.0], dtype=np.float32)
            t1_vec = np.array([-t_seam[1], t_seam[0], 0.0], dtype=np.float32)
            t2_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            w_seam = min(w_start, w_end)
            d_seam = max(end_dist, 0.5 * layer_h)
            seam_edges.append([na, nb])
            seam_types.append(BOND_SEAM_SCARF if is_scarf else BOND_SEAM_CLOSED)
            seam_pc.append(0.5 * (xyz_arr[-1] + xyz_arr[0]))
            seam_n.append(n_vec)
            seam_t1.append(t1_vec)
            seam_t2.append(t2_vec)
            seam_geom.append([w_seam * layer_h, w_seam, layer_h, d_seam])

        xyz_parts.append(xyz_arr)
        w_parts.append(np.asarray(node_w_list, dtype=np.float32))
        h_parts.append(np.full(n_pts, layer_h, dtype=np.float32))
        t_parts.append(np.asarray(node_t_list, dtype=np.float32))
        nflags_parts.append(nf)
        nfeat_parts.append(np.asarray(node_feat_list, dtype=np.uint8))
        nbead_parts.append(np.full(n_pts, bid, dtype=np.int32))
        poly_idx_parts.append(np.arange(n_pts, dtype=np.int32))
        trib_parts.append(trib.astype(np.float32))
        tang_parts.append(tang.astype(np.float32))

        b_edges_parts.append(np.column_stack([b_e0, b_e1]).astype(np.int32))
        b_w_parts.append(np.asarray(b_w_list, dtype=np.float32))
        b_h_parts.append(np.full(len(b_e0), layer_h, dtype=np.float32))
        b_feat_parts.append(np.asarray(b_feat_list, dtype=np.uint8))
        b_flags_parts.append(np.asarray(b_flag_list, dtype=np.uint8))

        cur_nid += n_pts

    if not xyz_parts:
        return None, stats

    layer_data = {
        "start_nid": int(node_offset),
        "end_nid": int(cur_nid),
        "layer_z": float(layer_z),
        "layer_h": float(layer_h),
        "xyz": np.concatenate(xyz_parts, axis=0),
        "w": np.concatenate(w_parts),
        "h": np.concatenate(h_parts),
        "t": np.concatenate(t_parts),
        "flags": np.concatenate(nflags_parts),
        "feature": np.concatenate(nfeat_parts),
        "bead": np.concatenate(nbead_parts),
        "poly_idx": np.concatenate(poly_idx_parts),
        "trib": np.concatenate(trib_parts),
        "tang": np.concatenate(tang_parts),
        "beam_edges": np.concatenate(b_edges_parts, axis=0),
        "beam_w": np.concatenate(b_w_parts),
        "beam_h": np.concatenate(b_h_parts),
        "beam_feature": np.concatenate(b_feat_parts),
        "beam_flags": np.concatenate(b_flags_parts),
        "seam_bonds": (
            np.asarray(seam_edges, dtype=np.int32),
            np.asarray(seam_types, dtype=np.uint8),
            np.asarray(seam_pc, dtype=np.float64),
            np.asarray(seam_n, dtype=np.float32),
            np.asarray(seam_t1, dtype=np.float32),
            np.asarray(seam_t2, dtype=np.float32),
            np.asarray(seam_geom, dtype=np.float32),
            np.zeros(len(seam_edges), dtype=np.float32),
        )
        if seam_edges
        else None,
    }
    layer_data["tree"] = cKDTree(layer_data["xyz"][:, :2])
    return layer_data, stats


def _build_lateral_bonds(layer, lateral_tol=1.15):
    """Build deduplicated lateral weld bonds and open start/stop anchor bonds on one layer."""
    n = len(layer["xyz"])
    if n < 2:
        return None
    tree = layer["tree"]
    xy = layer["xyz"][:, :2]
    w = layer["w"].astype(np.float64)
    h = layer["h"].astype(np.float64)
    bead = layer["bead"]
    pidx = layer["poly_idx"]
    flags = layer["flags"]
    tang = layer["tang"].astype(np.float64)
    trib = layer["trib"].astype(np.float64)
    start_nid = layer["start_nid"]

    max_w = float(w.max())
    r_max = max_w * lateral_tol * 1.01
    k_q = min(16, n)
    dists, nbrs = tree.query(xy, k=k_q, distance_upper_bound=r_max)
    if k_q == 1:
        dists = dists[:, None]
        nbrs = nbrs[:, None]

    pairs = []
    for i in range(n):
        ti_x, ti_y = tang[i]
        wi = w[i]
        bi = bead[i]
        pi_idx = pidx[i]
        is_open_end = bool(flags[i] & (NODE_START | NODE_STOP)) and not bool(flags[i] & NODE_SEAM_LOOP)

        best_left = (-1, 1e30)
        best_right = (-1, 1e30)
        best_end = (-1, 1e30)
        for slot in range(k_q):
            j = int(nbrs[i, slot])
            if j < 0 or j >= n or j == i:
                continue
            d = float(dists[i, slot])
            if not math.isfinite(d) or d <= 1e-8:
                continue
            if bead[j] == bi and abs(int(pidx[j]) - int(pi_idx)) < 4:
                continue
            limit = 0.5 * (wi + w[j]) * lateral_tol
            if d > limit:
                continue
            dx = xy[j, 0] - xy[i, 0]
            dy = xy[j, 1] - xy[i, 1]
            proj_n = dx * (-ti_y) + dy * ti_x
            proj_t = dx * ti_x + dy * ti_y
            if abs(proj_n) >= 0.35 * d:
                if proj_n >= 0.0 and d < best_left[1]:
                    best_left = (j, d)
                elif proj_n < 0.0 and d < best_right[1]:
                    best_right = (j, d)
            elif is_open_end and abs(proj_t) > 0.35 * d and d < best_end[1]:
                best_end = (j, d)

        for chosen in (best_left[0], best_right[0], best_end[0]):
            if chosen >= 0:
                a, b = (i, chosen) if i < chosen else (chosen, i)
                pairs.append((a, b))

    if not pairs:
        return None
    uniq = np.unique(np.asarray(pairs, dtype=np.int32), axis=0)
    ia = uniq[:, 0]
    ib = uniq[:, 1]

    vec = layer["xyz"][ib] - layer["xyz"][ia]
    dist_xy = np.hypot(vec[:, 0], vec[:, 1])
    valid = dist_xy > 1e-7
    ia, ib, vec, dist_xy = ia[valid], ib[valid], vec[valid], dist_xy[valid]
    if not len(ia):
        return None

    normal = np.column_stack([vec[:, 0] / dist_xy, vec[:, 1] / dist_xy, np.zeros_like(dist_xy)]).astype(np.float32)
    t1 = np.column_stack([-normal[:, 1], normal[:, 0], np.zeros_like(dist_xy)]).astype(np.float32)
    t2 = np.zeros_like(normal)
    t2[:, 2] = 1.0

    wa, wb = w[ia], w[ib]
    p_c = layer["xyz"][ia] + (wa / (wa + wb))[:, None] * vec
    l1 = np.minimum(trib[ia], trib[ib])
    l2 = 0.5 * (h[ia] + h[ib])
    area = l1 * l2
    d_eff = np.maximum(dist_xy, 0.5 * np.minimum(wa, wb))
    geom = np.column_stack([area, l1, l2, d_eff]).astype(np.float32)

    fa, fb = flags[ia], flags[ib]
    open_start = ((fa & NODE_START) != 0) & ((fa & NODE_SEAM_LOOP) == 0)
    open_start |= ((fb & NODE_START) != 0) & ((fb & NODE_SEAM_LOOP) == 0)
    open_stop = ((fa & NODE_STOP) != 0) & ((fa & NODE_SEAM_LOOP) == 0)
    open_stop |= ((fb & NODE_STOP) != 0) & ((fb & NODE_SEAM_LOOP) == 0)
    btype = np.full(len(ia), BOND_LATERAL, dtype=np.uint8)
    btype[open_start] = BOND_START_ANCHOR
    btype[open_stop] = BOND_STOP_ANCHOR

    edges = np.column_stack([ia + start_nid, ib + start_nid]).astype(np.int32)
    dt = np.zeros(len(ia), dtype=np.float32)
    return edges, btype, p_c, normal, t1, t2, geom, dt


def _build_interlayer_bonds(cur_layer, prev_layer, inter_tol=1.05):
    """Build interlayer bonds between cur_layer (Z_k) and prev_layer (Z_{k-1})."""
    n_cur = len(cur_layer["xyz"])
    n_prev = len(prev_layer["xyz"])
    if n_cur == 0 or n_prev == 0:
        return None
    tree_prev = prev_layer["tree"]
    xy_cur = cur_layer["xyz"][:, :2]
    xy_prev = prev_layer["xyz"][:, :2]
    w_cur = cur_layer["w"].astype(np.float64)
    w_prev = prev_layer["w"].astype(np.float64)
    h_cur = cur_layer["h"].astype(np.float64)
    h_prev = prev_layer["h"].astype(np.float64)
    trib_cur = cur_layer["trib"].astype(np.float64)
    trib_prev = prev_layer["trib"].astype(np.float64)
    bead_prev = prev_layer["bead"]

    r_max = 0.5 * (float(w_cur.max()) + float(w_prev.max())) * inter_tol * 1.01
    k_q = min(8, n_prev)
    dists, nbrs = tree_prev.query(xy_cur, k=k_q, distance_upper_bound=r_max)
    if k_q == 1:
        dists = dists[:, None]
        nbrs = nbrs[:, None]

    # Pick up to 2 nearest distinct-bead candidates in Z_{k-1} per node in Z_k.
    ia_list = []
    ib_list = []
    ov_list = []
    for i in range(n_cur):
        wi = w_cur[i]
        seen_beads = set()
        cand_j = []
        cand_ov = []
        for slot in range(k_q):
            j = int(nbrs[i, slot])
            if j < 0 or j >= n_prev:
                continue
            d = float(dists[i, slot])
            if not math.isfinite(d):
                continue
            bj = int(bead_prev[j])
            if bj in seen_beads:
                continue
            limit = 0.5 * (wi + w_prev[j]) * inter_tol
            if d > limit:
                continue
            seen_beads.add(bj)
            ov = min(1.0, max(0.1, (0.5 * (wi + w_prev[j]) - d) / max(min(wi, w_prev[j]), 1e-6)))
            cand_j.append(j)
            cand_ov.append(ov)
            if len(cand_j) == 2:
                break
        if cand_j:
            tot_ov = sum(cand_ov)
            scale_ov = min(1.0, 1.0 / tot_ov) if tot_ov > 1.0 else 1.0
            for j, ov in zip(cand_j, cand_ov):
                ia_list.append(i)
                ib_list.append(j)
                ov_list.append(ov * scale_ov)

    if not ia_list:
        return None
    ia = np.asarray(ia_list, dtype=np.int32)
    ib = np.asarray(ib_list, dtype=np.int32)
    ov = np.asarray(ov_list, dtype=np.float64)

    p_c = 0.5 * (cur_layer["xyz"][ia] + prev_layer["xyz"][ib])
    p_c[:, 2] = cur_layer["layer_z"] - cur_layer["layer_h"]

    m = len(ia)
    normal = np.zeros((m, 3), dtype=np.float32)
    normal[:, 2] = 1.0
    t1 = np.column_stack([cur_layer["tang"][ia], np.zeros(m, dtype=np.float32)]).astype(np.float32)
    t2 = np.cross(normal, t1).astype(np.float32)

    l1 = np.minimum(trib_cur[ia], trib_prev[ib])
    l2 = np.minimum(w_cur[ia], w_prev[ib])
    area = w_cur[ia] * trib_cur[ia] * ov
    d_eff = 0.5 * (h_cur[ia] + h_prev[ib])
    geom = np.column_stack([area, l1, l2, d_eff]).astype(np.float32)

    dt = np.maximum(0.0, cur_layer["t"][ia] - prev_layer["t"][ib]).astype(np.float32)
    edges = np.column_stack([ia + cur_layer["start_nid"], ib + prev_layer["start_nid"]]).astype(np.int32)
    btype = np.full(m, BOND_INTERLAYER, dtype=np.uint8)
    return edges, btype, p_c, normal, t1, t2, geom, dt


def filter_connected_graph(graph: ToolpathGraph, keep_largest_component=True):
    """Verify global 6-DOF connectivity across beams + bonds; keep largest or reject."""
    n = graph.num_nodes
    if n == 0:
        raise ValueError("走线网络为空")
    u = np.concatenate([graph.beam_edges[:, 0], graph.bond_edges[:, 0]])
    v = np.concatenate([graph.beam_edges[:, 1], graph.bond_edges[:, 1]])
    g = coo_matrix((np.ones(len(u) * 2, dtype=np.int8), (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(g, directed=False, return_labels=True)
    if n_comp == 1:
        return graph, {"components": 1, "kept_component": 0, "dropped_components": 0, "dropped_nodes": 0}

    counts = np.bincount(labels)
    best = int(np.argmax(counts))
    if not keep_largest_component:
        raise ValueError(f"检测到 {n_comp} 个不连通走线子图；请清理孤岛路径或启用 keep_largest_component")

    keep_node_mask = labels == best
    remap = np.full(n, -1, dtype=np.int32)
    kept_idx = np.flatnonzero(keep_node_mask)
    remap[kept_idx] = np.arange(len(kept_idx), dtype=np.int32)

    b_keep = keep_node_mask[graph.beam_edges[:, 0]] & keep_node_mask[graph.beam_edges[:, 1]]
    k_keep = keep_node_mask[graph.bond_edges[:, 0]] & keep_node_mask[graph.bond_edges[:, 1]]

    filtered = ToolpathGraph(
        node_xyz=np.ascontiguousarray(graph.node_xyz[kept_idx]),
        node_w=np.ascontiguousarray(graph.node_w[kept_idx]),
        node_h=np.ascontiguousarray(graph.node_h[kept_idx]),
        node_t_sec=np.ascontiguousarray(graph.node_t_sec[kept_idx]),
        node_flags=np.ascontiguousarray(graph.node_flags[kept_idx]),
        node_feature=np.ascontiguousarray(graph.node_feature[kept_idx]),
        node_bead=np.ascontiguousarray(graph.node_bead[kept_idx]),
        beam_edges=np.ascontiguousarray(remap[graph.beam_edges[b_keep]]),
        beam_w=np.ascontiguousarray(graph.beam_w[b_keep]),
        beam_h=np.ascontiguousarray(graph.beam_h[b_keep]),
        beam_feature=np.ascontiguousarray(graph.beam_feature[b_keep]),
        beam_flags=np.ascontiguousarray(graph.beam_flags[b_keep]),
        bond_edges=np.ascontiguousarray(remap[graph.bond_edges[k_keep]]),
        bond_type=np.ascontiguousarray(graph.bond_type[k_keep]),
        bond_pc=np.ascontiguousarray(graph.bond_pc[k_keep]),
        bond_normal=np.ascontiguousarray(graph.bond_normal[k_keep]),
        bond_t1=np.ascontiguousarray(graph.bond_t1[k_keep]),
        bond_t2=np.ascontiguousarray(graph.bond_t2[k_keep]),
        bond_geom=np.ascontiguousarray(graph.bond_geom[k_keep]),
        bond_dt=np.ascontiguousarray(graph.bond_dt[k_keep]),
        meta=dict(graph.meta),
    )
    conn = {
        "components": int(n_comp),
        "kept_component": best,
        "dropped_components": int(n_comp - 1),
        "dropped_nodes": int(n - len(kept_idx)),
    }
    return filtered, conn


def gcode_to_toolpath_graph(
    gcode_path,
    out_graph_path=None,
    offset=(0.0, 0.0, 0.0),
    max_seg_len=1.0,
    seam_zone_len=0.4,
    seam_zone_dist=1.0,
    lateral_tol=1.15,
    inter_tol=1.05,
    default_width_mm=0.42,
    default_layer_height_mm=0.2,
    filament_diameter_mm=1.75,
    skip_features=True,
    keep_largest_component=True,
):
    """Stream G-code with a 2-layer sliding window to build a 6-DOF filament-bond graph."""
    t0 = time.perf_counter()
    gcode_path = Path(gcode_path)
    if not gcode_path.is_file():
        raise FileNotFoundError(f"G-code not found: {gcode_path}")

    src_hash = hashlib.sha256()
    with gcode_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            src_hash.update(chunk)

    if out_graph_path is not None:
        out_p = Path(out_graph_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        scratch_dir = out_p.parent / f"._{out_p.stem}_{uuid.uuid4().hex}_gscratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        cleanup_scratch = lambda: shutil.rmtree(scratch_dir, ignore_errors=True)
    else:
        tmp_obj = tempfile.TemporaryDirectory(prefix="tpgraph_")
        scratch_dir = Path(tmp_obj.name)
        cleanup_scratch = tmp_obj.cleanup

    node_files = []
    beam_files = []
    bond_files = []
    prev_layer = None
    next_nid = 0
    layer_count = 0
    agg_stats = {
        "beads": 0,
        "closed_loops": 0,
        "scarf_seams": 0,
        "open_paths": 0,
        "retracted_starts": 0,
        "total_length_mm": 0.0,
    }

    def _spill_array_group(path, arrays):
        with path.open("wb") as fh:
            for arr in arrays:
                arr_c = np.ascontiguousarray(arr)
                fh.write(struct.pack("<Q", len(arr_c)))
                fh.write(arr_c.tobytes())

    def _spill_bonds(idx, bond_tuple):
        if bond_tuple is None or len(bond_tuple[0]) == 0:
            return
        bf = scratch_dir / f"bonds_{idx:06d}.bin"
        _spill_array_group(bf, bond_tuple)
        bond_files.append(bf)

    def _spill_layer(idx, layer):
        nf = scratch_dir / f"nodes_{idx:05d}.bin"
        _spill_array_group(
            nf,
            [
                layer["xyz"].astype("<f8", copy=False),
                layer["w"].astype("<f4", copy=False),
                layer["h"].astype("<f4", copy=False),
                layer["t"].astype("<f4", copy=False),
                layer["flags"].astype("u1", copy=False),
                layer["feature"].astype("u1", copy=False),
                layer["bead"].astype("<i4", copy=False),
            ],
        )
        node_files.append(nf)

        ef = scratch_dir / f"beams_{idx:05d}.bin"
        _spill_array_group(
            ef,
            [
                layer["beam_edges"].astype("<i4", copy=False),
                layer["beam_w"].astype("<f4", copy=False),
                layer["beam_h"].astype("<f4", copy=False),
                layer["beam_feature"].astype("u1", copy=False),
                layer["beam_flags"].astype("u1", copy=False),
            ],
        )
        beam_files.append(ef)
        if layer["seam_bonds"] is not None:
            _spill_bonds(len(bond_files), layer["seam_bonds"])

    try:
        for lz, lh, beads in iter_gcode_beads(
            gcode_path,
            offset=offset,
            default_width_mm=default_width_mm,
            default_layer_height_mm=default_layer_height_mm,
            filament_diameter_mm=filament_diameter_mm,
            skip_features=skip_features,
        ):
            cur_layer, lstats = _discretize_layer_beads(
                lz, lh, beads, next_nid, max_seg_len, seam_zone_len, seam_zone_dist
            )
            if cur_layer is None:
                continue
            layer_count += 1
            next_nid = cur_layer["end_nid"]
            for k in agg_stats:
                agg_stats[k] += lstats[k]

            lat_bonds = _build_lateral_bonds(cur_layer, lateral_tol=lateral_tol)
            _spill_bonds(len(bond_files), lat_bonds)
            if prev_layer is not None:
                inter_bonds = _build_interlayer_bonds(cur_layer, prev_layer, inter_tol=inter_tol)
                _spill_bonds(len(bond_files), inter_bonds)
                _spill_layer(layer_count - 1, prev_layer)
                prev_layer = None
            prev_layer = cur_layer

        if prev_layer is not None:
            _spill_layer(layer_count, prev_layer)
            prev_layer = None

        if not node_files:
            raise ValueError("未能从 G-code 离散任何走线节点")

        def _load_group(files, specs):
            cols = [[] for _ in specs]
            for f in files:
                with f.open("rb") as fh:
                    for c_idx, (dtype, tail_shape) in enumerate(specs):
                        (cnt,) = struct.unpack("<Q", fh.read(8))
                        row_bytes = np.dtype(dtype).itemsize * (int(np.prod(tail_shape)) if tail_shape else 1)
                        raw = fh.read(cnt * row_bytes)
                        shape = (cnt,) + tail_shape
                        cols[c_idx].append(np.frombuffer(raw, dtype=dtype).reshape(shape).copy())
            return [np.concatenate(c, axis=0) if c else np.empty((0,) + s[1], dtype=s[0]) for c, s in zip(cols, specs)]

        n_xyz, n_w, n_h, n_t, n_flags, n_feat, n_bead = _load_group(
            node_files,
            [("<f8", (3,)), ("<f4", ()), ("<f4", ()), ("<f4", ()), ("u1", ()), ("u1", ()), ("<i4", ())],
        )
        b_edges, b_w, b_h, b_feat, b_flags = _load_group(
            beam_files,
            [("<i4", (2,)), ("<f4", ()), ("<f4", ()), ("u1", ()), ("u1", ())],
        )
        if bond_files:
            k_edges, k_type, k_pc, k_n, k_t1, k_t2, k_geom, k_dt = _load_group(
                bond_files,
                [("<i4", (2,)), ("u1", ()), ("<f8", (3,)), ("<f4", (3,)), ("<f4", (3,)), ("<f4", (3,)), ("<f4", (4,)), ("<f4", ())],
            )
        else:
            k_edges = np.empty((0, 2), dtype=np.int32)
            k_type = np.empty((0,), dtype=np.uint8)
            k_pc = np.empty((0, 3), dtype=np.float64)
            k_n = np.empty((0, 3), dtype=np.float32)
            k_t1 = np.empty((0, 3), dtype=np.float32)
            k_t2 = np.empty((0, 3), dtype=np.float32)
            k_geom = np.empty((0, 4), dtype=np.float32)
            k_dt = np.empty((0,), dtype=np.float32)
    finally:
        cleanup_scratch()

    graph = ToolpathGraph(
        node_xyz=n_xyz,
        node_w=n_w,
        node_h=n_h,
        node_t_sec=n_t,
        node_flags=n_flags,
        node_feature=n_feat,
        node_bead=n_bead,
        beam_edges=b_edges,
        beam_w=b_w,
        beam_h=b_h,
        beam_feature=b_feat,
        beam_flags=b_flags,
        bond_edges=k_edges,
        bond_type=k_type,
        bond_pc=k_pc,
        bond_normal=k_n,
        bond_t1=k_t1,
        bond_t2=k_t2,
        bond_geom=k_geom,
        bond_dt=k_dt,
        meta={},
    )
    graph, conn_info = filter_connected_graph(graph, keep_largest_component=keep_largest_component)

    beam_lens = np.linalg.norm(graph.node_xyz[graph.beam_edges[:, 1]] - graph.node_xyz[graph.beam_edges[:, 0]], axis=1)
    est_vol = float((beam_lens * graph.beam_w * graph.beam_h).sum())
    bond_counts = {
        BOND_TYPE_NAMES[t]: int((graph.bond_type == t).sum())
        for t in sorted(BOND_TYPE_NAMES)
        if (graph.bond_type == t).any()
    }
    feature_beam_counts = {
        FEATURE_NAMES[f]: int((graph.beam_feature == f).sum())
        for f in sorted(FEATURE_NAMES)
        if (graph.beam_feature == f).any()
    }
    graph.meta = {
        "source_gcode": str(gcode_path.resolve()),
        "source_sha256": src_hash.hexdigest(),
        "layer_count": int(layer_count),
        "max_seg_len_mm": float(max_seg_len),
        "seam_zone_len_mm": float(seam_zone_len),
        "seam_zone_dist_mm": float(seam_zone_dist),
        "num_nodes": graph.num_nodes,
        "num_beams": graph.num_beams,
        "num_bonds": graph.num_bonds,
        "free_total_dofs_estimate": graph.num_nodes * 6,
        "beads_summary": agg_stats,
        "estimated_beam_volume_mm3": est_vol,
        "bond_counts": bond_counts,
        "feature_beam_counts": feature_beam_counts,
        "connectivity": conn_info,
        "build_seconds": time.perf_counter() - t0,
    }
    if out_graph_path is not None:
        save_toolpath_graph(out_graph_path, graph)
    return graph


def _graph_arrays_in_order(graph: ToolpathGraph):
    return [
        ("node_xyz", np.ascontiguousarray(graph.node_xyz, dtype="<f8")),
        ("node_w", np.ascontiguousarray(graph.node_w, dtype="<f4")),
        ("node_h", np.ascontiguousarray(graph.node_h, dtype="<f4")),
        ("node_t_sec", np.ascontiguousarray(graph.node_t_sec, dtype="<f4")),
        ("node_flags", np.ascontiguousarray(graph.node_flags, dtype="u1")),
        ("node_feature", np.ascontiguousarray(graph.node_feature, dtype="u1")),
        ("node_bead", np.ascontiguousarray(graph.node_bead, dtype="<i4")),
        ("beam_edges", np.ascontiguousarray(graph.beam_edges, dtype="<i4")),
        ("beam_w", np.ascontiguousarray(graph.beam_w, dtype="<f4")),
        ("beam_h", np.ascontiguousarray(graph.beam_h, dtype="<f4")),
        ("beam_feature", np.ascontiguousarray(graph.beam_feature, dtype="u1")),
        ("beam_flags", np.ascontiguousarray(graph.beam_flags, dtype="u1")),
        ("bond_edges", np.ascontiguousarray(graph.bond_edges, dtype="<i4")),
        ("bond_type", np.ascontiguousarray(graph.bond_type, dtype="u1")),
        ("bond_pc", np.ascontiguousarray(graph.bond_pc, dtype="<f8")),
        ("bond_normal", np.ascontiguousarray(graph.bond_normal, dtype="<f4")),
        ("bond_t1", np.ascontiguousarray(graph.bond_t1, dtype="<f4")),
        ("bond_t2", np.ascontiguousarray(graph.bond_t2, dtype="<f4")),
        ("bond_geom", np.ascontiguousarray(graph.bond_geom, dtype="<f4")),
        ("bond_dt", np.ascontiguousarray(graph.bond_dt, dtype="<f4")),
    ]


def _graph_payload_sha256(arr_list):
    h = hashlib.sha256()
    for _name, arr in arr_list:
        h.update(arr.tobytes())
    return h.hexdigest()


def save_toolpath_graph(path, graph: ToolpathGraph):
    """Atomically write ToolpathGraph to binary .tgraph file with SHA-256 verification."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_list = _graph_arrays_in_order(graph)
    header = {
        "format": "TPGRPH1",
        "num_nodes": graph.num_nodes,
        "num_beams": graph.num_beams,
        "num_bonds": graph.num_bonds,
        "payload_sha256": _graph_payload_sha256(arr_list),
        "meta": graph.meta,
    }
    header_bytes = json.dumps(header, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as fh:
            fh.write(MAGIC_GRAPH)
            fh.write(struct.pack("<Q", len(header_bytes)))
            fh.write(header_bytes)
            step = 16 * 1024 * 1024
            for _name, arr in arr_list:
                mv = memoryview(arr).cast("B")
                for off in range(0, len(mv), step):
                    fh.write(mv[off : off + step])
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return path


def _read_exact(stream, total_bytes, dtype, shape):
    out = np.empty(shape, dtype=dtype)
    mv = memoryview(out).cast("B")
    pos = 0
    step = 16 * 1024 * 1024
    while pos < total_bytes:
        got = stream.readinto(mv[pos : min(total_bytes, pos + step)])
        if not got:
            raise ValueError("Unexpected EOF reading .tgraph file")
        pos += got
    return out


def load_toolpath_graph(path, verify_checksum=True):
    """Load ToolpathGraph from .tgraph using explicit chunked reads (never np.memmap)."""
    path = Path(path)
    with path.open("rb") as fh:
        if fh.read(len(MAGIC_GRAPH)) != MAGIC_GRAPH:
            raise ValueError("Invalid .tgraph magic header")
        (jlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(jlen).decode("utf-8"))
        n = int(header["num_nodes"])
        mb = int(header["num_beams"])
        mk = int(header["num_bonds"])
        specs = [
            ("<f8", (n, 3)),
            ("<f4", (n,)),
            ("<f4", (n,)),
            ("<f4", (n,)),
            ("u1", (n,)),
            ("u1", (n,)),
            ("<i4", (n,)),
            ("<i4", (mb, 2)),
            ("<f4", (mb,)),
            ("<f4", (mb,)),
            ("u1", (mb,)),
            ("u1", (mb,)),
            ("<i4", (mk, 2)),
            ("u1", (mk,)),
            ("<f8", (mk, 3)),
            ("<f4", (mk, 3)),
            ("<f4", (mk, 3)),
            ("<f4", (mk, 3)),
            ("<f4", (mk, 4)),
            ("<f4", (mk,)),
        ]
        arrays = []
        for dt, shp in specs:
            nbytes = np.dtype(dt).itemsize * int(np.prod(shp))
            arrays.append(_read_exact(fh, nbytes, dt, shp))

    graph = ToolpathGraph(
        node_xyz=arrays[0].astype(np.float64, copy=False),
        node_w=arrays[1].astype(np.float32, copy=False),
        node_h=arrays[2].astype(np.float32, copy=False),
        node_t_sec=arrays[3].astype(np.float32, copy=False),
        node_flags=arrays[4].astype(np.uint8, copy=False),
        node_feature=arrays[5].astype(np.uint8, copy=False),
        node_bead=arrays[6].astype(np.int32, copy=False),
        beam_edges=arrays[7].astype(np.int32, copy=False),
        beam_w=arrays[8].astype(np.float32, copy=False),
        beam_h=arrays[9].astype(np.float32, copy=False),
        beam_feature=arrays[10].astype(np.uint8, copy=False),
        beam_flags=arrays[11].astype(np.uint8, copy=False),
        bond_edges=arrays[12].astype(np.int32, copy=False),
        bond_type=arrays[13].astype(np.uint8, copy=False),
        bond_pc=arrays[14].astype(np.float64, copy=False),
        bond_normal=arrays[15].astype(np.float32, copy=False),
        bond_t1=arrays[16].astype(np.float32, copy=False),
        bond_t2=arrays[17].astype(np.float32, copy=False),
        bond_geom=arrays[18].astype(np.float32, copy=False),
        bond_dt=arrays[19].astype(np.float32, copy=False),
        meta=header.get("meta", {}),
    )
    if verify_checksum and _graph_payload_sha256(_graph_arrays_in_order(graph)) != header["payload_sha256"]:
        raise ValueError(".tgraph payload checksum mismatch")
    return graph


def build_synthetic_graph(
    node_xyz,
    node_w,
    node_h,
    beam_edges,
    bond_edges=None,
    bond_type=None,
    bond_pc=None,
    bond_normal=None,
    bond_t1=None,
    bond_t2=None,
    bond_geom=None,
    bond_dt=None,
    node_flags=None,
    beam_flags=None,
    node_feature=None,
    beam_feature=None,
    node_bead=None,
):
    """Helper to construct clean ToolpathGraph instances for analytical unit tests."""
    node_xyz = np.asarray(node_xyz, dtype=np.float64)
    n = len(node_xyz)
    beam_edges = np.asarray(beam_edges, dtype=np.int32).reshape(-1, 2)
    mb = len(beam_edges)
    nw = np.broadcast_to(np.asarray(node_w, dtype=np.float32), (n,)).copy()
    nh = np.broadcast_to(np.asarray(node_h, dtype=np.float32), (n,)).copy()
    bw = 0.5 * (nw[beam_edges[:, 0]] + nw[beam_edges[:, 1]]) if mb else np.empty((0,), dtype=np.float32)
    bh = 0.5 * (nh[beam_edges[:, 0]] + nh[beam_edges[:, 1]]) if mb else np.empty((0,), dtype=np.float32)
    if bond_edges is None or len(bond_edges) == 0:
        mk = 0
        k_edges = np.empty((0, 2), dtype=np.int32)
        k_type = np.empty((0,), dtype=np.uint8)
        k_pc = np.empty((0, 3), dtype=np.float64)
        k_n = np.empty((0, 3), dtype=np.float32)
        k_t1 = np.empty((0, 3), dtype=np.float32)
        k_t2 = np.empty((0, 3), dtype=np.float32)
        k_geom = np.empty((0, 4), dtype=np.float32)
        k_dt = np.empty((0,), dtype=np.float32)
    else:
        k_edges = np.asarray(bond_edges, dtype=np.int32).reshape(-1, 2)
        mk = len(k_edges)
        k_type = np.asarray(bond_type, dtype=np.uint8)
        k_pc = np.asarray(bond_pc, dtype=np.float64)
        k_n = np.asarray(bond_normal, dtype=np.float32)
        k_t1 = np.asarray(bond_t1, dtype=np.float32)
        k_t2 = np.asarray(bond_t2, dtype=np.float32)
        k_geom = np.asarray(bond_geom, dtype=np.float32)
        k_dt = np.zeros(mk, dtype=np.float32) if bond_dt is None else np.asarray(bond_dt, dtype=np.float32)

    return ToolpathGraph(
        node_xyz=node_xyz,
        node_w=nw,
        node_h=nh,
        node_t_sec=np.zeros(n, dtype=np.float32),
        node_flags=np.zeros(n, dtype=np.uint8) if node_flags is None else np.asarray(node_flags, dtype=np.uint8),
        node_feature=np.full(n, FEATURE_SOLID_INFILL, dtype=np.uint8) if node_feature is None else np.asarray(node_feature, dtype=np.uint8),
        node_bead=np.ones(n, dtype=np.int32) if node_bead is None else np.asarray(node_bead, dtype=np.int32),
        beam_edges=beam_edges,
        beam_w=bw,
        beam_h=bh,
        beam_feature=np.full(mb, FEATURE_SOLID_INFILL, dtype=np.uint8) if beam_feature is None else np.asarray(beam_feature, dtype=np.uint8),
        beam_flags=np.zeros(mb, dtype=np.uint8) if beam_flags is None else np.asarray(beam_flags, dtype=np.uint8),
        bond_edges=k_edges,
        bond_type=k_type,
        bond_pc=k_pc,
        bond_normal=k_n,
        bond_t1=k_t1,
        bond_t2=k_t2,
        bond_geom=k_geom,
        bond_dt=k_dt,
        meta={"num_nodes": n, "num_beams": mb, "num_bonds": mk},
    )
