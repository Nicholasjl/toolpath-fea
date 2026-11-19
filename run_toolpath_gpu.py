"""CLI runner for GPU-accelerated 0.2 mm toolpath-level hexahedral (H8) FEA."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from gpu_hex_solver import HexInterfaceConfig, recover_stress_streaming, solve_hex_gpu
from solid import Material
from toolpath_bricks import (
    FEATURE_NAMES,
    gcode_to_bricks,
    load_bricks,
    unpack_payload,
)


def render_2d_projection_preview(grid, result, stress_summary, out_png):
    """Render compact 2D Z-projection PNG (displacement, sigma_layer max, von Mises max)."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    nx, ny, _ = grid.shape
    dx, dy, _ = (float(x) for x in grid.pitch)
    ox, oy, _ = (float(x) for x in grid.origin)

    disp_map = np.full((ny, nx), -np.inf, dtype=np.float32)
    sl_map = np.full((ny, nx), -np.inf, dtype=np.float32)
    vm_map = np.full((ny, nx), -np.inf, dtype=np.float32)

    disp_mag_nodes = np.linalg.norm(result["displacement"], axis=1).astype(np.float32)
    vm_vox = stress_summary["_vm_f32"]
    sl_vox = stress_summary["_sl_f32"]
    brick_nodes = result["brick_nodes"]
    corner_offs = np.array([0, 1, 5, 6, 25, 26, 30, 31], dtype=np.int32)
    bits64 = np.arange(64, dtype=np.uint64)[None, :]

    v_cursor = 0
    chunk_bricks = 8192
    n_bricks = len(grid.brick_coords)
    for b_start in range(0, n_bricks, chunk_bricks):
        b_stop = min(n_bricks, b_start + chunk_bricks)
        occ_2d = ((grid.brick_masks[b_start:b_stop, None] >> bits64) & np.uint64(1)) != 0
        ab, ac = np.nonzero(occ_2d)
        m = len(ab)
        if not m:
            continue
        coords_ch = grid.brick_coords[b_start:b_stop]
        bn_ch = brick_nodes[b_start:b_stop]
        lx = ac & 3
        ly = (ac >> 2) & 3
        lz = ac >> 4
        ix = coords_ch[ab, 0] * 4 + lx
        iy = coords_ch[ab, 1] * 4 + ly
        bases = lx + 5 * ly + 25 * lz
        nids = bn_ch[ab[:, None], bases[:, None] + corner_offs[None, :]]
        d_vals = disp_mag_nodes[nids].mean(axis=1)
        vm_vals = vm_vox[v_cursor : v_cursor + m]
        sl_vals = sl_vox[v_cursor : v_cursor + m]
        v_cursor += m

        np.maximum.at(disp_map, (iy, ix), d_vals)
        np.maximum.at(vm_map, (iy, ix), vm_vals)
        np.maximum.at(sl_map, (iy, ix), sl_vals)

    for arr in (disp_map, sl_map, vm_map):
        arr[arr == -np.inf] = np.nan

    extent = [ox, ox + nx * dx, oy, oy + ny * dy]
    fig = Figure(figsize=(16, 5.2), dpi=150)
    FigureCanvasAgg(fig)
    fig.subplots_adjust(left=0.04, right=0.97, wspace=0.25)

    panels = [
        (disp_map, "Max |u| over Z (mm)", "viridis"),
        (sl_map, "Max sigma_layer over Z (MPa)", "coolwarm"),
        (vm_map, "Max von Mises over Z (MPa)", "inferno"),
    ]
    for i, (arr, title, cmap_name) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1)
        cmap = matplotlib.colormaps[cmap_name].copy()
        cmap.set_bad(color="#f0f2f5")
        valid = arr[np.isfinite(arr)]
        if len(valid):
            vmin = float(valid.min())
            vmax = float(np.percentile(valid, 99.5)) if i > 0 else float(valid.max())
            if vmax <= vmin:
                vmax = vmin + 1e-12
            if i == 1 and vmin < 0 < vmax:
                lim = max(abs(vmin), abs(vmax))
                vmin, vmax = -lim, lim
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("X / mm")
        ax.set_ylabel("Y / mm")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    fig.savefig(out_png, bbox_inches="tight")


def run_case(
    case_path,
    pitch=0.2,
    dz=None,
    output_dir="output/toolpath_gpu",
    bricks_cache=None,
    interface_cfg: HexInterfaceConfig = None,
    device="auto",
    rtol=1e-7,
    max_levels=6,
    coarse_dof_target=12000,
    spill_stress=False,
    keep_largest_component=True,
    render_preview=True,
):
    case_path = Path(case_path).resolve()
    cfg = json.loads(case_path.read_text(encoding="utf-8-sig"))
    base_dir = case_path.parent
    interface_cfg = interface_cfg or HexInterfaceConfig()

    gcode_rel = cfg.get("gcode", "")
    if not gcode_rel:
        raise ValueError("Case JSON must define 'gcode' for toolpath-level analysis")
    gcode_path = Path(gcode_rel)
    if not gcode_path.is_absolute():
        gcode_path = (base_dir / gcode_path).resolve()

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(bricks_cache).resolve() if bricks_cache else (out_dir / "model.tbricks")

    offset = tuple(float(v) for v in cfg.get("offset", [0.0, 0.0, 0.0]))
    build_axis = cfg.get("build_axis", "Z")
    if build_axis != "Z":
        raise ValueError("G-code toolpath brick mode requires build_axis='Z'")

    # Load from verified .tbricks cache if matching pitch & gcode, else stream-rasterize.
    t_start = time.perf_counter()
    grid = None
    need_v2 = interface_cfg.is_weakened()
    if cache_path.is_file():
        try:
            cand = load_bricks(cache_path, verify_checksum=True)
            cand_pitch = cand.meta.get("pitch_mm", [])
            version_ok = (not need_v2) or (int(cand.meta.get("flags_version", 1)) >= 2)
            if (
                version_ok
                and len(cand_pitch) == 3
                and abs(cand_pitch[0] - pitch) < 1e-9
                and abs(cand_pitch[2] - (pitch if dz is None else dz)) < 1e-9
                and cand.meta.get("source_gcode") == str(gcode_path)
            ):
                grid = cand
        except Exception:
            grid = None

    cache_hit = grid is not None
    if grid is None:
        grid = gcode_to_bricks(
            gcode_path,
            out_bricks_path=cache_path,
            pitch=pitch,
            dz=dz,
            offset=offset,
            keep_largest_component=keep_largest_component,
            device=device,
        )

    material = Material(**cfg.get("material", {}))
    result = solve_hex_gpu(
        grid,
        material=material,
        interface_cfg=interface_cfg,
        fixed=cfg.get("fixed", "X-"),
        loaded=cfg.get("loaded", "X+"),
        force=cfg.get("force", [0.0, 0.0, -10.0]),
        load_point=cfg.get("load_point"),
        build_axis=build_axis,
        rtol=rtol,
        max_levels=max_levels,
        coarse_dof_target=coarse_dof_target,
        device=device,
    )

    spill_file = (out_dir / "stress_fields.bin") if spill_stress else None
    stress_summary = recover_stress_streaming(
        grid,
        result["brick_nodes"],
        result["displacement"],
        material=material,
        interface_cfg=interface_cfg,
        build_axis=build_axis,
        spill_path=spill_file,
    )

    preview_path = out_dir / "preview.png"
    if render_preview:
        render_2d_projection_preview(grid, result, stress_summary, preview_path)

    # Strip internal numpy arrays before JSON serialization.
    stress_json = {k: v for k, v in stress_summary.items() if not k.startswith("_")}
    summary = {
        "schema_version": "toolpath_hex_gpu_1.0",
        "case": str(case_path),
        "gcode": str(gcode_path),
        "bricks_path": str(cache_path),
        "bricks_cache_hit": cache_hit,
        "bricks_file_mib": float(cache_path.stat().st_size / 1024**2),
        "mesh": grid.meta,
        "solver": result["solver_info"],
        "stress": stress_json,
        "preview_png": str(preview_path) if render_preview else None,
        "end_to_end_seconds": time.perf_counter() - t_start,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="Path to PathFEA case JSON")
    parser.add_argument("--pitch", type=float, default=0.2, help="XY voxel pitch in mm (default: 0.2)")
    parser.add_argument("--dz", type=float, default=None, help="Z voxel pitch in mm (default: pitch)")
    parser.add_argument("--output", default="output/toolpath_gpu", help="Output directory")
    parser.add_argument("--bricks-cache", default=None, help="Optional .tbricks cache file path")
    parser.add_argument("--weakened-interfaces", action="store_true", help="Enable default Route-2 interface weakening")
    parser.add_argument("--seam-factor", type=float, default=None)
    parser.add_argument("--start-stop-factor", type=float, default=None)
    parser.add_argument("--bead-interface-factor", type=float, default=None)
    parser.add_argument("--interlayer-factor", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda:0", "cpu"])
    parser.add_argument("--rtol", type=float, default=1e-7, help="Outer fp64 residual tolerance")
    parser.add_argument("--max-levels", type=int, default=6)
    parser.add_argument("--coarse-dof-target", type=int, default=12000)
    parser.add_argument("--spill-stress", action="store_true", help="Stream float32 stress fields to disk")
    parser.add_argument("--no-keep-largest", action="store_true", help="Reject instead of pruning disconnected islands")
    parser.add_argument("--no-preview", action="store_true", help="Skip 2D projection PNG")
    args = parser.parse_args(argv)

    icfg = HexInterfaceConfig.default_weakened() if args.weakened_interfaces else HexInterfaceConfig.ideal()
    if args.seam_factor is not None:
        icfg.seam_factor = args.seam_factor
    if args.start_stop_factor is not None:
        icfg.start_stop_factor = args.start_stop_factor
    if args.bead_interface_factor is not None:
        icfg.bead_interface_factor = args.bead_interface_factor
    if args.interlayer_factor is not None:
        icfg.interlayer_shear_factor = args.interlayer_factor

    summary = run_case(
        args.case,
        pitch=args.pitch,
        dz=args.dz,
        output_dir=args.output,
        bricks_cache=args.bricks_cache,
        interface_cfg=icfg,
        device=args.device,
        rtol=args.rtol,
        max_levels=args.max_levels,
        coarse_dof_target=args.coarse_dof_target,
        spill_stress=args.spill_stress,
        keep_largest_component=not args.no_keep_largest,
        render_preview=not args.no_preview,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
