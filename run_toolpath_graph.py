"""CLI runner for 6-DOF filament-bond network FEA with explicit start/stop and seam modeling."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from beam_bond_solver import recover_graph_stress_streaming, solve_toolpath_graph_gpu
from solid import Material
from toolpath_graph import (
    BOND_INTERLAYER,
    BOND_LATERAL,
    BOND_SEAM_CLOSED,
    BOND_SEAM_SCARF,
    BOND_START_ANCHOR,
    BOND_STOP_ANCHOR,
    SeamConfig,
    gcode_to_toolpath_graph,
    load_toolpath_graph,
)


def render_graph_preview(graph, result, stress_summary, out_png, pixel_mm=0.25):
    """Render compact 3-panel 2D preview: displacement, weld peeling, and start/stop seam hotspots."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    xyz = graph.node_xyz
    xmin, ymin = float(xyz[:, 0].min()) - 1.0, float(xyz[:, 1].min()) - 1.0
    xmax, ymax = float(xyz[:, 0].max()) + 1.0, float(xyz[:, 1].max()) + 1.0
    nx = max(64, int(np.ceil((xmax - xmin) / pixel_mm)))
    ny = max(64, int(np.ceil((ymax - ymin) / pixel_mm)))

    disp_map = np.full((ny, nx), -np.inf, dtype=np.float32)
    peel_map = np.full((ny, nx), -np.inf, dtype=np.float32)
    bg_mask = np.zeros((ny, nx), dtype=bool)

    disp_mag = np.linalg.norm(result["displacement"], axis=1).astype(np.float32)
    e = graph.beam_edges
    p0 = xyz[e[:, 0], :2]
    p1 = xyz[e[:, 1], :2]
    d0 = disp_mag[e[:, 0]]
    d1 = disp_mag[e[:, 1]]
    for frac in (0.0, 0.33, 0.67, 1.0):
        px = p0[:, 0] + frac * (p1[:, 0] - p0[:, 0])
        py = p0[:, 1] + frac * (p1[:, 1] - p0[:, 1])
        dv = (1.0 - frac) * d0 + frac * d1
        ix = np.clip(((px - xmin) / pixel_mm).astype(np.int32), 0, nx - 1)
        iy = np.clip(((py - ymin) / pixel_mm).astype(np.int32), 0, ny - 1)
        np.maximum.at(disp_map, (iy, ix), dv.astype(np.float32))
        bg_mask[iy, ix] = True

    weld_mask = np.isin(graph.bond_type, [BOND_INTERLAYER, BOND_LATERAL])
    if weld_mask.any():
        w_pc = graph.bond_pc[weld_mask]
        w_sn = stress_summary["_bond_sn"][weld_mask]
        ix = np.clip(((w_pc[:, 0] - xmin) / pixel_mm).astype(np.int32), 0, nx - 1)
        iy = np.clip(((w_pc[:, 1] - ymin) / pixel_mm).astype(np.int32), 0, ny - 1)
        np.maximum.at(peel_map, (iy, ix), w_sn)

    disp_map[disp_map == -np.inf] = np.nan
    peel_map[peel_map == -np.inf] = np.nan

    extent = [xmin, xmin + nx * pixel_mm, ymin, ymin + ny * pixel_mm]
    fig = Figure(figsize=(16.5, 5.3), dpi=150)
    FigureCanvasAgg(fig)
    fig.subplots_adjust(left=0.04, right=0.97, wspace=0.26)

    # Panel 1: Displacement
    ax0 = fig.add_subplot(1, 3, 1)
    cm0 = matplotlib.colormaps["viridis"].copy()
    cm0.set_bad(color="#f0f2f5")
    valid0 = disp_map[np.isfinite(disp_map)]
    im0 = ax0.imshow(
        disp_map,
        origin="lower",
        extent=extent,
        cmap=cm0,
        vmin=float(valid0.min()) if len(valid0) else 0.0,
        vmax=float(valid0.max()) if len(valid0) else 1.0,
        interpolation="nearest",
    )
    ax0.set_title("Network |u| max over Z (mm)", fontsize=10)
    ax0.set_xlabel("X / mm")
    ax0.set_ylabel("Y / mm")
    fig.colorbar(im0, ax=ax0, fraction=0.045, pad=0.03)

    # Panel 2: Weld peeling stress
    ax1 = fig.add_subplot(1, 3, 2)
    cm1 = matplotlib.colormaps["coolwarm"].copy()
    cm1.set_bad(color="#f0f2f5")
    valid1 = peel_map[np.isfinite(peel_map)]
    if len(valid1):
        vmax1 = max(float(np.percentile(np.abs(valid1), 99.5)), 1e-6)
        vmin1 = -vmax1
    else:
        vmin1, vmax1 = -1.0, 1.0
    im1 = ax1.imshow(peel_map, origin="lower", extent=extent, cmap=cm1, vmin=vmin1, vmax=vmax1, interpolation="nearest")
    ax1.set_title("Interlayer / Lateral Weld sigma_n (MPa)", fontsize=10)
    ax1.set_xlabel("X / mm")
    ax1.set_ylabel("Y / mm")
    fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.03)

    # Panel 3: Start/Stop & Z-seam / Scarf-seam hotspots
    ax2 = fig.add_subplot(1, 3, 3)
    bg_img = np.where(bg_mask, 0.88, 0.97)
    ax2.imshow(bg_img, origin="lower", extent=extent, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    seam_mask = np.isin(graph.bond_type, [BOND_SEAM_CLOSED, BOND_SEAM_SCARF, BOND_START_ANCHOR, BOND_STOP_ANCHOR])
    if seam_mask.any():
        s_pc = graph.bond_pc[seam_mask]
        s_eq = stress_summary["_bond_eq_peak"][seam_mask]
        order = np.argsort(s_eq)
        if len(order) > 15000:
            # Keep background subsample + top 4000 hottest seams/anchors
            bg_sub = order[: len(order) - 4000 : max(1, (len(order) - 4000) // 6000)]
            order = np.concatenate([bg_sub, order[-4000:]])
        s_pc = s_pc[order]
        s_eq = s_eq[order]
        vmax2 = max(float(np.percentile(s_eq, 99.5)), 1e-6)
        sc = ax2.scatter(
            s_pc[:, 0],
            s_pc[:, 1],
            c=s_eq,
            cmap="inferno",
            vmin=0.0,
            vmax=vmax2,
            s=4.0 + 18.0 * np.clip(s_eq / vmax2, 0.0, 1.0),
            alpha=0.85,
            linewidths=0.0,
        )
        fig.colorbar(sc, ax=ax2, fraction=0.045, pad=0.03)
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])
    ax2.set_title("Start/Stop & Z-Seam Eq. Traction (MPa)", fontsize=10)
    ax2.set_xlabel("X / mm")
    ax2.set_ylabel("Y / mm")

    fig.savefig(out_png, bbox_inches="tight")


def run_graph_case(
    case_path,
    max_seg_len=1.0,
    output_dir="output/toolpath_graph",
    graph_cache=None,
    seam_cfg: SeamConfig = None,
    device="auto",
    rtol=1e-7,
    spill_stress=False,
    keep_largest_component=True,
    render_preview=True,
):
    case_path = Path(case_path).resolve()
    cfg = json.loads(case_path.read_text(encoding="utf-8-sig"))
    base_dir = case_path.parent
    gcode_rel = cfg.get("gcode", "")
    if not gcode_rel:
        raise ValueError("Case JSON must define 'gcode'")
    gcode_path = Path(gcode_rel)
    if not gcode_path.is_absolute():
        gcode_path = (base_dir / gcode_path).resolve()

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(graph_cache).resolve() if graph_cache else (out_dir / "model.tgraph")
    offset = tuple(float(v) for v in cfg.get("offset", [0.0, 0.0, 0.0]))
    if cfg.get("build_axis", "Z") != "Z":
        raise ValueError("G-code toolpath graph mode requires build_axis='Z'")

    t0 = time.perf_counter()
    graph = None
    if cache_path.is_file():
        try:
            cand = load_toolpath_graph(cache_path, verify_checksum=True)
            if (
                cand.meta.get("source_gcode") == str(gcode_path)
                and abs(float(cand.meta.get("max_seg_len_mm", -1.0)) - max_seg_len) < 1e-9
            ):
                graph = cand
        except Exception:
            graph = None

    cache_hit = graph is not None
    if graph is None:
        graph = gcode_to_toolpath_graph(
            gcode_path,
            out_graph_path=cache_path,
            offset=offset,
            max_seg_len=max_seg_len,
            keep_largest_component=keep_largest_component,
        )

    material = Material(**cfg.get("material", {}))
    seam_cfg = seam_cfg or SeamConfig()
    result = solve_toolpath_graph_gpu(
        graph,
        material=material,
        seam_cfg=seam_cfg,
        fixed=cfg.get("fixed", "X-"),
        loaded=cfg.get("loaded", "X+"),
        force=cfg.get("force", [0.0, 0.0, -10.0]),
        load_point=cfg.get("load_point"),
        rtol=rtol,
        device=device,
    )

    spill_file = (out_dir / "graph_stress.bin") if spill_stress else None
    stress_summary = recover_graph_stress_streaming(
        graph,
        result["displacement_6dof"],
        material=material,
        seam_cfg=seam_cfg,
        spill_path=spill_file,
    )

    preview_path = out_dir / "preview.png"
    if render_preview:
        render_graph_preview(graph, result, stress_summary, preview_path)

    stress_json = {k: v for k, v in stress_summary.items() if not k.startswith("_")}
    summary = {
        "schema_version": "toolpath_graph_gpu_1.0",
        "case": str(case_path),
        "gcode": str(gcode_path),
        "graph_path": str(cache_path),
        "graph_cache_hit": cache_hit,
        "graph_file_mib": float(cache_path.stat().st_size / 1024**2),
        "graph": graph.meta,
        "solver": result["solver_info"],
        "stress": stress_json,
        "preview_png": str(preview_path) if render_preview else None,
        "end_to_end_seconds": time.perf_counter() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="Path to PathFEA case JSON")
    parser.add_argument("--max-seg-len", type=float, default=1.0)
    parser.add_argument("--output", default="output/toolpath_graph")
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--ideal", action="store_true", help="Set all seam/weld/start-stop factors to 1.0")
    parser.add_argument("--seam-factor", type=float, default=0.5)
    parser.add_argument("--scarf-factor", type=float, default=0.85)
    parser.add_argument("--start-factor", type=float, default=0.7)
    parser.add_argument("--stop-factor", type=float, default=0.6)
    parser.add_argument("--start-beam-factor", type=float, default=0.85)
    parser.add_argument("--stop-beam-factor", type=float, default=0.85)
    parser.add_argument("--lateral-factor", type=float, default=0.85)
    parser.add_argument("--interlayer-factor", type=float, default=0.85)
    parser.add_argument("--thermal-tau", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda:0", "cpu"])
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--spill-stress", action="store_true")
    parser.add_argument("--no-keep-largest", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args(argv)

    seam_cfg = (
        SeamConfig.ideal()
        if args.ideal
        else SeamConfig(
            seam_stiffness_factor=args.seam_factor,
            scarf_seam_stiffness_factor=args.scarf_factor,
            start_anchor_factor=args.start_factor,
            stop_anchor_factor=args.stop_factor,
            start_zone_beam_factor=args.start_beam_factor,
            stop_zone_beam_factor=args.stop_beam_factor,
            lateral_weld_factor=args.lateral_factor,
            interlayer_weld_factor=args.interlayer_factor,
            thermal_decay_tau_s=args.thermal_tau,
        )
    )
    summary = run_graph_case(
        args.case,
        max_seg_len=args.max_seg_len,
        output_dir=args.output,
        graph_cache=args.graph_cache,
        seam_cfg=seam_cfg,
        device=args.device,
        rtol=args.rtol,
        spill_stress=args.spill_stress,
        keep_largest_component=not args.no_keep_largest,
        render_preview=not args.no_preview,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
