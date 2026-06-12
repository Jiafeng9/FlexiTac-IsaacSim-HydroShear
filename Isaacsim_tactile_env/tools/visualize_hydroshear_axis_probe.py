from __future__ import annotations

import argparse
import csv
import html
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".matplotlib"))

from tactile.backend import HydroShearTactileBackend, HydroShearTactileBackendCfg  # noqa: E402
from tactile.hydroshear import SurfacePointHydroShearCfg  # noqa: E402
from tactile.readout import (  # noqa: E402
    HydroShearMarkerReadoutCfg,
    SurfacePointForceProjectorCfg,
    TaxelGridCfg,
    tangential_axes,
)
from tactile.surface import ObjectSurfaceSamples  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a controlled HydroShear axis-separation visual proof without Isaac/robot control."
    )
    parser.add_argument("--normal_axis", type=int, default=0)
    parser.add_argument("--num_rows", type=int, default=12)
    parser.add_argument("--num_cols", type=int, default=32)
    parser.add_argument("--point_distance", type=float, default=0.002)
    parser.add_argument("--normal_offset", type=float, default=0.0)
    parser.add_argument("--surface_rows", type=int, default=30)
    parser.add_argument("--surface_cols", type=int, default=70)
    parser.add_argument("--clearance", type=float, default=0.001)
    parser.add_argument("--penetration", type=float, default=0.002)
    parser.add_argument("--slide", type=float, default=0.001)
    parser.add_argument("--lambda_s", type=float, default=10_800.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "output" / "hydroshear_axis_probe"),
        help="Directory for PNG/HTML/CSV outputs.",
    )
    parser.add_argument("--basename", type=str, default="hydroshear_axis_probe")
    return parser.parse_args()


def make_surface_patch(args) -> ObjectSurfaceSamples:
    normal_axis = int(args.normal_axis)
    axis_u, axis_v = tangential_axes(normal_axis)
    extent_u = float(args.point_distance) * float(args.num_rows) * 0.55
    extent_v = float(args.point_distance) * float(args.num_cols) * 0.55
    u = torch.linspace(-extent_u, extent_u, int(args.surface_rows), dtype=torch.float32)
    v = torch.linspace(-extent_v, extent_v, int(args.surface_cols), dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    points = torch.zeros((uu.numel(), 3), dtype=torch.float32)
    points[:, axis_u] = uu.reshape(-1)
    points[:, axis_v] = vv.reshape(-1)

    normals = torch.zeros_like(points)
    normals[:, normal_axis] = -1.0
    area = torch.full((points.shape[0],), (2.0 * extent_u * 2.0 * extent_v) / float(points.shape[0]), dtype=torch.float32)
    return ObjectSurfaceSamples(points_o=points, normals_o=normals, area=area)


def make_slab_mesh(normal_axis: int):
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    vertices = torch.zeros((8, 3), dtype=torch.float32)
    half_extent = 0.06
    back = -0.01
    front = 0.0
    values = [
        (-half_extent, -half_extent, back),
        (half_extent, -half_extent, back),
        (half_extent, half_extent, back),
        (-half_extent, half_extent, back),
        (-half_extent, -half_extent, front),
        (half_extent, -half_extent, front),
        (half_extent, half_extent, front),
        (-half_extent, half_extent, front),
    ]
    for i, (u, v, n) in enumerate(values):
        vertices[i, axes[0]] = u
        vertices[i, axes[1]] = v
        vertices[i, normal_axis] = n
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.long,
    )
    return vertices, faces


def make_backend(args) -> HydroShearTactileBackend:
    normal_axis = int(args.normal_axis)
    elastomer_vertices, elastomer_faces = make_slab_mesh(normal_axis)
    cfg = HydroShearTactileBackendCfg(
        grid=TaxelGridCfg(
            num_rows=int(args.num_rows),
            num_cols=int(args.num_cols),
            point_distance=float(args.point_distance),
            normal_axis=normal_axis,
            normal_offset=float(args.normal_offset),
        ),
        elastomer_vertices_p=elastomer_vertices,
        elastomer_faces=elastomer_faces,
        elastomer_sdf_object_name=f"axis_probe_slab_{normal_axis}",
        hydroshear=SurfacePointHydroShearCfg(
            normal_stiffness=1.0,
            shear_stiffness=1.0,
            friction_coefficient=float(args.mu),
            normal_axis=normal_axis,
        ),
        projection=SurfacePointForceProjectorCfg(
            lambda_s=float(args.lambda_s),
            shear_axis_signs=(1.0, 1.0),
        ),
        marker_projection=HydroShearMarkerReadoutCfg(
            lambda_s=float(args.lambda_s),
            lambda_d=0.0,
            shear_scale=1.0,
            dilation_scale=0.0,
            shear_axis_signs=(1.0, 1.0),
        ),
        output_mode="force_grid",
        output_key="tactile",
    )
    return HydroShearTactileBackend(cfg)


def run_case(args, channel: int):
    normal_axis = int(args.normal_axis)
    tangent = list(tangential_axes(normal_axis))
    move_axis = tangent[int(channel)]
    samples = make_surface_patch(args)
    backend = make_backend(args)
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    previous = torch.zeros(3, dtype=torch.float32)
    previous[normal_axis] = float(args.clearance)
    current = torch.zeros(3, dtype=torch.float32)
    current[normal_axis] = -float(args.penetration)
    moved = current.clone()
    moved[move_axis] = float(args.slide)

    backend.update(samples, object_pos_e=previous, object_quat_e=quat_identity)
    backend.update(samples, object_pos_e=current, object_quat_e=quat_identity)
    out = backend.update(samples, object_pos_e=moved, object_quat_e=quat_identity)

    shear_grid = out.observations["tactile_shear"].detach().cpu().numpy()
    marker_grid = out.observations["tactile_marker_shear"].detach().cpu().numpy()
    taxel_points = out.readout.taxel_positions_p.detach().cpu().numpy().reshape(int(args.num_rows), int(args.num_cols), 3)
    contact_points = out.contact.points_p.detach().cpu().numpy()
    contact_mask = out.contact.contact_mask.detach().cpu().numpy()
    penetration = out.contact.penetration.detach().cpu().numpy()
    surface_shear_e = out.surface.shear_force_e.detach().cpu().numpy()

    intended = float(np.mean(np.abs(shear_grid[..., channel])))
    leakage = float(np.mean(np.abs(shear_grid[..., 1 - channel])))
    marker_intended = float(np.mean(np.abs(marker_grid[..., channel + 1])))
    marker_leakage = float(np.mean(np.abs(marker_grid[..., (1 - channel) + 1])))
    metrics = {
        "case": "pure_shear_x" if channel == 0 else "pure_shear_y",
        "normal_axis": normal_axis,
        "move_axis": move_axis,
        "target_delta_shear_x": float(args.slide) if channel == 0 else 0.0,
        "target_delta_shear_y": 0.0 if channel == 0 else float(args.slide),
        "force_mean_abs_shear_x": float(np.mean(np.abs(shear_grid[..., 0]))),
        "force_mean_abs_shear_y": float(np.mean(np.abs(shear_grid[..., 1]))),
        "force_leakage_ratio": leakage / max(intended, 1.0e-30),
        "marker_mean_abs_shear_x": float(np.mean(np.abs(marker_grid[..., 1]))),
        "marker_mean_abs_shear_y": float(np.mean(np.abs(marker_grid[..., 2]))),
        "marker_leakage_ratio": marker_leakage / max(marker_intended, 1.0e-30),
        "contact_points": int(contact_mask.sum()),
        "max_penetration": float(np.max(penetration)),
    }
    return {
        "channel": channel,
        "tangent": tangent,
        "move_axis": move_axis,
        "shear_grid": shear_grid,
        "marker_grid": marker_grid,
        "taxel_points": taxel_points,
        "contact_points": contact_points,
        "contact_mask": contact_mask,
        "penetration": penetration,
        "surface_shear_e": surface_shear_e,
        "metrics": metrics,
    }


def _extent(taxel_points: np.ndarray, axis_u: int, axis_v: int):
    return [
        float(taxel_points[:, :, axis_v].min()),
        float(taxel_points[:, :, axis_v].max()),
        float(taxel_points[:, :, axis_u].min()),
        float(taxel_points[:, :, axis_u].max()),
    ]


def draw_report(args, cases, png_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal_axis = int(args.normal_axis)
    axis_u, axis_v = tangential_axes(normal_axis)
    signed_vmax = max(float(np.max(np.abs(case["shear_grid"]))) for case in cases)
    signed_vmax = max(signed_vmax, 1.0e-12)

    fig, axs = plt.subplots(2, 4, figsize=(18, 8.5), constrained_layout=True)
    cmap_signed = "coolwarm"
    cmap_pen = "magma"
    last_im = None

    for row, case in enumerate(cases):
        channel = int(case["channel"])
        title = "Pure pad shear_x input" if channel == 0 else "Pure pad shear_y input"
        taxel_points = case["taxel_points"]
        contact = case["contact_mask"]
        points = case["contact_points"]
        penetration = case["penetration"]
        shear_surface = case["surface_shear_e"]
        shear_grid = case["shear_grid"]
        ext = _extent(taxel_points, axis_u, axis_v)
        m = case["metrics"]

        ax = axs[row, 0]
        ax.set_title(
            f"{title}\n"
            f"target=({m['target_delta_shear_x']:.0e}, {m['target_delta_shear_y']:.0e}), "
            f"leak={m['force_leakage_ratio']:.0e}"
        )
        sc = ax.scatter(
            points[contact, axis_v],
            points[contact, axis_u],
            s=4,
            c=penetration[contact],
            cmap=cmap_pen,
            alpha=0.8,
        )
        center_v = 0.0
        center_u = 0.0
        arrow_u = float(args.slide) if channel == 0 else 0.0
        arrow_v = 0.0 if channel == 0 else float(args.slide)
        ax.arrow(
            center_v,
            center_u,
            arrow_v,
            arrow_u,
            color="tab:cyan",
            width=float(args.point_distance) * 0.15,
            head_width=float(args.point_distance) * 1.6,
            length_includes_head=True,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(f"pad axis {axis_v} / shear_y direction")
        ax.set_ylabel(f"pad axis {axis_u} / shear_x direction")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])

        ax = axs[row, 1]
        ax.set_title("surface-point shear vectors\nbefore taxel projection")
        surf_mag = np.linalg.norm(shear_surface, axis=-1)
        ids = np.flatnonzero(contact & (surf_mag > 1.0e-12))
        stride = max(1, ids.size // 180)
        ids = ids[::stride]
        ax.scatter(points[contact, axis_v], points[contact, axis_u], s=2, color="0.82", alpha=0.45)
        if ids.size:
            vec = shear_surface[ids]
            scale = float(args.slide) / max(float(np.max(np.linalg.norm(vec[:, [axis_u, axis_v]], axis=-1))), 1.0e-12)
            ax.quiver(
                points[ids, axis_v],
                points[ids, axis_u],
                vec[:, axis_v] * scale,
                vec[:, axis_u] * scale,
                color="tab:red",
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
            )
        ax.set_aspect("equal")
        ax.set_xlabel(f"pad axis {axis_v}")
        ax.set_ylabel(f"pad axis {axis_u}")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])

        ax = axs[row, 2]
        last_im = ax.imshow(
            shear_grid[..., 0],
            origin="lower",
            extent=ext,
            aspect="equal",
            cmap=cmap_signed,
            vmin=-signed_vmax,
            vmax=signed_vmax,
            interpolation="nearest",
        )
        ax.set_title(
            "taxel force_shear_x\n"
            f"mean_abs={m['force_mean_abs_shear_x']:.3e}"
        )
        ax.set_xlabel(f"pad axis {axis_v}")
        ax.set_ylabel(f"pad axis {axis_u}")

        ax = axs[row, 3]
        last_im = ax.imshow(
            shear_grid[..., 1],
            origin="lower",
            extent=ext,
            aspect="equal",
            cmap=cmap_signed,
            vmin=-signed_vmax,
            vmax=signed_vmax,
            interpolation="nearest",
        )
        ax.set_title(
            "taxel force_shear_y\n"
            f"mean_abs={m['force_mean_abs_shear_y']:.3e}"
        )
        ax.set_xlabel(f"pad axis {axis_v}")
        ax.set_ylabel(f"pad axis {axis_u}")

    if last_im is not None:
        fig.colorbar(last_im, ax=axs[:, 2:4], shrink=0.9, label="signed projected shear force")
    fig.suptitle(
        "HydroShear controlled axis probe: input -> surface state -> taxel shear channels\n"
        "The two heatmap columns use the same fixed color scale.",
        fontsize=14,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def write_metrics(metrics_path: Path, cases) -> None:
    rows = [case["metrics"] for case in cases]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(html_path: Path, png_path: Path, metrics_path: Path, cases) -> None:
    rows = [case["metrics"] for case in cases]
    headers = list(rows[0].keys())
    table_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row[h]))}</td>" for h in headers)
        table_rows.append(f"<tr>{cells}</tr>")
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HydroShear Axis Probe</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; }}
    table {{ border-collapse: collapse; font-size: 12px; margin-top: 16px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 4px 6px; text-align: right; }}
    th {{ background: #f6f8fa; }}
    td:first-child, th:first-child {{ text-align: left; }}
    code {{ background: #f6f8fa; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>HydroShear Axis Probe</h1>
  <p>
    This report is generated without Isaac or robot IK. It applies controlled pad-frame pure
    <code>shear_x</code> and pure <code>shear_y</code> motions to a flat contact patch, then shows the
    connected path from input motion to surface-point shear and projected taxel heatmaps.
  </p>
  <p>Metrics CSV: <code>{html.escape(metrics_path.name)}</code></p>
  <img src="{html.escape(png_path.name)}" alt="HydroShear axis probe figure">
  <table>
    <thead><tr>{header_html}</tr></thead>
    <tbody>
      {''.join(table_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(html_text)


def main():
    args = parse_args()
    if int(args.normal_axis) not in (0, 1, 2):
        raise ValueError("--normal_axis must be 0, 1, or 2")

    cases = [run_case(args, 0), run_case(args, 1)]
    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.basename}.png"
    html_path = output_dir / f"{args.basename}.html"
    metrics_path = output_dir / f"{args.basename}_metrics.csv"

    draw_report(args, cases, png_path)
    write_metrics(metrics_path, cases)
    write_html(html_path, png_path, metrics_path, cases)

    print(f"[INFO] saved PNG: {png_path}")
    print(f"[INFO] saved HTML: {html_path}")
    print(f"[INFO] saved metrics: {metrics_path}")
    for case in cases:
        m = case["metrics"]
        print(
            f"[INFO] {m['case']}: "
            f"force_mean_abs=({m['force_mean_abs_shear_x']:.6e}, {m['force_mean_abs_shear_y']:.6e}) "
            f"force_leakage_ratio={m['force_leakage_ratio']:.6e} "
            f"marker_leakage_ratio={m['marker_leakage_ratio']:.6e}"
        )


if __name__ == "__main__":
    main()
