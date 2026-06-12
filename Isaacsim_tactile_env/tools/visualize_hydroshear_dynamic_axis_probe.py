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
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".matplotlib"))

from tactile.readout import tangential_axes  # noqa: E402
from visualize_hydroshear_axis_probe import make_backend, make_surface_patch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dynamic controlled HydroShear axis probe without Isaac/robot IK."
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
    parser.add_argument("--slide", type=float, default=0.004)
    parser.add_argument("--press_steps", type=int, default=12)
    parser.add_argument("--slide_steps", type=int, default=48)
    parser.add_argument("--hold_steps", type=int, default=12)
    parser.add_argument("--gif_stride", type=int, default=2)
    parser.add_argument("--lambda_s", type=float, default=10_800.0)
    parser.add_argument("--mu", type=float, default=10.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "output" / "hydroshear_dynamic_axis_probe"),
    )
    parser.add_argument("--basename", type=str, default="hydroshear_dynamic_axis_probe")
    return parser.parse_args()


def _case_name(channel: int) -> str:
    return "pure_shear_x" if int(channel) == 0 else "pure_shear_y"


def _frame_metrics(case: dict, frame: dict) -> dict:
    shear_grid = frame["shear_grid"]
    marker_grid = frame["marker_grid"]
    channel = int(case["channel"])
    other = 1 - channel
    force_intended = float(np.mean(np.abs(shear_grid[..., channel])))
    force_other = float(np.mean(np.abs(shear_grid[..., other])))
    marker_intended = float(np.mean(np.abs(marker_grid[..., channel + 1])))
    marker_other = float(np.mean(np.abs(marker_grid[..., other + 1])))
    return {
        "case": _case_name(channel),
        "step": int(frame["step"]),
        "phase": frame["phase"],
        "target_shear_x": float(frame["target_shear"][0]),
        "target_shear_y": float(frame["target_shear"][1]),
        "target_delta_shear_x": float(frame["target_delta_shear"][0]),
        "target_delta_shear_y": float(frame["target_delta_shear"][1]),
        "force_mean_abs_shear_x": float(np.mean(np.abs(shear_grid[..., 0]))),
        "force_mean_abs_shear_y": float(np.mean(np.abs(shear_grid[..., 1]))),
        "force_leakage_ratio": force_other / max(force_intended, 1.0e-30),
        "marker_mean_abs_shear_x": float(np.mean(np.abs(marker_grid[..., 1]))),
        "marker_mean_abs_shear_y": float(np.mean(np.abs(marker_grid[..., 2]))),
        "marker_leakage_ratio": marker_other / max(marker_intended, 1.0e-30),
        "contact_points": int(frame["contact_mask"].sum()),
        "max_penetration": float(np.max(frame["penetration"])),
    }


def _collect_frame(args, backend, samples, pos, prev_pos, step: int, phase: str, channel: int) -> dict:
    normal_axis = int(args.normal_axis)
    axis_u, axis_v = tangential_axes(normal_axis)
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    out = backend.update(samples, object_pos_e=pos, object_quat_e=quat_identity)
    target_shear = np.array([float(pos[axis_u]), float(pos[axis_v])], dtype=np.float64)
    target_delta = np.array(
        [float(pos[axis_u] - prev_pos[axis_u]), float(pos[axis_v] - prev_pos[axis_v])],
        dtype=np.float64,
    )
    return {
        "step": step,
        "phase": phase,
        "channel": channel,
        "target_shear": target_shear,
        "target_delta_shear": target_delta,
        "shear_grid": out.observations["tactile_shear"].detach().cpu().numpy(),
        "marker_grid": out.observations["tactile_marker_shear"].detach().cpu().numpy(),
        "taxel_points": out.readout.taxel_positions_p.detach().cpu().numpy().reshape(
            int(args.num_rows), int(args.num_cols), 3
        ),
        "contact_points": out.contact.points_p.detach().cpu().numpy(),
        "contact_mask": out.contact.contact_mask.detach().cpu().numpy(),
        "penetration": out.contact.penetration.detach().cpu().numpy(),
        "surface_shear_e": out.surface.shear_force_e.detach().cpu().numpy(),
    }


def run_dynamic_case(args, channel: int) -> dict:
    normal_axis = int(args.normal_axis)
    tangent_axes = tangential_axes(normal_axis)
    move_axis = tangent_axes[int(channel)]
    samples = make_surface_patch(args)
    backend = make_backend(args)

    frames = []
    step = 0
    prev_pos = torch.zeros(3, dtype=torch.float32)
    prev_pos[normal_axis] = float(args.clearance)

    for i in range(max(1, int(args.press_steps))):
        alpha = float(i + 1) / float(max(1, int(args.press_steps)))
        pos = torch.zeros(3, dtype=torch.float32)
        pos[normal_axis] = (1.0 - alpha) * float(args.clearance) + alpha * (-float(args.penetration))
        frames.append(_collect_frame(args, backend, samples, pos, prev_pos, step, "press", channel))
        prev_pos = pos
        step += 1

    for i in range(max(1, int(args.slide_steps))):
        alpha = float(i + 1) / float(max(1, int(args.slide_steps)))
        pos = torch.zeros(3, dtype=torch.float32)
        pos[normal_axis] = -float(args.penetration)
        pos[move_axis] = alpha * float(args.slide)
        frames.append(_collect_frame(args, backend, samples, pos, prev_pos, step, "slide", channel))
        prev_pos = pos
        step += 1

    for _ in range(max(0, int(args.hold_steps))):
        pos = prev_pos.clone()
        frames.append(_collect_frame(args, backend, samples, pos, prev_pos, step, "hold", channel))
        prev_pos = pos
        step += 1

    case = {
        "channel": int(channel),
        "move_axis": int(move_axis),
        "frames": frames,
    }
    case["metrics"] = [_frame_metrics(case, frame) for frame in frames]
    return case


def _axis_extent(taxel_points: np.ndarray, axis_u: int, axis_v: int):
    return [
        float(taxel_points[:, :, axis_v].min()),
        float(taxel_points[:, :, axis_v].max()),
        float(taxel_points[:, :, axis_u].min()),
        float(taxel_points[:, :, axis_u].max()),
    ]


def _fig_to_image(fig):
    from PIL import Image

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3].copy())


def make_case_gif(args, case: dict, path: Path, signed_vmax: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal_axis = int(args.normal_axis)
    axis_u, axis_v = tangential_axes(normal_axis)
    stride = max(1, int(args.gif_stride))
    images = []

    for frame, metric in zip(case["frames"][::stride], case["metrics"][::stride]):
        taxel_points = frame["taxel_points"]
        contact = frame["contact_mask"]
        points = frame["contact_points"]
        penetration = frame["penetration"]
        shear_surface = frame["surface_shear_e"]
        shear_grid = frame["shear_grid"]
        ext = _axis_extent(taxel_points, axis_u, axis_v)

        fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.4), constrained_layout=True)
        title = f"{_case_name(case['channel'])} step={frame['step']} phase={frame['phase']}"
        fig.suptitle(title)

        ax = axs[0]
        ax.set_title(
            "target on pad plane\n"
            f"pos=({metric['target_shear_x']:.3e}, {metric['target_shear_y']:.3e})"
        )
        ax.scatter(points[contact, axis_v], points[contact, axis_u], s=3, c=penetration[contact], cmap="magma")
        ax.scatter([frame["target_shear"][1]], [frame["target_shear"][0]], s=80, marker=">", color="tab:cyan")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel(f"pad axis {axis_v}")
        ax.set_ylabel(f"pad axis {axis_u}")

        ax = axs[1]
        ax.set_title("surface shear vectors")
        surf_mag = np.linalg.norm(shear_surface, axis=-1)
        ids = np.flatnonzero(contact & (surf_mag > 1.0e-12))
        if ids.size:
            ids = ids[:: max(1, ids.size // 120)]
            vec = shear_surface[ids]
            scale = float(args.point_distance) * 2.5 / max(
                float(np.max(np.linalg.norm(vec[:, [axis_u, axis_v]], axis=-1))),
                1.0e-12,
            )
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
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect("equal")
        ax.set_xlabel(f"pad axis {axis_v}")
        ax.set_ylabel(f"pad axis {axis_u}")

        for idx, channel_name in enumerate(("x", "y")):
            ax = axs[idx + 2]
            ax.imshow(
                shear_grid[..., idx],
                origin="lower",
                extent=ext,
                aspect="equal",
                cmap="coolwarm",
                vmin=-signed_vmax,
                vmax=signed_vmax,
                interpolation="nearest",
            )
            ax.set_title(f"taxel shear_{channel_name}\nmean_abs={metric[f'force_mean_abs_shear_{channel_name}']:.2e}")
            ax.set_xlabel(f"pad axis {axis_v}")
            ax.set_ylabel(f"pad axis {axis_u}")

        images.append(_fig_to_image(fig))
        plt.close(fig)

    path.parent.mkdir(parents=True, exist_ok=True)
    if images:
        images[0].save(path, save_all=True, append_images=images[1:], duration=140, loop=0)


def make_timeseries(args, cases: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for ax, case in zip(axs, cases):
        m = case["metrics"]
        steps = np.array([row["step"] for row in m], dtype=float)
        sx = np.array([row["force_mean_abs_shear_x"] for row in m], dtype=float)
        sy = np.array([row["force_mean_abs_shear_y"] for row in m], dtype=float)
        tx = np.array([row["target_shear_x"] for row in m], dtype=float)
        ty = np.array([row["target_shear_y"] for row in m], dtype=float)
        ax.plot(steps, sx, label="force_shear_x mean_abs", linewidth=2)
        ax.plot(steps, sy, label="force_shear_y mean_abs", linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(steps, tx, "--", color="tab:blue", alpha=0.35, label="target shear_x")
        ax2.plot(steps, ty, "--", color="tab:orange", alpha=0.35, label="target shear_y")
        ax.set_title(_case_name(case["channel"]))
        ax.set_ylabel("mean_abs projected shear")
        ax2.set_ylabel("target position in pad shear axes")
        ax.grid(True, alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)
    axs[-1].set_xlabel("dynamic step")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_metrics(path: Path, cases: list[dict]) -> None:
    rows = []
    for case in cases:
        rows.extend(case["metrics"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, gif_paths: list[Path], timeseries_path: Path, metrics_path: Path, cases: list[dict]) -> None:
    rows = []
    for case in cases:
        slide_rows = [m for m in case["metrics"] if m["phase"] == "slide"]
        final = slide_rows[-1] if slide_rows else case["metrics"][-1]
        rows.append(final)
    headers = [
        "case",
        "force_mean_abs_shear_x",
        "force_mean_abs_shear_y",
        "force_leakage_ratio",
        "marker_leakage_ratio",
        "contact_points",
        "max_penetration",
    ]
    header_html = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_html = ""
    for row in rows:
        body_html += "<tr>" + "".join(f"<td>{html.escape(str(row[h]))}</td>" for h in headers) + "</tr>"

    gif_html = "\n".join(
        f'<section><h2>{html.escape(path.stem)}</h2><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></section>'
        for path in gif_paths
    )
    text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dynamic HydroShear Axis Probe</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    img {{ max-width: 100%; border: 1px solid #d0d7de; margin-bottom: 12px; }}
    table {{ border-collapse: collapse; font-size: 12px; margin: 16px 0; }}
    th, td {{ border: 1px solid #d0d7de; padding: 4px 6px; text-align: right; }}
    th {{ background: #f6f8fa; }}
    td:first-child, th:first-child {{ text-align: left; }}
    code {{ background: #f6f8fa; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Dynamic HydroShear Axis Probe</h1>
  <p>
    This runs a controlled sequence: press into the tactile pad, slide along one pad-frame shear axis
    for many frames, then hold. It does not use robot IK or PhysX, so it isolates the HydroShear
    state update and taxel projection.
  </p>
  <p>Metrics CSV: <code>{html.escape(metrics_path.name)}</code></p>
  <table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>
  <h2>Time Series</h2>
  <img src="{html.escape(timeseries_path.name)}" alt="dynamic time series">
  {gif_html}
</body>
</html>
"""
    path.write_text(text)


def main():
    args = parse_args()
    cases = [run_dynamic_case(args, 0), run_dynamic_case(args, 1)]
    signed_vmax = max(float(np.max(np.abs(frame["shear_grid"]))) for case in cases for frame in case["frames"])
    signed_vmax = max(signed_vmax, 1.0e-12)

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_paths = [
        output_dir / f"{args.basename}_{_case_name(case['channel'])}.gif"
        for case in cases
    ]
    timeseries_path = output_dir / f"{args.basename}_timeseries.png"
    metrics_path = output_dir / f"{args.basename}_metrics.csv"
    html_path = output_dir / f"{args.basename}.html"

    for case, gif_path in zip(cases, gif_paths):
        make_case_gif(args, case, gif_path, signed_vmax)
    make_timeseries(args, cases, timeseries_path)
    write_metrics(metrics_path, cases)
    write_html(html_path, gif_paths, timeseries_path, metrics_path, cases)

    print(f"[INFO] saved HTML: {html_path}")
    print(f"[INFO] saved timeseries: {timeseries_path}")
    for path in gif_paths:
        print(f"[INFO] saved GIF: {path}")
    print(f"[INFO] saved metrics: {metrics_path}")
    for case in cases:
        slide_rows = [m for m in case["metrics"] if m["phase"] == "slide"]
        final = slide_rows[-1] if slide_rows else case["metrics"][-1]
        print(
            f"[INFO] {_case_name(case['channel'])}: "
            f"final_slide_force_mean_abs=({final['force_mean_abs_shear_x']:.6e}, "
            f"{final['force_mean_abs_shear_y']:.6e}) "
            f"force_leakage_ratio={final['force_leakage_ratio']:.6e}"
        )


if __name__ == "__main__":
    main()
