from __future__ import annotations

import argparse
import time

import numpy as np

from hydroshear_demo_common import (
    build_case,
    contact_points_and_colors,
    select_frame,
    shear_segments,
    taxel_points_and_colors,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Viser HydroShear demo using the local tactile implementation.")
    parser.add_argument("--complex", action="store_true", help="Use a cube mesh object.")
    parser.add_argument("--channel", type=int, default=0, choices=(0, 1), help="Tangential shear channel to drive.")
    parser.add_argument("--normal_axis", type=int, default=0)
    parser.add_argument("--num_points", type=int, default=2048, help="Object surface sample count for complex mode.")
    parser.add_argument("--slide", type=float, default=0.004)
    parser.add_argument("--penetration", type=float, default=0.002)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def _require_viser():
    try:
        import viser
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install viser websockets>=13") from exc
    return viser


def _add_frame(server, frame: dict, *, channel: int, handles: list) -> None:
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass
    handles.clear()

    contact_points, contact_colors = contact_points_and_colors(frame)
    taxel_points, taxel_colors = taxel_points_and_colors(frame, channel=channel)
    starts, ends = shear_segments(frame)

    if "cube_vertices_w" in frame:
        handles.append(
            server.scene.add_mesh_simple(
                "/object",
                vertices=np.asarray(frame["cube_vertices_w"], dtype=np.float32),
                faces=np.asarray(frame["cube_faces"], dtype=np.uint32),
                color=(70, 110, 230),
                opacity=0.28,
            )
        )
    if contact_points.size:
        handles.append(
            server.scene.add_point_cloud(
                "/contact_points",
                points=contact_points.astype(np.float32),
                colors=contact_colors,
                point_size=0.0025,
            )
        )
    if taxel_points.size:
        handles.append(
            server.scene.add_point_cloud(
                "/taxel_shear",
                points=taxel_points.astype(np.float32),
                colors=taxel_colors,
                point_size=0.0018,
            )
        )
    if starts.size:
        segments = np.stack((starts, ends), axis=1).astype(np.float32)
        colors = np.tile(np.array([[255, 40, 30], [255, 40, 30]], dtype=np.uint8), (segments.shape[0], 1, 1))
        handles.append(
            server.scene.add_line_segments(
                "/surface_shear_vectors",
                points=segments,
                colors=colors,
                line_width=2.0,
            )
        )
    handles.append(
        server.scene.add_label(
            "/label",
            text=f"HydroShear step={frame['step']} phase={frame['phase']} channel={channel}",
            position=(0.0, 0.0, 0.035),
        )
    )


def main():
    args = parse_args()
    viser = _require_viser()
    case = build_case(
        complex_geometry=bool(args.complex),
        channel=int(args.channel),
        normal_axis=int(args.normal_axis),
        num_points=int(args.num_points),
        slide=float(args.slide),
        penetration=float(args.penetration),
    )

    server = viser.ViserServer(port=int(args.port))
    print(f"Open http://localhost:{int(args.port)}")
    handles: list = []

    if not args.animate:
        _add_frame(server, select_frame(case), channel=int(args.channel), handles=handles)
        while True:
            time.sleep(1.0)

    frames = case["frames"]
    dt = 1.0 / max(float(args.fps), 1.0)
    idx = 0
    while True:
        _add_frame(server, frames[idx], channel=int(args.channel), handles=handles)
        idx = (idx + 1) % len(frames)
        time.sleep(dt)


if __name__ == "__main__":
    main()
