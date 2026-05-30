from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".matplotlib"))

from tactile.surface import (  # noqa: E402
    ObjectSurfaceSampler,
    ObjectSurfaceSamplerCfg,
    first_visual_mesh_from_urdf,
    load_trimesh,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize HydroShear-style object surface samples.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", type=str, help="Path to an OBJ/STL mesh.")
    src.add_argument("--urdf", type=str, help="Path to a URDF; the first visual mesh is sampled.")
    parser.add_argument("--scale", type=float, nargs=3, default=None, help="Optional mesh scale for --mesh.")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal_stride", type=int, default=64)
    parser.add_argument("--normal_scale", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "output" / "surface_samples.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true", help="Show an interactive matplotlib window.")
    return parser.parse_args()


def load_source_mesh(args):
    if args.urdf:
        mesh_path, scale = first_visual_mesh_from_urdf(args.urdf)
        mesh = load_trimesh(mesh_path, scale=scale)
        label = f"{mesh_path} scale={scale}"
        return mesh, label

    mesh = load_trimesh(args.mesh, scale=tuple(args.scale) if args.scale is not None else None)
    label = f"{Path(args.mesh).resolve()} scale={args.scale or (1.0, 1.0, 1.0)}"
    return mesh, label


def set_axes_equal(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 1.0e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def main():
    args = parse_args()

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh, label = load_source_mesh(args)
    sampler = ObjectSurfaceSampler(
        ObjectSurfaceSamplerCfg(num_points=int(args.num_points), seed=int(args.seed), device="cpu")
    )
    samples = sampler.sample_trimesh(mesh)

    points = samples.points_o.cpu().numpy()
    normals = samples.normals_o.cpu().numpy()
    area = samples.area.cpu().numpy()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_count = faces.shape[0]
    if face_count > 8000:
        rng = np.random.default_rng(0)
        faces_vis = faces[rng.choice(face_count, size=8000, replace=False)]
    else:
        faces_vis = faces

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(vertices[faces_vis], alpha=0.12, linewidths=0.15)
    poly.set_facecolor((0.65, 0.7, 0.78, 0.12))
    poly.set_edgecolor((0.15, 0.18, 0.22, 0.2))
    ax.add_collection3d(poly)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=area, cmap="viridis", depthshade=False)

    stride = max(1, int(args.normal_stride))
    q_pts = points[::stride]
    q_normals = normals[::stride]
    ax.quiver(
        q_pts[:, 0],
        q_pts[:, 1],
        q_pts[:, 2],
        q_normals[:, 0],
        q_normals[:, 1],
        q_normals[:, 2],
        length=float(args.normal_scale),
        normalize=True,
        color="tab:red",
        linewidth=0.8,
    )

    set_axes_equal(ax, np.vstack((vertices, points)))
    ax.set_xlabel("x object")
    ax.set_ylabel("y object")
    ax.set_zlabel("z object")
    ax.set_title(
        "Object surface samples\n"
        f"M={samples.num_points}, total_area={float(samples.total_area):.6g}, "
        f"mean_A={float(samples.area.mean()):.6g}"
    )
    fig.text(0.02, 0.02, label, fontsize=8)
    fig.tight_layout()

    output = Path(os.path.expanduser(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"[INFO] saved {output}")
    print(f"[INFO] points_o shape={tuple(samples.points_o.shape)} normals_o shape={tuple(samples.normals_o.shape)}")
    print(f"[INFO] total_area={float(samples.total_area):.9g} mean_area={float(samples.area.mean()):.9g}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
