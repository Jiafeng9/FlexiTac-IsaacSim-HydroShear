from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output" / ".matplotlib"))

from tactile.contact import SurfacePointContactQuery  # noqa: E402
from tactile.elastomer import FlatPatchElastomerSdf, FlatPatchElastomerSdfCfg  # noqa: E402
from tactile.hydroshear import SurfacePointHydroShearCfg, SurfacePointHydroShearTracker  # noqa: E402
from tactile.readout import (  # noqa: E402
    ProjectedSurfacePointTracker,
    ProjectedSurfacePointTrackerCfg,
    SurfacePointForceProjector,
    SurfacePointForceProjectorCfg,
    TaxelGridCfg,
    tangential_axes,
)
from tactile.surface import ObjectSurfaceSampler, ObjectSurfaceSamplerCfg, first_visual_mesh_from_urdf, load_trimesh  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize one HydroShear surface-point update on a flat elastomer.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", type=str, help="Path to an OBJ/STL mesh.")
    src.add_argument("--urdf", type=str, help="Path to a URDF; the first visual mesh is sampled.")
    parser.add_argument("--scale", type=float, nargs=3, default=None, help="Optional mesh scale for --mesh.")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal_axis", type=int, default=2)
    parser.add_argument("--prev_clearance", type=float, default=0.002)
    parser.add_argument("--curr_penetration", type=float, default=0.003)
    parser.add_argument("--tangent_dx", type=float, default=0.003)
    parser.add_argument("--normal_stiffness", type=float, default=1.0)
    parser.add_argument("--shear_stiffness", type=float, default=1.0)
    parser.add_argument("--area_mode", choices=("unit", "mesh_area"), default="unit")
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--force_stride", type=int, default=64)
    parser.add_argument("--force_scale", type=float, default=50.0)
    parser.add_argument("--num_rows", type=int, default=12)
    parser.add_argument("--num_cols", type=int, default=32)
    parser.add_argument("--point_distance", type=float, default=0.002)
    parser.add_argument("--normal_offset", type=float, default=0.0)
    parser.add_argument("--lambda_s", type=float, default=10_800.0)
    parser.add_argument("--lambda_d", type=float, default=0.0)
    parser.add_argument("--projected_displacement_decay", type=float, default=0.0)
    parser.add_argument("--projected_displacement_max", type=float, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "output" / "hydroshear_debug.png"),
        help="Output PNG path.",
    )
    parser.add_argument("--show", action="store_true", help="Show an interactive matplotlib window.")
    return parser.parse_args()


def load_source_mesh(args):
    if args.urdf:
        mesh_path, scale = first_visual_mesh_from_urdf(args.urdf)
        mesh = load_trimesh(mesh_path, scale=scale)
        return mesh, f"{mesh_path} scale={scale}"

    scale = tuple(args.scale) if args.scale is not None else None
    mesh = load_trimesh(args.mesh, scale=scale)
    return mesh, f"{Path(args.mesh).resolve()} scale={scale or (1.0, 1.0, 1.0)}"


def set_axes_equal(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1.0e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def pose_for_flat_contact(samples_points: torch.Tensor, normal_axis: int, clearance: float, penetration: float, dx: float):
    min_coord = float(samples_points[:, normal_axis].min())
    prev = torch.zeros(3, dtype=samples_points.dtype)
    curr = torch.zeros(3, dtype=samples_points.dtype)
    prev[normal_axis] = -min_coord + float(clearance)
    curr[normal_axis] = -min_coord - float(penetration)
    tangent_axis = 0 if normal_axis != 0 else 1
    curr[tangent_axis] = float(dx)
    return prev, curr


def _projected_surface_points(
    args,
    surface_points_p: torch.Tensor,
    displacement_e: torch.Tensor,
    contact_mask: torch.Tensor,
):
    if float(args.lambda_d) == 0.0:
        return None
    tracker = ProjectedSurfacePointTracker(
        normal_axis=int(args.normal_axis),
        cfg=ProjectedSurfacePointTrackerCfg(
            lambda_d=float(args.lambda_d),
            decay=float(args.projected_displacement_decay),
            max_displacement=args.projected_displacement_max,
        ),
    )
    return tracker.update(
        surface_points_p=surface_points_p,
        displacement_e=displacement_e,
        contact_mask=contact_mask,
    ).projected_points_p


def main():
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh, label = load_source_mesh(args)
    sampler = ObjectSurfaceSampler(ObjectSurfaceSamplerCfg(num_points=args.num_points, seed=args.seed, device="cpu"))
    samples = sampler.sample_trimesh(mesh)

    prev_pos, curr_pos = pose_for_flat_contact(
        samples.points_o,
        int(args.normal_axis),
        clearance=float(args.prev_clearance),
        penetration=float(args.curr_penetration),
        dx=float(args.tangent_dx),
    )
    quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=samples.points_o.dtype)

    query = SurfacePointContactQuery(FlatPatchElastomerSdf(FlatPatchElastomerSdfCfg(normal_axis=args.normal_axis)))
    tracker = SurfacePointHydroShearTracker(
        SurfacePointHydroShearCfg(
            normal_stiffness=float(args.normal_stiffness),
            shear_stiffness=float(args.shear_stiffness),
            friction_coefficient=float(args.mu),
            normal_axis=int(args.normal_axis),
            area_mode=str(args.area_mode),
        )
    )
    prev_contact = query.compute(samples, object_pos_e=prev_pos, object_quat_e=quat_identity)
    tracker.update(samples, prev_contact)
    curr_contact = query.compute(samples, object_pos_e=curr_pos, object_quat_e=quat_identity)
    out = tracker.update(samples, curr_contact)

    projector = SurfacePointForceProjector(
        TaxelGridCfg(
            num_rows=int(args.num_rows),
            num_cols=int(args.num_cols),
            point_distance=float(args.point_distance),
            normal_axis=int(args.normal_axis),
            normal_offset=float(args.normal_offset),
        ),
        SurfacePointForceProjectorCfg(lambda_s=float(args.lambda_s)),
    )
    readout = projector.project(
        surface_points_p=curr_contact.points_p,
        penetration=out.penetration,
        normal_force=out.normal_force,
        shear_force_e=out.shear_force_e,
        projected_surface_points_p=_projected_surface_points(args, curr_contact.points_p, out.displacement_e, out.contact_mask),
    )

    points = curr_contact.points_e.cpu().numpy()
    penetration = curr_contact.penetration.cpu().numpy()
    contact = curr_contact.contact_mask.cpu().numpy()
    force = out.force_e.detach().cpu().numpy()
    shear = out.shear_force_e.detach().cpu().numpy()
    taxel_points = readout.taxel_positions_p.detach().cpu().numpy().reshape(args.num_rows, args.num_cols, 3)
    taxel_normal = readout.normal_force.detach().cpu().numpy()
    taxel_shear = readout.shear_force_uv.detach().cpu().numpy()

    vertices = np.asarray(mesh.vertices, dtype=np.float64) + curr_pos.cpu().numpy()
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_count = faces.shape[0]
    if face_count > 8000:
        rng = np.random.default_rng(0)
        faces_vis = faces[rng.choice(face_count, size=8000, replace=False)]
    else:
        faces_vis = faces

    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    ax = fig.add_subplot(121, projection="3d")

    poly = Poly3DCollection(vertices[faces_vis], alpha=0.10, linewidths=0.12)
    poly.set_facecolor((0.65, 0.70, 0.78, 0.10))
    poly.set_edgecolor((0.15, 0.18, 0.22, 0.18))
    ax.add_collection3d(poly)

    ax.scatter(points[~contact, 0], points[~contact, 1], points[~contact, 2], s=4, color="0.75", depthshade=False)
    sc = ax.scatter(
        points[contact, 0],
        points[contact, 1],
        points[contact, 2],
        s=8,
        c=penetration[contact],
        cmap="magma",
        depthshade=False,
    )
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08, label="penetration")

    force_norm = np.linalg.norm(force, axis=1)
    force_ids = np.flatnonzero(contact & (force_norm > 1.0e-12))[:: max(1, int(args.force_stride))]
    if force_ids.size > 0:
        dirs = force[force_ids] * float(args.force_scale)
        ax.quiver(
            points[force_ids, 0],
            points[force_ids, 1],
            points[force_ids, 2],
            dirs[:, 0],
            dirs[:, 1],
            dirs[:, 2],
            color="tab:red",
            linewidth=0.9,
            normalize=False,
        )

    shear_norm = np.linalg.norm(shear, axis=1)
    slip_ratio = shear_norm / (float(args.mu) * out.normal_force.detach().cpu().numpy() + 1.0e-8)

    set_axes_equal(ax, np.vstack((vertices, points)))
    ax.set_xlabel("x elastomer")
    ax.set_ylabel("y elastomer")
    ax.set_zlabel("z elastomer")
    ax.set_title(
        "Surface-point state\n"
        f"contact={int(contact.sum())}/{contact.size}, "
        f"max_pen={penetration.max():.4g}, max|f|={force_norm.max():.4g}"
    )

    ax_grid = fig.add_subplot(122)
    axis_u, axis_v = tangential_axes(int(args.normal_axis))
    grid_u = taxel_points[:, :, axis_u]
    grid_v = taxel_points[:, :, axis_v]
    extent = [grid_v.min(), grid_v.max(), grid_u.min(), grid_u.max()]
    im = ax_grid.imshow(
        taxel_normal,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="magma",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax_grid, shrink=0.75, pad=0.04, label="projected fn")

    shear_norm_grid = np.linalg.norm(taxel_shear, axis=-1)
    if shear_norm_grid.max() > 1.0e-12:
        display_shear = taxel_shear / shear_norm_grid.max() * float(args.point_distance) * 2.0
        ax_grid.quiver(
            grid_v,
            grid_u,
            display_shear[:, :, 1],
            display_shear[:, :, 0],
            color="white",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0025,
        )
    ax_grid.set_xlabel(f"patch axis {axis_v}")
    ax_grid.set_ylabel(f"patch axis {axis_u}")
    ax_grid.set_title(
        "Projected taxel force grid\n"
        f"max_fn={taxel_normal.max():.4g}, max_ft={shear_norm_grid.max():.4g}, "
        f"lambda_s={args.lambda_s:.3g}, lambda_d={args.lambda_d:.3g}"
    )
    fig.text(0.02, 0.02, label, fontsize=8)

    output = Path(os.path.expanduser(args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"[INFO] saved {output}")
    print(f"[INFO] contact={int(contact.sum())}/{contact.size}")
    print(f"[INFO] max_penetration={float(penetration.max()):.9g}")
    print(f"[INFO] max_normal_force={float(out.normal_force.max()):.9g}")
    print(f"[INFO] max_shear_force={float(shear_norm.max()):.9g}")
    print(f"[INFO] max_projected_taxel_normal={float(taxel_normal.max()):.9g}")
    print(f"[INFO] max_projected_taxel_shear={float(shear_norm_grid.max()):.9g}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
