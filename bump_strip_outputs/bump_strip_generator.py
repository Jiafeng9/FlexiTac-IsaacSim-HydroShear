#!/usr/bin/env python3
"""
Parametric bump strip STL generator.

Defaults are based on the uploaded 16 x 16 TPU bump-strip STL:
- pitch: 8.75 mm
- bump base/footprint diameter: 8.00 mm
- bump top flat diameter: 5.30 mm
- bump height above base top: 3.73 mm
- perimeter/rim thickness: 3.00 mm
- main sheet thickness: 1.00 mm
- circular tessellation: 126 segments

Requires: numpy, shapely, trimesh, plus triangle or mapbox_earcut for the most robust triangulation
Example:
  python bump_strip_generator.py --rows 4 --cols 8 --variant soft --output bump_4x8_soft.stl
  python bump_strip_generator.py --rows 4 --cols 8 --variant soft --relief-grooves --output bump_4x8_soft_grooved.stl
  python bump_strip_generator.py --mode mold --rows 4 --cols 8 --output mold_4x8.stl
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon, box, MultiPolygon, GeometryCollection
from shapely.ops import triangulate, unary_union

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]

TRIANGULATION_ENGINES: Tuple[str, ...] = ("earcut", "triangle")


def set_triangulation_engine(engine: str) -> None:
    """Choose preferred constrained triangulation backend."""
    global TRIANGULATION_ENGINES
    if engine == "triangle":
        TRIANGULATION_ENGINES = ("triangle", "earcut")
    elif engine == "earcut":
        TRIANGULATION_ENGINES = ("earcut", "triangle")
    elif engine == "auto":
        TRIANGULATION_ENGINES = ("earcut", "triangle")
    else:
        raise ValueError(f"Unknown triangulation engine: {engine}")


@dataclass(frozen=True)
class HollowPreset:
    hollow: bool
    side_wall: float
    top_skin: float


PRESETS = {
    "solid": HollowPreset(False, 0.0, 0.0),
    "medium": HollowPreset(True, 1.6, 1.2),
    "soft": HollowPreset(True, 1.2, 1.0),
    "extra-soft": HollowPreset(True, 0.8, 0.8),
    "custom": HollowPreset(True, 1.2, 1.0),
}


def _signed_area_xy(coords: Sequence[Point2]) -> float:
    if len(coords) < 3:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(coords, list(coords[1:]) + [coords[0]]):
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def ensure_ccw(coords: Sequence[Point2]) -> List[Point2]:
    c = list(coords)
    if len(c) > 1 and c[0] == c[-1]:
        c = c[:-1]
    if _signed_area_xy(c) < 0:
        c.reverse()
    return c


def ensure_cw(coords: Sequence[Point2]) -> List[Point2]:
    c = list(coords)
    if len(c) > 1 and c[0] == c[-1]:
        c = c[:-1]
    if _signed_area_xy(c) > 0:
        c.reverse()
    return c


def circle_points(cx: float, cy: float, radius: float, segments: int, clockwise: bool = False) -> List[Point2]:
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    if clockwise:
        angles = angles[::-1]
    return [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]


class MeshBuilder:
    def __init__(self, precision: int = 6):
        self.precision = precision
        self.vertices: List[Point3] = []
        self.faces: List[Tuple[int, int, int]] = []
        self._index = {}

    def v(self, p: Point3) -> int:
        key = (round(float(p[0]), self.precision), round(float(p[1]), self.precision), round(float(p[2]), self.precision))
        idx = self._index.get(key)
        if idx is None:
            idx = len(self.vertices)
            self.vertices.append((float(p[0]), float(p[1]), float(p[2])))
            self._index[key] = idx
        return idx

    def tri(self, p0: Point3, p1: Point3, p2: Point3, normal_hint: Optional[Point3] = None) -> None:
        pts = [np.array(p0, dtype=float), np.array(p1, dtype=float), np.array(p2, dtype=float)]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if normal_hint is not None and float(np.dot(n, np.array(normal_hint, dtype=float))) < 0.0:
            p1, p2 = p2, p1
        i0, i1, i2 = self.v(p0), self.v(p1), self.v(p2)
        if i0 != i1 and i1 != i2 and i2 != i0:
            self.faces.append((i0, i1, i2))

    def quad(self, p0: Point3, p1: Point3, p2: Point3, p3: Point3, normal_hint: Optional[Point3] = None) -> None:
        self.tri(p0, p1, p2, normal_hint)
        self.tri(p0, p2, p3, normal_hint)

    def mesh(self, name: str = "bump_strip") -> trimesh.Trimesh:
        # Vertices are already de-duplicated by MeshBuilder.v(). Avoid trimesh's
        # heavier process/fix_normals pass here because large hollow arrays have
        # many openings and can take a long time to repair even when already valid.
        mesh = trimesh.Trimesh(vertices=np.array(self.vertices), faces=np.array(self.faces), process=False)
        mesh.metadata["name"] = name
        mesh.remove_unreferenced_vertices()
        return mesh


def _polygon_from_bounds(bounds: Tuple[float, float, float, float]) -> Polygon:
    xmin, xmax, ymin, ymax = bounds
    return box(xmin, ymin, xmax, ymax)


def _rect_coords(bounds: Tuple[float, float, float, float], ccw: bool = True) -> List[Point2]:
    xmin, xmax, ymin, ymax = bounds
    coords = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return coords if ccw else list(reversed(coords))


def _iter_polygons(geom) -> Iterable[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def add_polygon_surface(
    mb: MeshBuilder,
    poly,
    z: float,
    normal: str,
    area_check_label: Optional[str] = None,
    tolerance: float = 1e-7,
) -> float:
    """Triangulate shapely polygon(s) and add a horizontal surface.

    Uses triangle/mapbox-earcut via trimesh when available, with a shapely fallback.
    Install with: pip install triangle mapbox_earcut
    """
    hint = (0.0, 0.0, 1.0) if normal == "up" else (0.0, 0.0, -1.0)
    total_area = 0.0
    for part in _iter_polygons(poly):
        if part.area <= 1e-9:
            continue
        did_triangulate = False
        for engine in TRIANGULATION_ENGINES:
            try:
                verts2, faces = trimesh.creation.triangulate_polygon(part, engine=engine, force_vertices=True)
                for f in faces:
                    p0 = (float(verts2[f[0], 0]), float(verts2[f[0], 1]), float(z))
                    p1 = (float(verts2[f[1], 0]), float(verts2[f[1], 1]), float(z))
                    p2 = (float(verts2[f[2], 0]), float(verts2[f[2], 1]), float(z))
                    mb.tri(p0, p1, p2, hint)
                total_area += float(part.area)
                did_triangulate = True
                break
            except Exception:
                continue
        if did_triangulate:
            continue

        # Last-resort fallback: shapely's unconstrained triangulation, filtered against the polygon.
        # This is fine for simple polygons, but triangle/earcut is much better for many holes/grooves.
        cover_poly = part.buffer(tolerance)
        tris = triangulate(part)
        for tri in tris:
            if tri.area <= 1e-10:
                continue
            if not cover_poly.covers(tri):
                continue
            coords = list(tri.exterior.coords)[:3]
            p = [(float(x), float(y), float(z)) for x, y in coords]
            mb.tri(p[0], p[1], p[2], hint)
            total_area += tri.area
    return total_area


def add_vertical_wall(mb: MeshBuilder, ring: Sequence[Point2], z0: float, z1: float) -> None:
    """Add side wall for a ring. Ring orientation controls normal direction.

    For a CCW outer boundary, normals point outward from the polygon.
    For a CW hole boundary, normals point into the hole/void.
    """
    coords = list(ring)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    if abs(z1 - z0) < 1e-9 or len(coords) < 2:
        return
    for (x0, y0), (x1, y1) in zip(coords, coords[1:] + coords[:1]):
        p0 = (x0, y0, z0)
        p1 = (x1, y1, z0)
        p2 = (x1, y1, z1)
        p3 = (x0, y0, z1)
        # The orientation p0->p1->p2 gives the right-hand side normal.
        mb.quad(p0, p1, p2, p3)


def add_disc_cap(mb: MeshBuilder, cx: float, cy: float, radius: float, z: float, segments: int, normal: str) -> None:
    pts = circle_points(cx, cy, radius, segments, clockwise=False)
    center = (cx, cy, z)
    hint = (0.0, 0.0, 1.0) if normal == "up" else (0.0, 0.0, -1.0)
    for p0, p1 in zip(pts, pts[1:] + pts[:1]):
        mb.tri(center, (p0[0], p0[1], z), (p1[0], p1[1], z), hint)


def add_frustum_wall(
    mb: MeshBuilder,
    cx: float,
    cy: float,
    r0: float,
    z0: float,
    r1: float,
    z1: float,
    segments: int,
    outward: str = "external",
) -> None:
    """Add a frustum/cylinder side wall.

    outward='external' uses CCW rings and normals point away from the center.
    outward='internal' uses CW rings and normals point toward the center/void.
    """
    clockwise = outward == "internal"
    ring0 = circle_points(cx, cy, r0, segments, clockwise=clockwise)
    ring1 = circle_points(cx, cy, r1, segments, clockwise=clockwise)
    for p0, p1, q0, q1 in zip(ring0, ring0[1:] + ring0[:1], ring1, ring1[1:] + ring1[:1]):
        mb.quad((p0[0], p0[1], z0), (p1[0], p1[1], z0), (q1[0], q1[1], z1), (q0[0], q0[1], z1))


def add_bump_external(
    mb: MeshBuilder,
    cx: float,
    cy: float,
    base_radius: float,
    top_radius: float,
    base_z: float,
    top_z: float,
    segments: int,
) -> None:
    add_frustum_wall(mb, cx, cy, base_radius, base_z, top_radius, top_z, segments, outward="external")
    add_disc_cap(mb, cx, cy, top_radius, top_z, segments, normal="up")


def add_cavity(
    mb: MeshBuilder,
    cx: float,
    cy: float,
    base_radius: float,
    top_radius: float,
    base_z: float,
    top_z: float,
    main_bottom_z: float,
    side_wall: float,
    top_skin: float,
    segments: int,
) -> None:
    opening_radius = base_radius - side_wall
    if opening_radius <= 0:
        raise ValueError(f"side_wall={side_wall} is too large for base_radius={base_radius}")
    inner_top_z = top_z - top_skin
    if inner_top_z <= base_z + 0.05:
        raise ValueError("top_skin is too large; cavity would not enter the bump.")
    # Outer radius at the internal top cap height, then leave side_wall material there.
    outer_r_at_inner_top = base_radius + (top_radius - base_radius) * ((inner_top_z - base_z) / (top_z - base_z))
    inner_top_radius = outer_r_at_inner_top - side_wall
    if inner_top_radius <= 0.1:
        raise ValueError(
            f"Hollow settings are too aggressive: inner_top_radius={inner_top_radius:.3f}. "
            "Reduce side_wall or increase top_skin."
        )
    # Through the 1 mm main sheet, use a vertical cylinder; then taper inside the bump.
    add_frustum_wall(mb, cx, cy, opening_radius, main_bottom_z, opening_radius, base_z, segments, outward="internal")
    add_frustum_wall(mb, cx, cy, opening_radius, base_z, inner_top_radius, inner_top_z, segments, outward="internal")
    add_disc_cap(mb, cx, cy, inner_top_radius, inner_top_z, segments, normal="down")


def make_groove_union(
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    base_radius: float,
    groove_width: float,
    inner_bounds: Tuple[float, float, float, float],
    include_x: bool = True,
    include_y: bool = True,
    end_margin: float = 1.0,
) -> Optional[Polygon]:
    grooves = []
    xmin, xmax, ymin, ymax = inner_bounds
    # Keep grooves between bumps and away from the outer rim.
    x_span_min = float(centers_x.min() - base_radius + end_margin)
    x_span_max = float(centers_x.max() + base_radius - end_margin)
    y_span_min = float(centers_y.min() - base_radius + end_margin)
    y_span_max = float(centers_y.max() + base_radius - end_margin)
    if include_x and len(centers_x) > 1:
        for x in (centers_x[:-1] + centers_x[1:]) / 2.0:
            grooves.append(box(float(x - groove_width / 2.0), y_span_min, float(x + groove_width / 2.0), y_span_max))
    if include_y and len(centers_y) > 1:
        for y in (centers_y[:-1] + centers_y[1:]) / 2.0:
            grooves.append(box(x_span_min, float(y - groove_width / 2.0), x_span_max, float(y + groove_width / 2.0)))
    if not grooves:
        return None
    clipped = unary_union(grooves).intersection(_polygon_from_bounds(inner_bounds))
    if clipped.is_empty or clipped.area <= 1e-9:
        return None
    return clipped


def build_strip_mesh(
    rows: int,
    cols: int,
    pitch: float = 8.75,
    base_radius: float = 4.0,
    top_radius: float = 2.65,
    bump_height: float = 3.73,
    border: float = 2.0,
    main_sheet_thickness: float = 1.0,
    rim_thickness: float = 3.0,
    rim_width: Optional[float] = None,
    segments: int = 126,
    hollow: bool = False,
    side_wall: float = 1.2,
    top_skin: float = 1.0,
    relief_grooves: bool = False,
    groove_width: float = 0.8,
    groove_depth: float = 0.5,
    groove_end_margin: float = 1.0,
    groove_axis: str = "both",
    name: str = "bump_strip",
) -> trimesh.Trimesh:
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    if rim_width is None:
        rim_width = border
    if rim_width < 0 or border < 0:
        raise ValueError("border/rim_width must be non-negative")
    if rim_width > border + base_radius - max(side_wall, 0.0):
        # Not fatal, but usually means the raised underside cuts too far under outer cavities.
        pass
    if relief_grooves and groove_depth >= main_sheet_thickness:
        raise ValueError("groove_depth must be less than main_sheet_thickness")

    z_rim_bottom = 0.0
    z_main_bottom = rim_thickness - main_sheet_thickness
    z_base_top = rim_thickness
    z_bump_top = rim_thickness + bump_height

    centers_x = (np.arange(cols, dtype=float) - (cols - 1) / 2.0) * pitch
    centers_y = (np.arange(rows, dtype=float) - (rows - 1) / 2.0) * pitch
    centers = [(float(x), float(y)) for y in centers_y for x in centers_x]

    footprint_xmin = float(centers_x.min() - base_radius)
    footprint_xmax = float(centers_x.max() + base_radius)
    footprint_ymin = float(centers_y.min() - base_radius)
    footprint_ymax = float(centers_y.max() + base_radius)

    outer_bounds = (
        footprint_xmin - border,
        footprint_xmax + border,
        footprint_ymin - border,
        footprint_ymax + border,
    )
    inner_bounds = (
        outer_bounds[0] + rim_width,
        outer_bounds[1] - rim_width,
        outer_bounds[2] + rim_width,
        outer_bounds[3] - rim_width,
    )
    if inner_bounds[0] >= inner_bounds[1] or inner_bounds[2] >= inner_bounds[3]:
        raise ValueError("rim_width/border leaves no inner raised underside area")

    outer_poly = _polygon_from_bounds(outer_bounds)
    inner_poly = _polygon_from_bounds(inner_bounds)

    bump_base_holes = [Polygon(circle_points(cx, cy, base_radius, segments, clockwise=False)) for cx, cy in centers]
    top_poly = Polygon(list(outer_poly.exterior.coords), holes=[list(h.exterior.coords) for h in bump_base_holes])
    if not top_poly.is_valid:
        top_poly = top_poly.buffer(0)

    cavity_holes = []
    if hollow:
        opening_radius = base_radius - side_wall
        cavity_holes = [Polygon(circle_points(cx, cy, opening_radius, segments, clockwise=False)) for cx, cy in centers]

    groove_union = None
    if relief_grooves:
        groove_union = make_groove_union(
            centers_x,
            centers_y,
            base_radius,
            groove_width,
            inner_bounds,
            include_x=groove_axis in ("y", "both"),
            include_y=groove_axis in ("x", "both"),
            end_margin=groove_end_margin,
        )
        if groove_union is not None:
            groove_union = groove_union.simplify(1e-7, preserve_topology=True)
        # The default groove width is intentionally narrow enough to avoid the cavity openings.
        # Keeping grooves and circular openings separate makes the underside mesh much cleaner.

    # Build the main underside as a polygon with holes when possible. This is
    # much faster than boolean-differencing hundreds of circular pockets for
    # larger arrays. For the optional groove case, use a boolean difference
    # because the groove union can be a more complex polygon.
    if cavity_holes and groove_union is None:
        central_poly = Polygon(list(inner_poly.exterior.coords), holes=[list(h.exterior.coords) for h in cavity_holes])
    else:
        central_cutouts = []
        if cavity_holes:
            central_cutouts.extend(cavity_holes)
        if groove_union is not None:
            central_cutouts.extend(_iter_polygons(groove_union))
        if central_cutouts:
            central_poly = inner_poly.difference(unary_union(central_cutouts))
        else:
            central_poly = inner_poly
    if not central_poly.is_valid:
        central_poly = central_poly.buffer(0)

    rim_poly = outer_poly.difference(inner_poly)
    if not rim_poly.is_valid:
        rim_poly = rim_poly.buffer(0)

    mb = MeshBuilder(precision=6)

    # Top face of the base/rim sheet, excluding bump footprints.
    add_polygon_surface(mb, top_poly, z_base_top, normal="up", area_check_label="top")
    # Underside: thick outer rim bottom and raised main sheet bottom.
    add_polygon_surface(mb, rim_poly, z_rim_bottom, normal="down", area_check_label="rim_bottom")
    add_polygon_surface(mb, central_poly, z_main_bottom, normal="down", area_check_label="main_bottom")

    # Outer vertical walls and underside step wall.
    add_vertical_wall(mb, ensure_ccw(_rect_coords(outer_bounds, ccw=True)), z_rim_bottom, z_base_top)
    add_vertical_wall(mb, ensure_cw(_rect_coords(inner_bounds, ccw=True)), z_rim_bottom, z_main_bottom)

    # Relief groove recessed ceilings and side walls.
    if groove_union is not None:
        groove_ceiling_z = z_main_bottom + groove_depth
        add_polygon_surface(mb, groove_union, groove_ceiling_z, normal="down", area_check_label="groove_ceiling")
        for gpoly in _iter_polygons(groove_union):
            add_vertical_wall(mb, ensure_cw(list(gpoly.exterior.coords)), z_main_bottom, groove_ceiling_z)
            # If the union somehow has holes, orient the reverse; these are islands of solid inside the groove.
            for interior in gpoly.interiors:
                add_vertical_wall(mb, ensure_ccw(list(interior.coords)), z_main_bottom, groove_ceiling_z)

    # Bumps and optional hollow cavities.
    for cx, cy in centers:
        add_bump_external(mb, cx, cy, base_radius, top_radius, z_base_top, z_bump_top, segments)
        if hollow:
            add_cavity(mb, cx, cy, base_radius, top_radius, z_base_top, z_bump_top, z_main_bottom, side_wall, top_skin, segments)

    mesh = mb.mesh(name=name)
    return mesh


def build_mold_mesh(
    rows: int,
    cols: int,
    pitch: float = 8.75,
    base_radius: float = 4.0,
    top_radius: float = 2.65,
    bump_height: float = 3.73,
    border: float = 2.0,
    cast_back_thickness: float = 1.0,
    mold_wall: float = 5.0,
    mold_floor: float = 2.5,
    segments: int = 126,
    name: str = "bump_strip_open_face_mold",
) -> trimesh.Trimesh:
    """Build a one-piece open-face mold for casting a solid silicone bump strip.

    The mold has a rectangular open cavity with negative frustum bump recesses.
    The exposed pour face becomes the flat back of the cast strip.
    """
    centers_x = (np.arange(cols, dtype=float) - (cols - 1) / 2.0) * pitch
    centers_y = (np.arange(rows, dtype=float) - (rows - 1) / 2.0) * pitch
    centers = [(float(x), float(y)) for y in centers_y for x in centers_x]

    cavity_xmin = float(centers_x.min() - base_radius - border)
    cavity_xmax = float(centers_x.max() + base_radius + border)
    cavity_ymin = float(centers_y.min() - base_radius - border)
    cavity_ymax = float(centers_y.max() + base_radius + border)
    cavity_bounds = (cavity_xmin, cavity_xmax, cavity_ymin, cavity_ymax)
    outer_bounds = (cavity_xmin - mold_wall, cavity_xmax + mold_wall, cavity_ymin - mold_wall, cavity_ymax + mold_wall)

    z_bottom = 0.0
    z_tip_floor = mold_floor
    z_cavity_floor = mold_floor + bump_height
    z_top = z_cavity_floor + cast_back_thickness

    outer_poly = _polygon_from_bounds(outer_bounds)
    cavity_poly = _polygon_from_bounds(cavity_bounds)
    bump_base_holes = [Polygon(circle_points(cx, cy, base_radius, segments, clockwise=False)) for cx, cy in centers]
    cavity_floor_poly = Polygon(list(cavity_poly.exterior.coords), holes=[list(h.exterior.coords) for h in bump_base_holes])
    if not cavity_floor_poly.is_valid:
        cavity_floor_poly = cavity_floor_poly.buffer(0)
    top_ring_poly = outer_poly.difference(cavity_poly)
    if not top_ring_poly.is_valid:
        top_ring_poly = top_ring_poly.buffer(0)

    mb = MeshBuilder(precision=6)
    # Bottom of the mold block.
    add_polygon_surface(mb, outer_poly, z_bottom, normal="down")
    # Top rim of the mold block around the open cavity.
    add_polygon_surface(mb, top_ring_poly, z_top, normal="up")
    # Flat cavity floor between bump recesses: solid below, void above -> normal up.
    add_polygon_surface(mb, cavity_floor_poly, z_cavity_floor, normal="up")

    # Outer walls of block.
    add_vertical_wall(mb, ensure_ccw(_rect_coords(outer_bounds, ccw=True)), z_bottom, z_top)
    # Inner vertical wall of cavity: solid outside cavity, void inside -> normals into cavity.
    add_vertical_wall(mb, ensure_cw(_rect_coords(cavity_bounds, ccw=True)), z_cavity_floor, z_top)

    # Negative bump recesses: solid outside/under frustum, void above/inside.
    # Use internal orientation so normals point into the cavity void.
    for cx, cy in centers:
        add_frustum_wall(mb, cx, cy, base_radius, z_cavity_floor, top_radius, z_tip_floor, segments, outward="internal")
        # Bottom of recess/tip flat: solid below, void above -> normal up.
        add_disc_cap(mb, cx, cy, top_radius, z_tip_floor, segments, normal="up")

    mesh = mb.mesh(name=name)
    # Mold cavity walls can have mixed local orientation depending on how the
    # negative geometry is stitched. Repairing normals is quick enough for mold
    # meshes and makes exported molds winding-consistent.
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    return mesh


def mesh_summary(mesh: trimesh.Trimesh) -> str:
    b = mesh.bounds
    e = mesh.extents
    return (
        f"bounds min={b[0].round(3).tolist()} max={b[1].round(3).tolist()} | "
        f"extents={e.round(3).tolist()} | faces={len(mesh.faces)} | "
        f"watertight={mesh.is_watertight} | winding={mesh.is_winding_consistent} | euler={mesh.euler_number}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parametric bump-strip STLs and open-face molds.")
    parser.add_argument("--mode", choices=["strip", "mold"], default="strip", help="Generate a printable strip or an open-face casting mold.")
    parser.add_argument("--rows", type=int, default=4, help="Number of bump rows along Y.")
    parser.add_argument("--cols", type=int, default=8, help="Number of bump columns along X.")
    parser.add_argument("--pitch", type=float, default=8.75, help="Bump center-to-center spacing in mm.")
    parser.add_argument("--base-radius", type=float, default=4.0, help="Bump footprint radius in mm.")
    parser.add_argument("--top-radius", type=float, default=2.65, help="Bump top-flat radius in mm.")
    parser.add_argument("--bump-height", type=float, default=3.73, help="Bump height above base top in mm.")
    parser.add_argument("--border", type=float, default=2.0, help="Border outside the bump footprint in mm.")
    parser.add_argument("--segments", type=int, default=126, help="Circle/frustum resolution. Original STL used 126 segments.")
    parser.add_argument("--output", type=Path, required=True, help="Output STL path.")
    parser.add_argument("--triangulation-engine", choices=["auto", "earcut", "triangle"], default="auto", help="Constrained triangulation backend. Auto uses earcut normally and triangle for relief grooves.")

    # Strip-only controls.
    parser.add_argument("--variant", choices=list(PRESETS.keys()), default="soft", help="Hollowing preset for strip mode.")
    parser.add_argument("--main-sheet-thickness", type=float, default=1.0, help="Raised underside main sheet thickness in mm.")
    parser.add_argument("--rim-thickness", type=float, default=3.0, help="Thick perimeter/rim thickness in mm.")
    parser.add_argument("--rim-width", type=float, default=None, help="Width of the thick perimeter/rim underside. Defaults to --border.")
    parser.add_argument("--side-wall", type=float, default=None, help="Override hollow side-wall target in mm.")
    parser.add_argument("--top-skin", type=float, default=None, help="Override hollow top skin thickness in mm.")
    parser.add_argument("--hollow", action="store_true", help="Force hollowing on regardless of variant.")
    parser.add_argument("--no-hollow", action="store_true", help="Force hollowing off regardless of variant.")
    parser.add_argument("--relief-grooves", action="store_true", help="Add shallow underside grid grooves between bumps.")
    parser.add_argument("--groove-width", type=float, default=0.8, help="Underside relief groove width in mm.")
    parser.add_argument("--groove-depth", type=float, default=0.5, help="Underside relief groove depth into the 1 mm main sheet.")
    parser.add_argument("--groove-end-margin", type=float, default=1.0, help="Clearance from groove ends to the raised-underside edge in mm.")
    parser.add_argument("--groove-axis", choices=["x", "y", "both"], default="x", help="Groove direction: x = long grooves between rows; y = grooves between columns; both = grid.")

    # Mold-only controls.
    parser.add_argument("--cast-back-thickness", type=float, default=1.0, help="For mold mode: cast strip flat-back thickness above the bump base plane.")
    parser.add_argument("--mold-wall", type=float, default=5.0, help="For mold mode: mold wall around the casting cavity in mm.")
    parser.add_argument("--mold-floor", type=float, default=2.5, help="For mold mode: material below the deepest bump recess in mm.")

    args = parser.parse_args()

    preferred_engine = args.triangulation_engine
    if preferred_engine == "auto":
        preferred_engine = "triangle" if (args.mode == "strip" and args.relief_grooves) else "earcut"
    set_triangulation_engine(preferred_engine)

    if args.mode == "mold":
        mesh = build_mold_mesh(
            rows=args.rows,
            cols=args.cols,
            pitch=args.pitch,
            base_radius=args.base_radius,
            top_radius=args.top_radius,
            bump_height=args.bump_height,
            border=args.border,
            cast_back_thickness=args.cast_back_thickness,
            mold_wall=args.mold_wall,
            mold_floor=args.mold_floor,
            segments=args.segments,
            name=f"open_face_mold_{args.rows}rows_{args.cols}cols",
        )
    else:
        preset = PRESETS[args.variant]
        hollow = preset.hollow
        if args.hollow:
            hollow = True
        if args.no_hollow:
            hollow = False
        side_wall = preset.side_wall if args.side_wall is None else args.side_wall
        top_skin = preset.top_skin if args.top_skin is None else args.top_skin
        mesh = build_strip_mesh(
            rows=args.rows,
            cols=args.cols,
            pitch=args.pitch,
            base_radius=args.base_radius,
            top_radius=args.top_radius,
            bump_height=args.bump_height,
            border=args.border,
            main_sheet_thickness=args.main_sheet_thickness,
            rim_thickness=args.rim_thickness,
            rim_width=args.rim_width,
            segments=args.segments,
            hollow=hollow,
            side_wall=side_wall,
            top_skin=top_skin,
            relief_grooves=args.relief_grooves,
            groove_width=args.groove_width,
            groove_depth=args.groove_depth,
            groove_end_margin=args.groove_end_margin,
            groove_axis=args.groove_axis,
            name=f"bump_strip_{args.rows}rows_{args.cols}cols_{args.variant}",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(args.output)
    print(mesh_summary(mesh))
    if not mesh.is_watertight:
        print("WARNING: mesh is not watertight. Try reducing segment count or adjusting parameters.")


if __name__ == "__main__":
    main()
