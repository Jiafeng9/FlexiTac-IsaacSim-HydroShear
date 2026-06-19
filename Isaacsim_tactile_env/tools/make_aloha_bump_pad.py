from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "bump_strip_outputs" / "bump_strip_generator.py"
ASSET_DIR = REPO_ROOT / "Isaacsim_tactile_env" / "assets"
MESH_DIR = ASSET_DIR / "meshes"
SOURCE_URDF = ASSET_DIR / "aloha_tactile.urdf"
FLAT_PAD_MESH = "./meshes/aloha_flat_pad.obj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an ALOHA-sized bump elastomer pad mesh and URDF.")
    parser.add_argument("--rows", type=int, default=6, help="Bump rows across the pad width.")
    parser.add_argument("--cols", type=int, default=16, help="Bump columns along the pad length.")
    parser.add_argument(
        "--auto-grid-from-target-size",
        action="store_true",
        help="Derive rows/cols as floor(real active width/length divided by real bump pitch).",
    )
    parser.add_argument("--real-bump-pitch-mm", type=float, default=None, help="Real-world bump center spacing in mm.")
    parser.add_argument("--bump-pitch-mm", type=float, default=None, help="Deprecated alias for --real-bump-pitch-mm.")
    parser.add_argument("--source-pitch-mm", type=float, default=8.75, help="Source generator bump pitch in mm.")
    parser.add_argument("--variant", choices=("solid", "medium", "soft", "extra-soft"), default="soft")
    parser.add_argument("--target-width-mm", type=float, default=26.0, help="ALOHA pad local X extent.")
    parser.add_argument("--target-thickness-mm", type=float, default=3.0, help="ALOHA pad local Y extent.")
    parser.add_argument("--target-length-mm", type=float, default=66.65, help="ALOHA pad local Z extent.")
    parser.add_argument(
        "--real-active-width-mm",
        type=float,
        default=None,
        help="Real-world active width in mm. Defaults to --target-width-mm.",
    )
    parser.add_argument(
        "--real-active-length-mm",
        type=float,
        default=None,
        help="Real-world active length in mm. Defaults to --target-length-mm.",
    )
    parser.add_argument(
        "--fit-target-extents",
        action="store_true",
        help="Legacy mode: stretch tangential axes to exactly match target width/length instead of preserving pitch.",
    )
    parser.add_argument(
        "--back-plane-y-mm",
        type=float,
        default=1.5,
        help="Local Y coordinate of the flat mounting/back plane. Defaults to the original flat pad +Y face.",
    )
    parser.add_argument("--mesh-name", default="aloha_bump_pad.obj", help="Output mesh file under assets/meshes.")
    parser.add_argument("--urdf-name", default="aloha_tactile_bump.urdf", help="Output URDF file under assets.")
    parser.add_argument(
        "--urdf-replace-scope",
        choices=("both", "visual", "collision"),
        default="both",
        help="Which URDF pad mesh references to replace. Default replaces visual and collision.",
    )
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep the generated source STL.")
    return parser.parse_args()


def _run_generator(args: argparse.Namespace, output_stl: Path) -> None:
    if not GENERATOR.is_file():
        raise FileNotFoundError(GENERATOR)
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--rows",
        str(int(args.rows)),
        "--cols",
        str(int(args.cols)),
        "--pitch",
        str(float(args.source_pitch_mm)),
        "--variant",
        str(args.variant),
        "--output",
        str(output_stl),
    ]
    subprocess.run(cmd, check=True)


def _convert_to_aloha_pad(source_stl: Path, output_obj: Path, args: argparse.Namespace) -> trimesh.Trimesh:
    mesh = trimesh.load(str(source_stl), force="mesh", process=False)
    if mesh.is_empty:
        raise ValueError(f"empty mesh: {source_stl}")
    mesh.merge_vertices()

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    src_min = vertices.min(axis=0)
    src_max = vertices.max(axis=0)
    src_center = (src_min + src_max) * 0.5
    src_extent = src_max - src_min

    target_width_m = float(args.target_width_mm) * 0.001
    target_thickness_m = float(args.target_thickness_mm) * 0.001
    target_length_m = float(args.target_length_mm) * 0.001
    back_plane_y_m = float(args.back_plane_y_mm) * 0.001

    if bool(args.fit_target_extents):
        width_scale = target_width_m / max(src_extent[1], 1.0e-12)
        length_scale = target_length_m / max(src_extent[0], 1.0e-12)
    else:
        real_pitch_mm = _real_bump_pitch_mm(args)
        real_width_mm, real_length_mm = _real_active_size_mm(args)
        if float(args.source_pitch_mm) <= 0.0:
            raise ValueError("--source-pitch-mm must be positive")
        sim_pitch_width_mm = real_pitch_mm * float(args.target_width_mm) / real_width_mm
        sim_pitch_length_mm = real_pitch_mm * float(args.target_length_mm) / real_length_mm
        width_scale = sim_pitch_width_mm * 0.001 / float(args.source_pitch_mm)
        length_scale = sim_pitch_length_mm * 0.001 / float(args.source_pitch_mm)

    # Source bump strip: X=length, Y=width, Z=outward bump height, units mm.
    # ALOHA flat pad: local X=width, local Y=thickness/normal, local Z=length, units m.
    # Bumps face local -Y, which maps inward for both left and right elastomer links.
    # Keep the flat back plane fixed so thicker variants grow only toward contact.
    out = np.zeros_like(vertices, dtype=np.float64)
    out[:, 0] = (vertices[:, 1] - src_center[1]) * width_scale
    out[:, 2] = (vertices[:, 0] - src_center[0]) * length_scale
    out[:, 1] = back_plane_y_m - (vertices[:, 2] - src_min[2]) * (
        target_thickness_m / max(src_extent[2], 1.0e-12)
    )

    converted = trimesh.Trimesh(vertices=out, faces=faces, process=False)
    converted.metadata["name"] = Path(args.mesh_name).stem
    output_obj.parent.mkdir(parents=True, exist_ok=True)
    converted.export(output_obj)
    return converted


def _resolve_grid_from_target_size(args: argparse.Namespace) -> None:
    if not bool(args.auto_grid_from_target_size):
        return
    pitch_mm = _real_bump_pitch_mm(args)
    real_width_mm, real_length_mm = _real_active_size_mm(args)
    args.rows = max(1, int(np.floor(real_width_mm / pitch_mm + 1.0e-9)))
    args.cols = max(1, int(np.floor(real_length_mm / pitch_mm + 1.0e-9)))


def _real_bump_pitch_mm(args: argparse.Namespace) -> float:
    pitch = args.real_bump_pitch_mm
    if pitch is None:
        pitch = args.bump_pitch_mm
    if pitch is None:
        pitch = 4.0
    pitch = float(pitch)
    if pitch <= 0.0:
        raise ValueError("--real-bump-pitch-mm must be positive")
    return pitch


def _real_active_size_mm(args: argparse.Namespace) -> tuple[float, float]:
    width_mm = float(args.target_width_mm if args.real_active_width_mm is None else args.real_active_width_mm)
    length_mm = float(args.target_length_mm if args.real_active_length_mm is None else args.real_active_length_mm)
    if width_mm <= 0.0 or length_mm <= 0.0:
        raise ValueError("--real-active-width-mm and --real-active-length-mm must be positive")
    return width_mm, length_mm


def _replace_pad_mesh_references(text: str, *, mesh_name: str, scope: str) -> tuple[str, int]:
    selected = {"visual", "collision"} if scope == "both" else {scope}
    replacement = f"./meshes/{mesh_name}"
    count = 0
    block_re = re.compile(r"<(visual|collision)\b[^>]*>.*?</\1>", re.DOTALL)

    def replace_block(match: re.Match) -> str:
        nonlocal count
        kind = match.group(1)
        block = match.group(0)
        if kind not in selected:
            return block
        count += block.count(FLAT_PAD_MESH)
        return block.replace(FLAT_PAD_MESH, replacement)

    return block_re.sub(replace_block, text), count


def _write_bump_urdf(output_urdf: Path, mesh_name: str, *, scope: str) -> None:
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(SOURCE_URDF)
    text = SOURCE_URDF.read_text(encoding="utf-8")
    text, replaced = _replace_pad_mesh_references(text, mesh_name=mesh_name, scope=scope)
    if replaced == 0:
        raise RuntimeError(f"No {scope} URDF references to {FLAT_PAD_MESH} were replaced")
    output_urdf.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    _resolve_grid_from_target_size(args)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    output_obj = MESH_DIR / args.mesh_name
    output_urdf = ASSET_DIR / args.urdf_name
    intermediate = MESH_DIR / f"{Path(args.mesh_name).stem}_source_{args.rows}x{args.cols}_{args.variant}.stl"

    _run_generator(args, intermediate)
    converted = _convert_to_aloha_pad(intermediate, output_obj, args)
    _write_bump_urdf(output_urdf, output_obj.name, scope=str(args.urdf_replace_scope))

    if not args.keep_intermediate:
        intermediate.unlink(missing_ok=True)

    print(f"wrote mesh: {output_obj}")
    print(f"wrote urdf: {output_urdf}")
    print(f"bounds m: {converted.bounds.tolist()}")
    print(f"extents m: {converted.extents.tolist()}")
    real_pitch_mm = _real_bump_pitch_mm(args)
    real_width_mm, real_length_mm = _real_active_size_mm(args)
    sim_pitch_width_mm = real_pitch_mm * float(args.target_width_mm) / real_width_mm
    sim_pitch_length_mm = real_pitch_mm * float(args.target_length_mm) / real_length_mm
    print(
        "layout: "
        f"rows={int(args.rows)} cols={int(args.cols)} "
        f"real_active={real_width_mm:.3f}x{real_length_mm:.3f} mm "
        f"sim_active={float(args.target_width_mm):.3f}x{float(args.target_length_mm):.3f} mm "
        f"real_pitch={real_pitch_mm:.3f} mm "
        f"sim_pitch={sim_pitch_width_mm:.3f}x{sim_pitch_length_mm:.3f} mm "
        f"mode={'fit_target_extents' if bool(args.fit_target_extents) else 'preserve_bump_pitch'}"
    )
    print(f"faces: {len(converted.faces)} watertight={converted.is_watertight}")


if __name__ == "__main__":
    main()
