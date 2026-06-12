from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tactile shear CSV probe results.")
    parser.add_argument("csv", nargs="*", type=str, help="CSV files. Default: latest two tactile_shear CSVs.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "output" / "tactile_shear_csv_summary"),
    )
    return parser.parse_args()


def _latest_csvs() -> list[Path]:
    csv_dir = ROOT / "output" / "tactile_shear_csv"
    files = sorted(csv_dir.glob("tactile_shear_*.csv"), key=lambda p: p.stat().st_mtime)
    return files[-2:]


def _float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except Exception:
        return 0.0


def _load(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _active_pad(rows: list[dict]) -> tuple[str, str]:
    pads = sorted({(r.get("pad", ""), r.get("pad_label", "")) for r in rows})
    best = ("", "")
    best_score = -1.0
    for pad, label in pads:
        rr = [r for r in rows if r.get("pad", "") == pad]
        if not rr:
            continue
        score = max(
            max(abs(_float(r, "shear_x")), abs(_float(r, "shear_y")), abs(_float(r, "normal_force")))
            for r in rr
        )
        if score > best_score:
            best = (pad, label)
            best_score = score
    return best


def _frame_series(rows: list[dict], pad: str) -> dict[str, np.ndarray]:
    frames = sorted({int(r["frame"]) for r in rows if r.get("pad", "") == pad})
    out: dict[str, list[float | str]] = {
        "frame": [],
        "axis": [],
        "phase": [],
        "force_x": [],
        "force_y": [],
        "hydro_rel_x": [],
        "hydro_rel_y": [],
        "object_x": [],
        "object_y": [],
        "normal": [],
    }
    for frame in frames:
        rr = [r for r in rows if r.get("pad", "") == pad and int(r["frame"]) == frame]
        if not rr:
            continue
        out["frame"].append(float(frame))
        out["axis"].append(rr[0].get("commanded_axis", ""))
        out["phase"].append(rr[0].get("trial_phase", ""))
        out["force_x"].append(float(np.mean([abs(_float(r, "shear_x")) for r in rr])))
        out["force_y"].append(float(np.mean([abs(_float(r, "shear_y")) for r in rr])))
        out["hydro_rel_x"].append(float(np.mean([abs(_float(r, "hydro_rel_delta_shear_x")) for r in rr])))
        out["hydro_rel_y"].append(float(np.mean([abs(_float(r, "hydro_rel_delta_shear_y")) for r in rr])))
        out["object_x"].append(float(np.mean([abs(_float(r, "object_delta_shear_x")) for r in rr])))
        out["object_y"].append(float(np.mean([abs(_float(r, "object_delta_shear_y")) for r in rr])))
        out["normal"].append(float(np.mean([_float(r, "normal_force") for r in rr])))
    return {
        key: np.asarray(value, dtype=object if key in ("axis", "phase") else np.float64)
        for key, value in out.items()
    }


def _move_stats(rows: list[dict], pad: str) -> dict[str, float | str]:
    rr = [r for r in rows if r.get("pad", "") == pad and r.get("commanded_axis", "") in ("x", "y")]
    if not rr:
        return {}
    force_x = float(np.mean([abs(_float(r, "shear_x")) for r in rr]))
    force_y = float(np.mean([abs(_float(r, "shear_y")) for r in rr]))
    rel_x = float(np.mean([abs(_float(r, "hydro_rel_delta_shear_x")) for r in rr]))
    rel_y = float(np.mean([abs(_float(r, "hydro_rel_delta_shear_y")) for r in rr]))
    obj_x = float(np.mean([abs(_float(r, "object_delta_shear_x")) for r in rr]))
    obj_y = float(np.mean([abs(_float(r, "object_delta_shear_y")) for r in rr]))
    return {
        "axis": rr[0].get("commanded_axis", ""),
        "force_x": force_x,
        "force_y": force_y,
        "rel_x": rel_x,
        "rel_y": rel_y,
        "obj_x": obj_x,
        "obj_y": obj_y,
        "normal": float(np.mean([_float(r, "normal_force") for r in rr])),
    }


def _ratio(a: float, b: float) -> float:
    return float(abs(a) / (abs(b) + 1.0e-30))


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.csv] if args.csv else _latest_csvs()
    if len(paths) < 1:
        raise SystemExit("No tactile_shear CSV files found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for path in paths:
        rows, _ = _load(path)
        pad, label = _active_pad(rows)
        cases.append(
            {
                "path": path,
                "rows": rows,
                "pad": pad,
                "label": label,
                "series": _frame_series(rows, pad),
                "stats": _move_stats(rows, pad),
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(len(cases), 2, figsize=(12.5, 4.2 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axs = np.asarray([axs])

    for row_idx, case in enumerate(cases):
        series = case["series"]
        stats = case["stats"]
        axis = str(stats.get("axis", "?"))
        title = f"{case['path'].name} | pad{case['pad']} {case['label']} | commanded shear_{axis}"

        ax = axs[row_idx, 0]
        ax.plot(series["frame"], series["hydro_rel_x"], label="hydro_rel_x", color="tab:blue")
        ax.plot(series["frame"], series["hydro_rel_y"], label="hydro_rel_y", color="tab:orange")
        ax.plot(series["frame"], series["object_x"], "--", label="object_x", color="tab:blue", alpha=0.45)
        ax.plot(series["frame"], series["object_y"], "--", label="object_y", color="tab:orange", alpha=0.45)
        ax.set_title(title + "\ninput motion on pad axes")
        ax.set_xlabel("CSV frame")
        ax.set_ylabel("abs delta (m)")
        ax.set_yscale("symlog", linthresh=1.0e-10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

        ax = axs[row_idx, 1]
        ax.plot(series["frame"], series["force_x"], label="force_shear_x", color="tab:blue")
        ax.plot(series["frame"], series["force_y"], label="force_shear_y", color="tab:orange")
        ax.set_title("final tactile force shear")
        ax.set_xlabel("CSV frame")
        ax.set_ylabel("mean abs force")
        ax.set_yscale("symlog", linthresh=1.0e-12)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    stem = "_".join(p.stem.replace("tactile_shear_", "") for p in paths)
    png_path = output_dir / f"tactile_shear_summary_{stem}.png"
    html_path = output_dir / f"tactile_shear_summary_{stem}.html"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    rows_html = []
    for case in cases:
        stats = case["stats"]
        axis = str(stats.get("axis", "?"))
        if axis == "x":
            force_ratio = _ratio(float(stats["force_x"]), float(stats["force_y"]))
            rel_ratio = _ratio(float(stats["rel_x"]), float(stats["rel_y"]))
            obj_ratio = _ratio(float(stats["obj_x"]), float(stats["obj_y"]))
            ratio_label = "x/y"
        else:
            force_ratio = _ratio(float(stats["force_y"]), float(stats["force_x"]))
            rel_ratio = _ratio(float(stats["rel_y"]), float(stats["rel_x"]))
            obj_ratio = _ratio(float(stats["obj_y"]), float(stats["obj_x"]))
            ratio_label = "y/x"
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(case['path'].name)}</td>"
            f"<td>pad{html.escape(str(case['pad']))} {html.escape(case['label'])}</td>"
            f"<td>shear_{html.escape(axis)}</td>"
            f"<td>{obj_ratio:.1f}:1 {ratio_label}</td>"
            f"<td>{rel_ratio:.1f}:1 {ratio_label}</td>"
            f"<td>{force_ratio:.1f}:1 {ratio_label}</td>"
            f"<td>{float(stats.get('normal', 0.0)):.3e}</td>"
            "</tr>"
        )

    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Tactile shear CSV summary</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;margin-bottom:18px;}td,th{border:1px solid #ccc;padding:6px 9px;}"
        "th{background:#f5f5f5;}img{max-width:100%;height:auto;border:1px solid #ddd;}</style>"
        "</head><body>"
        "<h1>Tactile shear CSV summary</h1>"
        "<table><thead><tr><th>CSV</th><th>Active pad</th><th>Command</th>"
        "<th>Object purity</th><th>Hydro rel purity</th><th>Force shear purity</th><th>Mean normal</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows_html)
        + "</tbody></table>"
        f"<img src='{html.escape(png_path.name)}' alt='summary plot'>"
        "</body></html>",
        encoding="utf-8",
    )

    print(f"[SAVE] {png_path}")
    print(f"[SAVE] {html_path}")


if __name__ == "__main__":
    main()
