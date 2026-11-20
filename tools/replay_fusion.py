"""Replays a recorded scan through the fusion the app uses, offline.

A recording comes from the app run with ALL3DSCAN_RECORD_DIR (see
gui_python/recording.py). Replaying it feeds the same frames to
all3dscan_engine.Fusion as the app did, so the fusion's settings can be
compared on one scan instead of on a new scan each time:

    python tools/replay_fusion.py <recording>                # as it ran live
    python tools/replay_fusion.py <recording> --check        # ...and compare with live
    python tools/replay_fusion.py <recording> --frames all   # every frame, not just offered
    python tools/replay_fusion.py <recording> --grid 0.3,0.6,0.85 2.5,1.8,1.4
    python tools/replay_fusion.py <recording> --ply model.ply

Overlap and RMSE are the fusion's own opinion of a fit, and a frame placed
in the wrong orientation can score well on both, so this also reports what
the poses say: how far the scanner moved between merged frames (a merge
that jumps further than a hand moves in that time is a misplaced frame),
and where the last merged frame sits relative to the first.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import all3dscan_engine as engine


def load(recording: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((recording / "manifest.json").read_text(encoding="utf-8"))
    return manifest, manifest["frames"]


def pose_matrix(pose) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def step_between(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """How far, in mm and degrees, pose b is from pose a."""
    relative = np.linalg.inv(a) @ b
    shift = float(np.linalg.norm(relative[:3, 3]))
    cosine = (np.trace(relative[:3, :3]) - 1.0) / 2.0
    return shift, math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def replay(recording: Path, frames: list[dict], which: str, voxel_mm: float,
           min_overlap: float | None, max_rmse_mm: float | None,
           jump_mm: float, jump_deg: float) -> dict:
    fusion = engine.Fusion(voxel_mm, min_overlap=min_overlap, max_rmse_mm=max_rmse_mm)
    fits, add_seconds = [], []
    last_pose = None
    last_time = None
    steps = []  # (frame number, mm, degrees, seconds since the last merge)
    first_pose = None
    for number, entry in enumerate(frames):
        if which == "offered" and not entry.get("offered"):
            continue
        data = np.load(recording / entry["file"])
        if not len(data["points"]):
            continue
        step = int(entry.get("step", 1))
        started = time.perf_counter()
        fit = fusion.add(data["points"][::step], data["colors"][::step],
                         data["normals"][::step], depth_mm=data["depth_mm"])
        add_seconds.append((time.perf_counter() - started, fit["model_points"]))
        fits.append((number, fit))
        if fit["merged"]:
            pose = pose_matrix(fit["pose"])
            if first_pose is None:
                first_pose = pose
            if last_pose is not None:
                mm, deg = step_between(last_pose, pose)
                steps.append((number, mm, deg, entry["arrived_s"] - last_time))
            last_pose, last_time = pose, entry["arrived_s"]
    merged = [fit for _n, fit in fits if fit["merged"]]
    jumps = [s for s in steps if s[1] > jump_mm or s[2] > jump_deg]
    result = {
        "min_overlap": fusion.min_overlap, "max_rmse_mm": fusion.max_rmse_mm,
        "frames": which, "given": len(fits), "merged": len(merged),
        "model_points": fusion.model_points, "fits": fits, "steps": steps, "jumps": jumps,
        "add_seconds": add_seconds,
    }
    if first_pose is not None and last_pose is not None:
        result["end_vs_start"] = step_between(first_pose, last_pose)
    result["cloud"] = fusion.cloud
    return result


def check_against_live(frames: list[dict], fits: list) -> str:
    """How closely an --frames offered replay matched what the app recorded."""
    compared = agree = 0
    overlap_diff = []
    for number, fit in fits:
        live = frames[number].get("fit")
        if live is None:
            continue
        compared += 1
        agree += live["merged"] == fit["merged"]
        overlap_diff.append(abs(live["overlap"] - fit["overlap"]))
    if not compared:
        return "nothing to compare: the recording holds no live fits"
    first_split = next((n for n, fit in fits
                        if frames[n].get("fit") and frames[n]["fit"]["merged"] != fit["merged"]),
                       None)
    return (f"live vs replay: {agree}/{compared} frames agree on merged; overlap differs by "
            f"median {np.median(overlap_diff):.4f}, max {max(overlap_diff):.4f}"
            + (f"; first disagreement at frame {first_split}" if first_split is not None else ""))


def report(result: dict, verbose: bool) -> str:
    lines = [f"min_overlap={result['min_overlap']:.2f} max_rmse={result['max_rmse_mm']:.2f} mm "
             f"frames={result['frames']}: merged {result['merged']}/{result['given']}, "
             f"model {result['model_points']:,} points"]
    steps = result["steps"]
    if steps:
        mm = np.array([s[1] for s in steps])
        deg = np.array([s[2] for s in steps])
        lines.append(f"  between merges: median {np.median(mm):.1f} mm / {np.median(deg):.2f} deg, "
                     f"max {mm.max():.1f} mm / {deg.max():.2f} deg; "
                     f"{len(result['jumps'])} jump(s) over the limit")
        for number, step_mm, step_deg, seconds in result["jumps"][:10]:
            lines.append(f"    jump at frame {number}: {step_mm:.1f} mm, {step_deg:.1f} deg "
                         f"in {seconds:.2f} s")
    if "end_vs_start" in result:
        lines.append("  last merged pose vs first: {:.1f} mm, {:.1f} deg".format(*result["end_vs_start"]))
    timings = result["add_seconds"]
    if timings:
        quarter = max(len(timings) // 4, 1)
        early = np.mean([t for t, _ in timings[:quarter]]) * 1000
        late = np.mean([t for t, _ in timings[-quarter:]]) * 1000
        lines.append(f"  add(): {early:.0f} ms a frame in the first quarter, {late:.0f} ms in "
                     f"the last (model {timings[-1][1]:,} points)")
    if verbose:
        for number, fit in result["fits"]:
            lines.append(f"    {number:4d} {'M' if fit['merged'] else '.'} overlap={fit['overlap']:.3f} "
                         f"rmse={fit['rmse_mm']:.2f} model={fit['model_points']}")
    return "\n".join(lines)


def write_ply(path: Path, cloud) -> None:
    points, colors = cloud()[:2]
    with open(path, "wb") as out:
        out.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
                   "property float x\nproperty float y\nproperty float z\n"
                   "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                   "end_header\n").encode("ascii"))
        rows = np.empty(len(points), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rows["p"], rows["c"] = points, colors
        out.write(rows.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--frames", choices=("offered", "all"), default="offered",
                        help="the frames the app handed to the fusion, or every frame")
    parser.add_argument("--min-overlap", type=float)
    parser.add_argument("--max-rmse", type=float)
    parser.add_argument("--grid", nargs=2, metavar=("OVERLAPS", "RMSES"),
                        help="comma-separated values to try every pair of")
    parser.add_argument("--jump-mm", type=float, default=25.0,
                        help="a move between merged frames above this is reported as a jump")
    parser.add_argument("--jump-deg", type=float, default=10.0)
    parser.add_argument("--check", action="store_true",
                        help="compare with the fits the app recorded (use --frames offered)")
    parser.add_argument("--ply", type=Path, help="write the fused model here")
    parser.add_argument("--verbose", action="store_true", help="one line per frame")
    args = parser.parse_args()

    manifest, frames = load(args.recording)
    voxel = float(manifest["settings"].get("fusion_voxel_mm", 1.0))
    summary = manifest.get("summary", {})
    print(f"{args.recording.name}: {len(frames)} frames recorded, "
          f"{sum(e['offered'] for e in frames)} offered live, live model "
          f"{summary.get('model_points', '?'):,} points (commit {manifest.get('commit', '?')})")
    if args.grid:
        pairs = [(float(o), float(r)) for o in args.grid[0].split(",")
                 for r in args.grid[1].split(",")]
    else:
        pairs = [(args.min_overlap, args.max_rmse)]
    for overlap, rmse in pairs:
        result = replay(args.recording, frames, args.frames, voxel, overlap, rmse,
                        args.jump_mm, args.jump_deg)
        print(report(result, args.verbose))
        if args.check:
            print("  " + check_against_live(frames, result["fits"]))
        if args.ply and len(pairs) == 1:
            write_ply(args.ply, result["cloud"])
            print(f"  wrote {args.ply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
