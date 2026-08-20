"""TrashCan synthesizer for UT-V2X frames.

TrashCan = 42% of the V2X truck meta-class (4055/9689 in legal train frames)
but is ABSENT from UT-V2X synthetic. This synthesizer ray-casts a boxy
TrashCan against the OS1-64 sensor model at real positions extracted from
legal train yamls.

Output: per-frame .npy (Nx3 float32 point deltas) + .txt label rows,
stored as deltas under trashcan_deltas/. Merged with scene points and
training labels at training-prep time (TrashCan maps to 'truck' via
SUPER_CLASS_MAP).

Run: .venv/bin/python src/generate_trashcan_deltas_v2x.py [start] [end]
"""
from __future__ import annotations

import json, sys, numpy as np
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PCD = ROOT / "data/ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1"
OUT = PCD / "trashcan_deltas"
STATS = ROOT / "tracks/v2x_real/trashcan_stats.npz"
SENSOR = ROOT / "tracks/v2x_real/v2x_sensor.json"

RANGE_NOISE = 0.02  # m, 1 sigma
AZ_STEP = 2 * np.pi / 1024  # OS1-64 1024-column mode
LO = np.array([-40, -40, -8], dtype=np.float64)
HI = np.array([40, 40, 2], dtype=np.float64)

_beams = None
_positions = None
_dims = None
_yaws = None


def _init():
    global _beams, _positions, _dims, _yaws
    model = json.load(open(SENSOR))
    _beams = np.array(model["beam_elevations_rad"], dtype=np.float64)
    d = np.load(STATS)
    _positions = d["centers"]  # (N, 3) lidar-frame positions
    _dims = d["dims"]  # (N, 3) full dims dx, dy, dz
    _yaws = d["yaws"]  # (N,) yaw in lidar frame


def ray_box_intersect(ray_origin, ray_dir, box_center, box_dims, box_yaw):
    """Ray-box intersection (slab method). Returns t (distance) or None."""
    # Transform ray to box frame
    c_y = np.cos(-box_yaw)
    s_y = np.sin(-box_yaw)
    d = ray_origin - box_center
    ro_local = np.array([c_y * d[0] - s_y * d[1], s_y * d[0] + c_y * d[1], d[2]])
    rd_local = np.array([c_y * ray_dir[0] - s_y * ray_dir[1], s_y * ray_dir[0] + c_y * ray_dir[1], ray_dir[2]])

    half = box_dims / 2
    inv_d = np.where(np.abs(rd_local) > 1e-10, 1.0 / rd_local, 1e10)
    t1 = (-half - ro_local) * inv_d
    t2 = (half - ro_local) * inv_d
    tmin = np.minimum(t1, t2).max()
    tmax = np.maximum(t1, t2).min()
    if tmax < 0 or tmin > tmax or tmin < 0:
        return None
    return tmin


def synthesize_frame(scene_pts, rng, n_trashcans=1):
    """Place n_trashcans and ray-cast against OS1-64 model."""
    if _beams is None:
        _init()

    out_pts = []
    out_labels = []

    # Sample TrashCan positions (from real prior)
    indices = rng.choice(len(_positions), size=min(n_trashcans, len(_positions)), replace=False)

    for idx in indices:
        center = _positions[idx].copy()
        dims = _dims[idx].copy()
        yaw = _yaws[idx]

        # Add small position jitter (±0.5m)
        center[:2] += rng.uniform(-0.5, 0.5, size=2)

        # Check if in range
        if np.any(center < LO) or np.any(center > HI):
            continue

        # Ground z from local scene
        mask = (np.abs(scene_pts[:, 0] - center[0]) < 5) & (np.abs(scene_pts[:, 1] - center[1]) < 5)
        if mask.sum() > 10:
            ground_z = np.percentile(scene_pts[mask, 2], 10)
            center[2] = ground_z + dims[2] / 2  # bottom-anchored
        else:
            continue  # no ground support, skip

        # Ray-cast: for each azimuth × elevation beam
        n_az = 1024
        azs = np.arange(n_az) * AZ_STEP - np.pi
        for az in azs:
            for el in _beams:
                # Ray from origin (sensor at [0,0,0])
                ray_dir = np.array([
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el)
                ])
                t = ray_box_intersect(np.zeros(3), ray_dir, center, dims, yaw)
                if t is not None and t > 0.5 and t < 80:
                    pt = ray_dir * t
                    pt += np.random.normal(0, RANGE_NOISE, 3)
                    # Check not inside existing scene
                    out_pts.append(pt)

        if len(out_pts) == 0:
            continue

        # Label: x y z dx dy dz heading class
        label = f"{center[0]:.4f} {center[1]:.4f} {center[2]:.4f} {dims[0]:.4f} {dims[1]:.4f} {dims[2]:.4f} {yaw:.4f} truck"
        out_labels.append(label)

    return np.array(out_pts, dtype=np.float32), out_labels


def process_frame(args):
    idx, total = args
    if _beams is None:
        _init()
    rng = np.random.default_rng(920000 + idx)
    npy_path = PCD / "pcd" / f"{idx:06d}.npy"
    if not npy_path.exists():
        return None
    scene = np.load(npy_path).astype(np.float32)
    pts, labels = synthesize_frame(scene, rng, n_trashcans=1)
    if len(pts) == 0:
        return idx, None, None
    return idx, pts, labels


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    OUT.mkdir(exist_ok=True)

    args = [(i, end - start) for i in range(start, end)]
    with Pool(processes=8) as pool:
        for idx, pts, labels in pool.imap_unordered(process_frame, args):
            if pts is None:
                continue
            np.save(OUT / f"{idx:06d}.npy", pts)
            (OUT / f"{idx:06d}.txt").write_text("\n".join(labels) + "\n")
            if idx % 1000 == 0:
                print(f"frame {idx}: {len(pts)} trashcan points")

    # Stats
    n = sum(1 for f in OUT.glob("*.npy"))
    total_pts = sum(np.load(f).shape[0] for f in OUT.glob("*.npy"))
    print(f"\nDone: {n} frames, {total_pts} total points, {total_pts/max(n,1):.0f} pts/frame")


if __name__ == "__main__":
    main()
