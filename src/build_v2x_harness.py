"""Build the V2X-Real validation harness (legal train frames, guard-checked).

GT comes from the OPV2V-format yamls of the infrastructure agent (-1). Boxes
are stored in world coordinates; we transform into the agent lidar frame via
lidar_pose ([x,y,z,roll,yaw,pitch] degrees, OpenCOOD convention) and map raw
object types through the official SUPER_CLASS_MAP. A point-containment
self-check validates the transform per frame.

Output: tracks/v2x_real/val_harness/{frames/*.bin, detection_gt.json, frame_ids.json}
Run: .venv/bin/python src/build_v2x_harness.py
"""
from __future__ import annotations

import json
import sys
import numpy as np
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import forbidden_guard as fg

LO = np.array([-40, -40, -8], dtype=np.float64)
HI = np.array([40, 40, 2], dtype=np.float64)
N_FRAMES = 100

SUPER = {
    "Car": "vehicle", "PoliceCar": "vehicle", "LongVehicle": "vehicle",
    "Child": "pedestrian", "RoadWorker": "pedestrian", "Pedestrian": "pedestrian",
    "Scooter": "pedestrian", "ScooterRider": "pedestrian", "Motorcycle": "pedestrian",
    "MotorcyleRider": "pedestrian", "BicycleRider": "pedestrian", "Bicycle": "pedestrian",
    "Truck": "truck", "Van": "truck", "TrashCan": "truck",
    "ConcreteTruck": "truck", "Bus": "truck", "FireHydrant": "truck",
    "ConstructionCart": "truck",
}


def pose_to_matrix(pose):
    # EXACT replica of OpenCOOD's x_to_world (nonstandard signs; do not
    # substitute scipy euler conventions - pitch sign differs)
    x, y, z, roll, yaw, pitch = pose
    c_y, s_y = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    c_r, s_r = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    c_p, s_p = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    T[0, 0] = c_p * c_y
    T[0, 1] = c_y * s_p * s_r - s_y * c_r
    T[0, 2] = -c_y * s_p * c_r - s_y * s_r
    T[1, 0] = s_y * c_p
    T[1, 1] = s_y * s_p * s_r + c_y * c_r
    T[1, 2] = -s_y * s_p * c_r + c_y * s_r
    T[2, 0] = s_p
    T[2, 1] = -c_p * s_r
    T[2, 2] = c_p * c_r
    return T


def boxes_from_yaml(y, world_to_lidar):
    out = []
    for _, v in y.get("vehicles", {}).items():
        t = str(v.get("obj_type", ""))
        cls = SUPER.get(t)
        if cls is None:
            continue
        loc = np.array(v["location"], dtype=np.float64)
        center = np.array(v.get("center", [0, 0, 0]), dtype=np.float64)
        ext = np.array(v["extent"], dtype=np.float64)  # half dims l,w,h
        ang = v.get("angle", [0, 0, 0])                # roll, yaw, pitch (deg)
        cw = loc + center
        c_h = world_to_lidar @ np.append(cw, 1.0)
        # heading: rotate the object's world heading direction into lidar frame
        yw = np.deg2rad(ang[1])
        hvec = world_to_lidar[:3, :3] @ np.array([np.cos(yw), np.sin(yw), 0.0])
        yaw_lidar = float(np.arctan2(hvec[1], hvec[0]))
        out.append({
            "box": [float(c_h[0]), float(c_h[1]), float(c_h[2]),
                    float(2 * ext[0]), float(2 * ext[1]), float(2 * ext[2]),
                    float(yaw_lidar)],
            "class": cls,
        })
    return out


def containment(pts, boxes):
    inside = 0
    for b in boxes:
        cx, cy, cz, l, w, h, hd = b["box"]
        q = pts - [cx, cy, cz]
        c, s = np.cos(-hd), np.sin(-hd)
        x = q[:, 0] * c - q[:, 1] * s
        yy = q[:, 0] * s + q[:, 1] * c
        inside += int((((np.abs(x) <= l / 2 + 0.1) & (np.abs(yy) <= w / 2 + 0.1)
                        & (np.abs(q[:, 2]) <= h / 2 + 0.2))).sum())
    return inside


def main() -> None:
    mapping = json.load(open(ROOT / "tracks/v2x_real/id_mapping.json"))
    scen_map = mapping["scenario_map"]
    root = ROOT / "data/v2x_real/train"

    rng = np.random.default_rng(47)
    out = ROOT / "tracks/v2x_real/val_harness"
    (out / "frames").mkdir(parents=True, exist_ok=True)
    gt, ids, checks = {}, [], []

    prefixes = list(scen_map.keys())
    while len(ids) < N_FRAMES:
        pfx = prefixes[int(rng.integers(len(prefixes)))]
        scen = scen_map[pfx]
        adir = root / scen / "-1"
        bins = sorted(adir.glob("*.bin"))
        f = bins[int(rng.integers(len(bins)))]
        cid = f"{pfx}_{f.stem}"
        if fg.is_forbidden(cid, "v2x_real") or cid in gt:
            continue
        y = yaml.safe_load((adir / f"{f.stem}.yaml").read_text())
        lidar_pose = y["lidar_pose"]
        world_to_lidar = np.linalg.inv(pose_to_matrix(lidar_pose))
        boxes = boxes_from_yaml(y, world_to_lidar)
        boxes = [b for b in boxes
                 if (LO <= np.array(b["box"][:3])).all() and (np.array(b["box"][:3]) <= HI).all()]
        if not boxes:
            continue
        pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)[:, :3].astype(np.float64)
        pts = pts[((pts >= LO) & (pts <= HI)).all(1)]
        checks.append(containment(pts, boxes) / max(1, len(boxes)))
        pts.astype(np.float32).tofile(out / "frames" / f"{cid}.bin")
        gt[cid] = boxes
        ids.append(cid)

    (out / "detection_gt.json").write_text(json.dumps(gt))
    (out / "frame_ids.json").write_text(json.dumps(ids))
    from collections import Counter
    cc = Counter(b["class"] for v in gt.values() for b in v)
    print(f"harness: {len(ids)} frames, {sum(len(v) for v in gt.values())} GT; {dict(cc)}")
    print(f"containment self-check: mean pts/box {np.mean(checks):.0f} "
          f"(should be >> 10; near 0 means transform convention is wrong)")


if __name__ == "__main__":
    main()
