"""Pedestrian synthesizer for UT-V2X frames (port of the LUMPI machinery).

Differences from the LUMPI version (src/pedestrian_synth.py):
  - ONE sensor, sitting at the frame ORIGIN (V2X bins are agent-lidar-frame
    scans: OS1-64, 64 beams measured from real frames, azimuth step 2pi/1024).
    No trilateration needed -- the sensor's own frame is the data frame.
  - Placement prior + dims from real train-split Pedestrian boxes
    (tracks/v2x_real/ped_stats.npz, guard-filtered).
  - Per-frame count bootstrapped from the real empirical distribution.
  - Ground height MUST come from the local scene (V2X ground is tilted in the
    lidar frame -- a global fallback z would float/bury pedestrians), so
    placements without local ground support are rejected outright.
  - Scene-occupancy rejection: the real prior mixes 48 scenarios' walkable
    areas (sensor-relative), so a candidate spot that already contains scene
    structure (wall, foliage, pole) is rejected before ray-casting.

Validation gate: reproduce the real points-per-pedestrian-vs-range curve
(tracks/v2x_real/v2x_ped_pointcurve.json) before any training use.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

AZ_STEP = 2 * np.pi / 1024     # OS1-64 1024-column mode (measured 0.351 deg)
RANGE_NOISE = 0.02             # m, 1 sigma
KEEP_PROB = (0.46, 0.58, 15.0)  # (near, far, split_m): calibrated vs real curve
DEPTH_AZ_BIN = np.deg2rad(0.4)
DEPTH_EL_BIN = np.deg2rad(0.8)
OCCLUSION_MARGIN = 0.6         # m
MIN_POINTS_VISIBLE = 5         # placements yielding fewer points are resampled
OCCUPANCY_MAX = 8              # scene points allowed inside the capsule volume


def load_beams(path=ROOT / "tracks/v2x_real/v2x_sensor.json") -> np.ndarray:
    model = json.load(open(path))
    return np.array(model["beam_elevations_rad"], dtype=np.float64)


def _spherical(pts):
    rng = np.linalg.norm(pts, axis=1)
    az = np.arctan2(pts[:, 1], pts[:, 0])
    el = np.arcsin(np.clip(pts[:, 2] / np.maximum(rng, 1e-9), -1, 1))
    return az, el, rng


def _depth_map(pts):
    az, el, rng = _spherical(pts)
    ai = np.floor((az + np.pi) / DEPTH_AZ_BIN).astype(np.int64)
    ei = np.floor((el + np.pi / 2) / DEPTH_EL_BIN).astype(np.int64)
    key = ai * 1000 + ei
    order = np.argsort(key, kind="stable")
    key_s, rng_s = key[order], rng[order]
    first = np.ones(len(key_s), dtype=bool)
    first[1:] = key_s[1:] != key_s[:-1]
    idx = np.flatnonzero(first)
    dmap = {}
    for i, start in enumerate(idx):
        end = idx[i + 1] if i + 1 < len(idx) else len(key_s)
        dmap[int(key_s[start])] = float(rng_s[start:end].min())
    return dmap


def _depth_at(dmap, az, el):
    ai = int((az + np.pi) / DEPTH_AZ_BIN)
    ei = int((el + np.pi / 2) / DEPTH_EL_BIN)
    return dmap.get(ai * 1000 + ei, np.inf)


def cast_person(beams, dmap, center, dims, rng_state, keep_prob):
    """Ray-cast one elliptic-capsule person from the origin sensor."""
    l, w, h = dims
    rx, ry = l / 2 * 0.8, w / 2 * 0.8   # capsule slightly inside box
    cz_bottom = center[2] - h / 2
    dist = np.hypot(center[0], center[1])
    if dist < 1.0 or dist > 120.0:
        return np.zeros((0, 3))
    az_c = np.arctan2(center[1], center[0])
    half_az = np.arctan2(max(rx, ry), dist) * 1.2
    azs = np.arange(az_c - half_az, az_c + half_az, AZ_STEP)
    el_lo = np.arctan2(cz_bottom, dist)
    el_hi = np.arctan2(cz_bottom + h, dist)
    els = beams[(beams >= el_lo - 0.005) & (beams <= el_hi + 0.005)]
    if len(azs) == 0 or len(els) == 0:
        return np.zeros((0, 3))
    A, E = np.meshgrid(azs, els)
    A, E = A.ravel(), E.ravel()
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], 1)
    # ray-elliptic-cylinder intersection (origin sensor, vertical capsule)
    ox, oy = -center[0], -center[1]
    dx, dy = dirs[:, 0], dirs[:, 1]
    a = (dx / rx) ** 2 + (dy / ry) ** 2
    b = 2 * (ox * dx / rx ** 2 + oy * dy / ry ** 2)
    c = (ox / rx) ** 2 + (oy / ry) ** 2 - 1
    disc = b ** 2 - 4 * a * c
    ok = disc > 0
    t = np.full(len(dirs), np.inf)
    t[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
    zhit = t * dirs[:, 2]
    ok &= (t > 0) & (zhit >= cz_bottom) & (zhit <= cz_bottom + h)
    if not ok.any():
        return np.zeros((0, 3))
    t = t[ok]; dirs = dirs[ok]; A = A[ok]; E = E[ok]
    keep = np.array([
        _depth_at(dmap, a_, e_) > (t_ - OCCLUSION_MARGIN)
        for a_, e_, t_ in zip(A, E, t)
    ])
    t = t[keep]; dirs = dirs[keep]
    if isinstance(keep_prob, tuple):
        lo_kp, hi_kp, split = keep_prob
        probs = np.where(t < split, lo_kp, hi_kp)
        drop = rng_state.random(len(t)) < probs
    else:
        drop = rng_state.random(len(t)) < keep_prob
    t = t[drop]; dirs = dirs[drop]
    t = t + rng_state.normal(0, RANGE_NOISE, len(t))
    return t[:, None] * dirs


def ground_z(scene, x, y):
    m = (np.abs(scene[:, 0] - x) < 1.2) & (np.abs(scene[:, 1] - y) < 1.2)
    if m.sum() < 5:
        return None
    return float(np.percentile(scene[m, 2], 5))


def _occupied(scene, x, y, gz, dims):
    """Scene structure inside the capsule volume -> reject (wall, pole, bush)."""
    r = max(dims[0], dims[1]) / 2 * 0.8
    m = ((scene[:, 0] - x) ** 2 + (scene[:, 1] - y) ** 2 < r * r)
    if not m.any():
        return False
    z = scene[m, 2]
    return int(((z > gz + 0.25) & (z < gz + dims[2])).sum()) > OCCUPANCY_MAX


def synthesize_frame(scene_pts, labels, beams, prior_pos, prior_dims,
                     counts_dist, rng, n_persons=None, keep_prob=KEEP_PROB,
                     group_frac=0.30):
    """Returns (new_points list, new_label_rows list).

    v2: ~30% of placements are GROUPS (2-4 people ~1m apart) — real
    pedestrians cluster; v1 placed everyone independently.
    """
    if n_persons is None:
        n_persons = int(counts_dist[int(rng.integers(len(counts_dist)))])
    boxes = [(float(f[0]), float(f[1]), max(float(f[3]), float(f[4])) / 2 + 0.6)
             for f in (ln.split() for ln in labels) if len(f) >= 8]
    clusters = []
    while len(clusters) < n_persons:
        if rng.random() < group_frac and len(clusters) < n_persons - 1:
            clusters.append(int(rng.integers(2, 5)))
        else:
            clusters.append(1)
    dmap = None  # lazy
    out_pts, out_rows = [], []
    tries = 0
    ci = 0
    while ci < len(clusters) and tries < max(n_persons * 8, 40):
        tries += 1
        size = clusters[ci]
        k = int(rng.integers(len(prior_pos)))
        gx, gy = prior_pos[k] + rng.normal(0, 0.8, 2)
        members = []
        for _ in range(size):
            x, y = gx + rng.normal(0, 0.6), gy + rng.normal(0, 0.6)
            if not (-39.0 < x < 39.0 and -39.0 < y < 39.0):
                continue
            if any((x - bx) ** 2 + (y - by) ** 2 < br ** 2 for bx, by, br in boxes):
                continue
            dims = prior_dims[int(rng.integers(len(prior_dims)))] * rng.normal(1.0, 0.05, 3)
            gz = ground_z(scene_pts, x, y)
            if gz is None or not (-10.0 < gz < 2.0):
                continue
            if _occupied(scene_pts, x, y, gz, dims):
                continue
            center = np.array([x, y, gz + dims[2] / 2])
            if dmap is None:
                dmap = _depth_map(scene_pts)
            pts = cast_person(beams, dmap, center, dims, rng, keep_prob)
            if len(pts) < MIN_POINTS_VISIBLE:
                continue
            heading = float(rng.uniform(-np.pi, np.pi))
            members.append((pts.astype(np.float32),
                            f"{x:.4f} {y:.4f} {center[2]:.4f} "
                            f"{dims[0]:.3f} {dims[1]:.3f} {dims[2]:.3f} {heading:.4f} Pedestrian 0"))
            boxes.append((x, y, 0.8))
        if members:
            for p, r in members:
                out_pts.append(p); out_rows.append(r)
            ci += 1
    return out_pts, out_rows
