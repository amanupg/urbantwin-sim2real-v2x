"""Generate pedestrian injection deltas for all UT-V2X frames.

For each frame: synthesized pedestrian points (Nx3 float32 .npy) and
Pedestrian label rows (.txt), stored as deltas under ped_deltas/. Merged with
the scene points and training labels at training-prep time on the VM
(Pedestrian maps to the 'pedestrian' super-class via SUPER_CLASS_MAP).

Run: .venv/bin/python src/generate_ped_deltas_v2x.py [start] [end]
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pedestrian_synth_v2x import load_beams, synthesize_frame

PCD = ROOT / "data/ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1"
OUT = PCD / (sys.argv[3] if len(sys.argv) > 3 else "ped_deltas")

_beams = None
_prior = None


def _init():
    global _beams, _prior
    _beams = load_beams()
    d = np.load(ROOT / "tracks/v2x_real/ped_stats.npz")
    _prior = (d["pos"], d["dims"], d["per_frame_counts"])


def one(name: str):
    out_pts = OUT / f"{name}.npy"
    out_lbl = OUT / f"{name}.txt"
    if out_pts.exists() and out_lbl.exists():
        return 0
    rng = np.random.default_rng(910_000 + int(name))
    scene = np.load(PCD / "pcd" / f"{name}.npy").astype(np.float64)
    labels = (PCD / "lbl" / f"{name}.txt").read_text().splitlines()
    pts_list, rows = synthesize_frame(scene, labels, _beams, _prior[0],
                                      _prior[1], _prior[2], rng)
    pts = (np.concatenate(pts_list) if pts_list else np.zeros((0, 3), dtype=np.float32))
    np.save(out_pts, pts.astype(np.float32))
    out_lbl.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(rows)


def main():
    OUT.mkdir(exist_ok=True)
    names = sorted(p.stem for p in (PCD / "pcd").glob("*.npy"))
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else len(names)
    names = names[lo:hi]
    with Pool(8, initializer=_init) as pool:
        counts = []
        for i, c in enumerate(pool.imap_unordered(one, names, chunksize=20)):
            counts.append(c)
            if (i + 1) % 1000 == 0:
                print(f"{i+1}/{len(names)}", flush=True)
    print(f"done: {len(counts)} frames, {sum(counts)} pedestrians injected")


if __name__ == "__main__":
    main()
