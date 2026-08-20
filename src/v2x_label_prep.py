"""V2X training-label prep: rider rescale + pedestrian merge.

Two operations per UT-V2X label file (output: lbl_prepped/):
  1. BicycleRider / MotorcyleRider boxes are rescaled to real train-split
     per-type medians (median-ratio, bottom-anchored -- same method as
     src/label_surgery.py for LUMPI). NOT shrunk to person size: real rider
     boxes are genuinely large (measured 2026-07-20: real BicycleRider
     1.81x0.77x1.72, MotorcyleRider 1.63x0.80x1.70; the "0.76m" figure was
     the true-Pedestrian median, a different sub-population of the class).
  2. Pedestrian label rows from ped_deltas/ are appended, so the VM training
     prep only has to merge POINTS (pcd + ped_deltas npy) and can consume
     lbl_prepped/ verbatim.

Legal domain adaptation: only our own synthetic labels are modified; real
statistics come from guard-filtered public train-split yamls.

Run: .venv/bin/python src/v2x_label_prep.py
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PCD = ROOT / "data/ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1"
SRC = PCD / "lbl"
DST = PCD / "lbl_prepped"
DELTAS = PCD / "ped_deltas"

# real per-type median dims (l, w, h) from guard-filtered train yamls,
# measured 2026-07-20 (BicycleRider n=2015, MotorcyleRider n=545)
REAL = {
    "BicycleRider": (1.81, 0.77, 1.72),
    "MotorcyleRider": (1.63, 0.80, 1.70),
}


def synthetic_medians() -> dict:
    vals = {c: [] for c in REAL}
    for lp in sorted(SRC.glob("*.txt")):
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) >= 8 and f[7] in vals:
                vals[f[7]].append([float(f[3]), float(f[4]), float(f[5])])
    return {c: tuple(np.median(np.array(v), axis=0)) for c, v in vals.items()}


def main() -> None:
    syn = synthetic_medians()
    scale = {c: tuple(r / s for r, s in zip(REAL[c], syn[c])) for c in REAL}
    print("synthetic medians:", {c: tuple(round(v, 2) for v in s) for c, s in syn.items()})
    print("scales:", {c: tuple(round(v, 3) for v in s) for c, s in scale.items()})

    DST.mkdir(exist_ok=True)
    n_boxes = n_scaled = n_peds = 0
    for lp in sorted(SRC.glob("*.txt")):
        out = []
        for line in lp.read_text().splitlines():
            f = line.split()
            if len(f) < 8:
                continue
            n_boxes += 1
            cls = f[7]
            if cls in scale:
                sx, sy, sz = scale[cls]
                x, y, z, dx, dy, dz, hd = (float(v) for v in f[:7])
                ndx, ndy, ndz = dx * sx, dy * sy, dz * sz
                z = z - dz / 2 + ndz / 2  # keep bottom fixed
                f[:7] = [f"{v:.4f}" for v in (x, y, z, ndx, ndy, ndz, hd)]
                n_scaled += 1
            out.append(" ".join(f))
        ped_file = DELTAS / lp.name
        if ped_file.exists():
            ped_rows = [ln for ln in ped_file.read_text().splitlines() if ln.strip()]
            out.extend(ped_rows)
            n_peds += len(ped_rows)
        (DST / lp.name).write_text("\n".join(out) + ("\n" if out else ""))
    print(f"rewrote {n_boxes} boxes ({n_scaled} riders rescaled, "
          f"{n_peds} pedestrians appended) into {DST}")


if __name__ == "__main__":
    main()
