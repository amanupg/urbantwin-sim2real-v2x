"""Verify the V2X realism chain the same way the LUMPI chain was verified.

src/realism_affine.py is hardcoded to LUMPI paths, so the V2X alpha probes were
produced by some edited variant that no longer exists in src/. This reimplements
the identical transform against the V2X reference and tests whether it reproduces
the shipped frames from the banked ones.

Transform (same as realism_affine.apply):
    x' = (x - mu_syn) * (1 + alpha*(sigma_ref/sigma_syn - 1)) + mu_syn
         + alpha*(mu_ref - mu_syn)

Usage: .venv/bin/python reproducibility/test/verify_v2x_affine.py [src_zip]
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUB = ROOT / "tracks/v2x_real/submissions"
REF_DIR = ROOT / "tracks/v2x_real/reference_data_local/realism_reference"
TARGET = SUB / "probe_V_a25.zip"   # shares all 50 frames with FINAL_v2x.zip


def load_ref(n: int = 50) -> list[np.ndarray]:
    out = []
    for f in sorted(REF_DIR.glob("*.bin"))[:n]:
        a = np.fromfile(f, dtype=np.float32)
        nc = 4 if a.size % 4 == 0 and a.size % 3 != 0 else 3
        out.append(a.reshape(-1, nc)[:, :3].astype(np.float64))
    return out


def syn_frames(z: Path) -> tuple[list[str], list[np.ndarray]]:
    with zipfile.ZipFile(z) as f:
        names = sorted(n for n in f.namelist() if n.startswith("synthetic/"))
        return names, [np.frombuffer(f.read(n), dtype=np.float32)
                       .reshape(-1, 3).astype(np.float64) for n in names]


def hashes(z: Path) -> set[str]:
    with zipfile.ZipFile(z) as f:
        return {hashlib.sha256(f.read(n)).hexdigest()
                for n in f.namelist() if n.startswith("synthetic/")}


def main(src_name: str) -> None:
    src = SUB / src_name
    print(f"source (banked frames): {src.name}")
    print(f"target (shipped frames): {TARGET.name}\n")

    _, syn = syn_frames(src)
    ref = load_ref()
    print(f"loaded {len(syn)} synthetic, {len(ref)} reference frames")

    S = np.concatenate(syn)
    R = np.concatenate(ref)
    mS, sS = S.mean(0), S.std(0)
    mR, sR = R.mean(0), R.std(0)
    print(f"  syn moments mean {np.round(mS,4)}  std {np.round(sS,4)}")
    print(f"  ref moments mean {np.round(mR,4)}  std {np.round(sR,4)}\n")

    tgt = hashes(TARGET)
    best = (None, -1)
    for alpha in (0.0, 0.15, 0.25, 0.40, 0.75, 1.0):
        sc = 1 + alpha * (sR / sS - 1)
        sh = mS + alpha * (mR - mS)
        got = {hashlib.sha256(((f - mS) * sc + sh).astype(np.float32).tobytes())
               .hexdigest() for f in syn}
        n = len(got & tgt)
        print(f"  alpha={alpha:<5}: {n}/50 byte-identical to shipped")
        if n > best[1]:
            best = (alpha, n)

    print()
    if best[1] == 50:
        print(f"RESULT: EXACT -- shipped V2X frames = affine(alpha={best[0]}) of {src.name}")
    else:
        print(f"RESULT: best alpha={best[0]} matched only {best[1]}/50")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "probe_realism_v2x_b.zip")
