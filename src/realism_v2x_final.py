"""V2X realism optimizer: null-return modelling + z-align + density match.

Root cause found 2026-07-26: real V2X frames (and the organizers' own released
detection-test frames) encode dropped returns as the literal point (0,0,0) --
31% of every frame. Our synthetic frames had none, so ~a third of the reference
distribution's mass had no counterpart and EMD (86% of the realism headroom)
was pinned near 1.94 against a real-vs-real floor of 0.765.

Fixes, in order of impact:
  1. inject null returns (rate swept; the score-optimal rate is below the true
     31% because our non-null distribution still differs, and over-injecting
     costs FPD)
  2. global z-shift (real median z -0.39 vs synthetic -1.83)
  3. sliced-Wasserstein frame selection + fine-grained density resampling

Run: .venv/bin/python src/realism_v2x_final.py [n_candidates]
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tracks/v2x_real/starter_kit/scoring_program"))
import metrics

RB = [-40, -40, -8, 40, 40, 2]
LO, HI = np.array(RB[:3]), np.array(RB[3:])
NORM = {"CD": (2, 30), "MMD": (0, 0.5), "EMD": (0, 5), "FPD": (0, 50)}
PCD = ROOT / "data/ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1/pcd"
PED = ROOT / "data/ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1/ped_deltas_v2"
OUT = ROOT / "tracks/v2x_real/emd_frames"
N_SEL, N_TOT = 50, 50_000
VOX = np.array([0.6, 0.6, 0.4])


def realism_of(r):
    return float(np.mean([1 - min(max((r[k] - lo) / (hi - lo), 0), 1)
                          for k, (lo, hi) in NORM.items()]))


def sliced_w1(A, B, n_proj=64, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n_proj, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = min(len(A), len(B))
    ia = np.linspace(0, len(A) - 1, n).astype(int)
    ib = np.linspace(0, len(B) - 1, n).astype(int)
    return float(np.abs(np.sort(A @ d.T, 0)[ia] - np.sort(B @ d.T, 0)[ib]).mean())


def load_real(a, b):
    fs = sorted((ROOT / "tracks/v2x_real/val_harness/frames").glob("*.bin"))[a:b]
    return [np.fromfile(f, dtype=np.float32).reshape(-1, 3) for f in fs]


def main(n_cand=300):
    build_ref, holdout = load_real(0, 50), load_real(50, 100)
    ref_pts = np.concatenate([r.astype(np.float64) for r in build_ref])
    ref_nn = ref_pts[np.hypot(ref_pts[:, 0], ref_pts[:, 1]) > 0.01]   # non-null part
    rs = np.random.default_rng(3)
    ref_sub = ref_nn[rs.choice(len(ref_nn), 40000, replace=False)]
    print(f"reference: {len(build_ref)} build / {len(holdout)} holdout frames")

    files = sorted(PCD.glob("*.npy"))
    pick = np.random.default_rng(5).choice(len(files), min(n_cand, len(files)), replace=False)
    cands = []
    for i in pick:
        f = files[i]
        p = np.load(f).astype(np.float64)
        pf = PED / f"{f.stem}.npy"
        if pf.exists():
            e = np.load(pf).astype(np.float64)
            if len(e):
                p = np.concatenate([p, e])
        p[:, 2] += 0.6                                   # z-align
        p = p[((p >= LO) & (p <= HI)).all(1)]
        if len(p) > 5000:
            cands.append(p)
    print(f"candidates: {len(cands)} (z-aligned +0.6)")

    # greedy frame selection on sliced-W1 against the real non-null distribution
    sel, pool = [], None
    for step in range(N_SEL):
        rs2 = np.random.default_rng(step)
        trial = [j for j in range(len(cands)) if j not in sel]
        if len(trial) > 70:
            trial = list(rs2.choice(trial, 70, replace=False))
        best, bs = None, np.inf
        for j in trial:
            c = cands[j] if pool is None else np.concatenate([pool, cands[j]])
            sub = c[rs2.choice(len(c), min(12000, len(c)), replace=False)]
            s = sliced_w1(sub, ref_sub, 40, step)
            if s < bs:
                best, bs = j, s
        sel.append(best)
        pool = cands[best] if pool is None else np.concatenate([pool, cands[best]])
    print(f"selected 50 frames, final slicedW1={bs:.4f}")

    frames = [cands[j] for j in sel]

    # fine density resampling toward the real non-null density
    dims = np.ceil((HI - LO) / VOX).astype(int); nvox = int(np.prod(dims))
    def gidx(p):
        ij = np.clip(np.floor((p - LO) / VOX).astype(int), 0, dims - 1)
        return np.ravel_multi_index(ij.T, dims)
    h_ref = np.bincount(gidx(ref_nn), minlength=nvox).astype(float); h_ref /= h_ref.sum()

    def assemble(frames_in, nf, cap):
        h_cur = np.bincount(gidx(np.concatenate(frames_in)), minlength=nvox).astype(float)
        h_cur /= h_cur.sum()
        w = np.clip((h_ref + 1e-9) / (h_cur + 1e-9), 1.0 / cap, cap)
        out, rng = [], np.random.default_rng(21)
        n_null = int(N_TOT * nf); n_real = N_TOT - n_null
        for k, f in enumerate(frames_in):
            p = w[gidx(f)]; p = p / p.sum()
            keep = rng.choice(len(f), min(n_real, len(f)), replace=False, p=p)
            out.append(np.concatenate([f[keep], np.zeros((n_null, 3))]).astype(np.float32))
        return out

    best = (None, -1, "")
    for cap in (1.0, 2.5, 5.0):          # cap 1.0 == no resampling (control)
        for nf in (0.18, 0.20, 0.23):
            cl = assemble(frames, nf, cap)
            r = metrics.compute_realism(cl, holdout, RB, seed=1234)
            s = realism_of(r)
            print(f"  cap={cap:<4} null={nf:.2f}: EMD {r['EMD']:.3f} CD {r['CD']:.3f} "
                  f"MMD {r['MMD']:.5f} FPD {r['FPD']:.2f} -> {s:.4f}")
            if s > best[1]:
                best = (cl, s, f"cap{cap}_null{nf}")
    cl, s, tag = best
    # cross-validate on the OTHER disjoint real set (guards against fitting one draw)
    r2 = metrics.compute_realism(cl, build_ref, RB, seed=1234)
    print(f"BEST {tag}: holdout {s:.4f} | cross-val {realism_of(r2):.4f} "
          f"(baseline 0.8800, real-vs-real ceiling 0.9612)")

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.bin"):
        old.unlink()
    for i, c in enumerate(cl, start=1):
        c.tofile(OUT / f"{i:04d}.bin")
    sz = [f.stat().st_size for f in OUT.glob("*.bin")]
    print(f"wrote 50 frames -> {OUT} ({min(sz)/1e6:.2f}-{max(sz)/1e6:.2f} MB)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
