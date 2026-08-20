"""Audit the shipped V2X post-processing stack, stage by stage.

Mirrors src/audit_stack.py. V2X has 2.3x the per-class leverage of LUMPI (mAP
divides by 3, not 7), so a point of AP here is worth 2.3 points there.

CAVEAT vs the LUMPI audit: LUMPI has a genuinely held-out late slice. V2X does
NOT -- val_harness is the only labelled real split, and the shipped stack was
partly fitted on it. So a stage that looks good here is NOT independently
validated the way keep-both was. Treat these as directional, and prefer
MECHANISM changes (which transferred) over refitted PARAMETERS (which did not).

The specific question this run answers: does V2X have a destructive relabelling
rule like LUMPI's Van fill-miss, which cost 0.14 Van AP by deleting predictions?
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tracks/v2x_real/starter_kit/scoring_program"))
from detection_eval import evaluate_3d_detection, _bev_iou  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from forbidden_guard import filter_allowed  # noqa: E402

VAL = ROOT / "tracks/v2x_real/val_harness"
CLASSES = ["vehicle", "pedestrian", "truck"]

_pc: dict[str, np.ndarray | None] = {}


def pts(fid):
    if fid not in _pc:
        f = VAL / f"frames/{fid}.bin"
        if f.exists():
            a = np.fromfile(f, dtype=np.float32)
            n = 4 if a.size % 4 == 0 else 3
            _pc[fid] = a.reshape(-1, n)[:, :3].astype(np.float64)
        else:
            _pc[fid] = None
    return _pc[fid]


def interior(P, box, pad=0.0):
    cx, cy, cz, L, W, H, yaw = box
    d = P - np.array([cx, cy, cz])
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = d[:, 0] * c - d[:, 1] * s
    y = d[:, 0] * s + d[:, 1] * c
    return P[(np.abs(x) <= L / 2 + pad) & (np.abs(y) <= W / 2 + pad)
             & (np.abs(d[:, 2]) <= H / 2 + pad)]


def ev(preds, gt):
    r = evaluate_3d_detection(preds, gt, CLASSES, iou_thresholds=(0.5, 0.7),
                              iou_type="3d")
    ap = {}
    for c in CLASSES:
        v = r["per_class"].get(c, {}).get("AP_0.5_R40", float("nan"))
        ap[c] = 0.0 if v != v else float(v)
    return float(np.mean([ap[c] for c in CLASSES])), ap


def show(tag, m, ap, base=None):
    cells = []
    for c in CLASSES:
        s = f"{ap[c]:.4f}"
        if base is not None and abs(ap[c] - base[c]) >= 0.0005:
            s += f"({ap[c]-base[c]:+.3f})"
        cells.append(f"{c[:4]} {s}")
    print(f"{tag:30s} mAP3={m:.4f}  " + "  ".join(cells))


# ------------------------------------------------------------------- stages
def s_thresh(preds, t):
    return {f: [d for d in v if d["score"] >= t] for f, v in preds.items()}


def s_pedrefiner(preds):
    """Pedestrian geometric re-centring -- the V2X analogue of LUMPI's Person
    refiner (shipped since v2x_run9_hybrid_pedrefiner)."""
    out = {}
    for fid, dets in preds.items():
        P = pts(fid)
        o = []
        for d in dets:
            e = dict(d)
            if d["class"] == "pedestrian" and P is not None:
                q = interior(P, d["box"], 0.02)
                if len(q) >= 3:
                    b = list(map(float, d["box"]))
                    b[0], b[1] = float(q[:, 0].mean()), float(q[:, 1].mean())
                    e["box"] = b
            o.append(e)
        out[fid] = o
    return out


def s_trucklen(preds, k=1.154):
    return {f: [({**x, "box": [*x["box"][:3], x["box"][3] * k, *x["box"][4:]]}
                 if x["class"] == "truck" else x) for x in v]
            for f, v in preds.items()}


def s_nms(preds, thr_by_class):
    out = {}
    for fid, dets in preds.items():
        keep = []
        by_c = collections.defaultdict(list)
        for d in dets:
            by_c[d["class"]].append(d)
        for c, ds in by_c.items():
            thr = thr_by_class.get(c)
            if thr is None or thr >= 1.0:
                keep.extend(ds)
                continue
            ds = sorted(ds, key=lambda d: -d["score"])
            sel = []
            for d in ds:
                b = np.asarray(d["box"])
                if all(_bev_iou(b, np.asarray(s["box"])) <= thr for s in sel):
                    sel.append(d)
            keep.extend(sel)
        out[fid] = keep
    return out


def s_rerank(preds, betas, floor=1e-6):
    out = {}
    for fid, dets in preds.items():
        P = pts(fid)
        o = []
        for x in dets:
            b = betas.get(x["class"], 0.0)
            if b > 0 and P is not None:
                n = len(interior(P, x["box"], 0.0))
                q = np.log1p(n) / np.log1p(40.0)
                o.append({**x, "score": float(x["score"] * max(1 - b + b * q, floor))})
            else:
                o.append(x)
        out[fid] = o
    return out


SHIPPED_NMS = {"pedestrian": 0.7, "vehicle": 0.3, "truck": 0.3}
SHIPPED_BETAS = {"pedestrian": 0.7, "vehicle": 0.2}


def main():
    gt = json.loads((VAL / "detection_gt.json").read_text())
    pr = json.loads((VAL / "predictions_v2x_val_run9_lt01.json").read_text())
    ids = filter_allowed(sorted(set(gt) & set(pr)), "v2x_real")
    gt = {i: gt[i] for i in ids}
    pr = {i: pr[i] for i in ids}
    print(f"V2X val harness: {len(ids)} allowed frames")
    print("GT:", {c: sum(1 for i in gt for g in gt[i] if g["class"] == c)
                  for c in CLASSES})
    print("pred:", {c: sum(1 for i in pr for p in pr[i] if p["class"] == c)
                    for c in CLASSES})
    print()

    # v23's lesson: a lower score threshold HURT on LUMPI. Test it here.
    base = s_thresh(pr, 0.1)
    m0, ap0 = ev(base, gt)
    show("run9 @ thresh 0.1 (base)", m0, ap0)
    for t in (0.01, 0.05, 0.2, 0.3):
        m, ap = ev(s_thresh(pr, t), gt)
        show(f"  thresh {t}", m, ap, ap0)

    print("\n--- each shipped stage applied ALONE ---")
    for name, fn in [("+ ped refiner", s_pedrefiner),
                     ("+ truck len x1.154", s_trucklen),
                     ("+ per-class NMS", lambda p: s_nms(p, SHIPPED_NMS)),
                     ("+ re-rank", lambda p: s_rerank(p, SHIPPED_BETAS))]:
        m, ap = ev(fn(base), gt)
        show(name, m, ap, ap0)

    print("\n--- cumulative, shipped order ---")
    cur = s_rerank(s_nms(s_trucklen(s_pedrefiner(base)), SHIPPED_NMS), SHIPPED_BETAS)
    m_ship, ap_ship = ev(cur, gt)
    show("full shipped stack", m_ship, ap_ship, ap0)

    print("\n--- LEAVE-ONE-OUT: which stage is a net negative? ---")
    stages = {"ped refiner": s_pedrefiner,
              "truck len": s_trucklen,
              "per-class NMS": lambda p: s_nms(p, SHIPPED_NMS),
              "re-rank": lambda p: s_rerank(p, SHIPPED_BETAS)}
    for drop in stages:
        cur = base
        for name, fn in stages.items():
            if name != drop:
                cur = fn(cur)
        m, ap = ev(cur, gt)
        d = m - m_ship
        flag = "  <<< DROPPING THIS HELPS" if d > 0.0005 else ""
        show(f"  without {drop}", m, ap, ap_ship)
        print(f"      mAP3 {m_ship:.4f} -> {m:.4f} ({d:+.4f}){flag}")


if __name__ == "__main__":
    main()
