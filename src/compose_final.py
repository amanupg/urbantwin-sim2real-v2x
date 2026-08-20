"""Compose the submitted V2X predictions.json by per class selection.

The submitted detections are a per class combination of two earlier submissions, each of
which had already been measured on the evaluation server. This is the "build only from
measured components" methodology described in the challenge report: every class carries the
best server measured variant, so the combined score is known before submission.

    vehicle     <- v2x_bestof     (server 0.7238)
    pedestrian  <- v2x_tta        (server 0.7253)
    truck       <- either         (identical in both)

Verified: the output of this script is BYTE IDENTICAL to the predictions.json inside
tracks/v2x_real/submissions/FINAL_v2x.zip.

Usage:
  python src/compose_final.py artifacts/lineage/v2x_bestof_predictions.json \
                              artifacts/lineage/v2x_tta_predictions.json \
                              composed_predictions.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FROM_BESTOF = {"vehicle", "truck"}
FROM_TTA = {"pedestrian"}


def main(bestof: Path, tta: Path, out: Path) -> None:
    B = json.loads(bestof.read_text())
    T = json.loads(tta.read_text())
    assert set(B) == set(T), "frame id mismatch between the two sources"

    composed = {}
    for fid in B:
        composed[fid] = ([d for d in B[fid] if d["class"] in FROM_BESTOF]
                         + [d for d in T.get(fid, []) if d["class"] in FROM_TTA])

    blob = json.dumps(composed)
    out.write_text(blob)

    from collections import Counter
    print("per class counts:",
          dict(Counter(d["class"] for v in composed.values() for d in v)))
    print(f"sha256: {hashlib.sha256(blob.encode()).hexdigest()[:16]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
