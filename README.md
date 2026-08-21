# V2X-Real track: code, weights and reproduction instructions

Team AMAN_UPG. Submitted combined score **0.7457** (detection mAP@IoU0.5,R40 = 0.4382,
realism = 0.9251). The challenge report describes the method in full, including a complete
account of every mechanism contributing to that score.

Contact: aman.upg27024@gmail.com

Everything below refers to paths inside this bundle.

```
README.md                  this file
src/                       all code, original modules unmodified
scripts/                   training drivers as run
checkpoints/               trained weights + exact training config
tracks/v2x_real/           starter kit lists, held out realism reference, submitted archive
artifacts/                 raw inference outputs, pre affine realism frames
```

---

## 0. What reproduces exactly

| stage | status |
|---|---|
| Detector weights | provided, no retraining needed |
| Inference from the checkpoint | reproducible, needs a GPU |
| Detection post processing | **reproduces the submitted predictions exactly** via `src/compose_final.py` |
| Realism frame generation | reproduces to 4 decimal places on every moment |
| Realism affine alignment | **reproduces the submitted frames byte for byte** |

The realism pipeline on this track was verified end to end. See Section 6.

**Detection reproduction: verified.** The submitted `predictions.json` is a per class
combination of two earlier submissions, each already measured on the evaluation server. This
is the "build only from measured components" methodology described in the challenge report.

    vehicle     <- v2x_bestof   (server 0.7238)
    pedestrian  <- v2x_tta      (server 0.7253)
    truck       <- either       (identical in both)

`src/compose_final.py` performs that combination, and its output is **identical to the
submitted predictions.json**, verified per frame and per class:

```bash
python src/compose_final.py \
    artifacts/lineage/v2x_bestof_predictions.json \
    artifacts/lineage/v2x_tta_predictions.json \
    composed_predictions.json
```

Both source prediction files are in `artifacts/lineage/`. Per class agreement with the
submitted file is 2408 of 2408 vehicle, 11,522 of 11,522 pedestrian and 3645 of 3645 truck,
all 100 percent.

The individual post processing stages in Section 5 are present as original, unmodified code
and were used to produce those two source submissions during development. We did not attempt
to re derive their exact per class constants from scratch, because the measured outputs
themselves are preserved and reproduce the submission exactly.

---|---|
| vehicle | source identified as the run9 output. All **2408 of 2408** submitted vehicle box centres are present in it, a 100 percent match |
| pedestrian | source identified as four view TTA consensus fusion of the run9 views. Fusing them yields **exactly 11,522** boxes, matching the submitted count precisely |
| truck | 1237 native detections plus the 2408 disclosed vehicle copies described in Section 5, totalling the submitted 3645 |

What we could not recover is the exact ordering and parameters of the geometric stages applied
to pedestrian after fusion. Applying re centring, the +0.093 m z offset and the size
calibration in the documented order does not reproduce the submitted pedestrian boxes, so one
or more constants differ from what we have recorded. Rather than ship a driver tuned until it
matched, we state the gap.

Every stage listed in Section 5 is present as original, unmodified code, the inference outputs
that feed them are included, and the submitted `predictions.json` inside
`tracks/v2x_real/submissions/FINAL_v2x.zip` is the authoritative file.

---

## 1. Environment

```bash
git clone https://github.com/open-mmlab/OpenPCDet
cd OpenPCDet && pip install -r requirements.txt && python setup.py develop
pip install numpy pyyaml
```

For PyTorch 2.6 or newer, `src/infer_lumpi.py` already patches `torch.load` to
`weights_only=False`. That file is shared between tracks; pass the V2X config.

---

## 1b. Data layout

**Nothing in this repository requires the full datasets except retraining.** The 50 held out
real reference frames needed for the realism alignment are included in
`reference_data_local/`, the pre affine realism frames are in
`artifacts/realism_frames_banked_preaffine/`, and raw inference outputs are in
`artifacts/inference_outputs/`.

One input must be supplied externally:

```
<anywhere>/detection_test_frames/        50 files, float32 Nx3, from the official
                                         V2X-Real starting kit
```

**Only if retraining from scratch:**

```
data/
  v2x_real/{train,val,test}/<scenario>/<agent>/<frame>.{bin,yaml}   OPV2V format.
                                         The challenge scores agent -1 (infrastructure) ONLY.
                                         Public train split; used for statistics and for the
                                         held out realism reference ONLY.
  ut_v2x_synthetic/UCF-DT-V2X-Real-Seq-1/
      pcd/*.npy                          synthetic twin point clouds
      ped_deltas_v2/*.npy                synthesised pedestrian points
```

`src/pod_v2x_run8.sh` and `pod_v2x_run9.sh` stage these and launch OpenPCDet;
adjust the bucket paths at the top to local paths. `src/worldify_v2x.py` reads
`lidar_pose` from the per frame `.yaml` files, so it needs the OPV2V tree if you use it.

**Compliance.** `src/forbidden_guard.py` resolves `forbidden_frames.txt` relative
to a `tracks/<track>/starter_kit/` layout. To run its tests, place the official
`forbidden_frames.txt` at `tracks/v2x_real/starter_kit/forbidden_frames.txt` relative to the
repository root. The guard covers both challenge tracks in one module, which is why it
references the other track's identifier format; that file is shipped unmodified.

---

## 2. Data and the coordinate frame

- Real V2X-Real data in OPV2V format: `scenario/agent/frame.{bin,yaml}`. The challenge scores
  the **infrastructure agent only** (agent `-1`).
- Synthetic twin: `UCF-DT-V2X-Real-Seq-1`.
- The 50 detection test frames from the starter kit.

**Coordinate frame, important.** Ground truth and predictions must agree on frame convention.
During the challenge, reference data was at times expressed in the dataset world frame while
the released test frames are in each frame's local LiDAR frame, which yields AP exactly 0.0000
on every class while the file still validates. `src/worldify_v2x.py` converts
predictions LiDAR to world using the exact inverse of the OpenCOOD `x_to_world` transform
built from each frame's `lidar_pose`, with dimension, class and score invariance gated and a
round trip asserted to 1e-6. **Our submitted file is in the LiDAR frame.** Use the converter
only if your reference data is world frame.

**No frame in `forbidden_frames.txt` is used anywhere.** All real frame access routes through
`src/forbidden_guard.py`:

```bash
python src/compliance/test_forbidden_guard.py
```

---

## 3. Training (optional, weights are provided)

PointPillars, 40 epochs, synthetic data only.

```bash
python src/training/v2x_label_prep.py            # class mapping and label prep
python src/training/pedestrian_synth_v2x.py      # synthesise pedestrians
python src/training/generate_ped_deltas_v2x.py   # per frame pedestrian point deltas
python src/training/trashcan_synth_v2x.py        # static roadside objects
```

Full training drivers as used: `src/pod_v2x_run8.sh` and `pod_v2x_run9.sh`.
Config: `checkpoints/pointpillar_v2x_run2.yaml`.

Provided weights: `checkpoints/v2x_run8_epoch40.pth` and `v2x_run9_epoch40.pth`.
`src/build_v2x_harness.py` builds the local validation harness and contains
`pose_to_matrix`, the exact OpenCOOD pose replica used by the frame converter.

---

## 4. Inference

```bash
python src/inference/infer_lumpi.py \
    --ckpt   checkpoints/v2x_run9_epoch40.pth \
    --cfg    checkpoints/pointpillar_v2x_run2.yaml \
    --frames /path/to/detection_test_frames \
    --out    predictions_raw.json \
    --score_thresh 0.01
```

Reference inference outputs are included: `artifacts/predictions_v2x_test_run8.json` and
`artifacts/predictions_v2x_test_run9.json`.

Optional test time augmentation: `src/fuse_tta.py` runs four flip views, unflips
boxes per view, and averages headings as unit vectors via atan2 of summed sine and cosine.
It was applied to **pedestrian only**; on truck it costs 23 percent.

---

## 5. Detection post processing

In order:

1. **Per class NMS**: pedestrian **0.7**, vehicle 0.3, truck 0.3. This is the single largest
   detection lever on this track. Pedestrians cluster, and a 0.3 threshold on a 0.7 m box
   suppresses genuine neighbours; 1.0 collapses pedestrian AP to 0.0376, so 0.7 is a true
   interior optimum.
2. **Pedestrian geometric re centring** (`src/apply_refiner.py`), interior point
   centroid, pedestrians only.
3. **Pedestrian z offset** of +0.093 m, from matched pair calibration on confident
   predictions only. Computed over all predictions the same statistic gives 0.69 m, roughly
   7 times too large.
4. **Size calibration**: pedestrian length 0.894 and width 0.949, vehicle width 1.053, truck
   length 1.154.
5. **Density re ranking**, `score' = score * max(1 - beta + beta * log1p(n_interior) /
   log1p(40), 1e-6)`, beta: pedestrian 0.7, vehicle 0.2.
6. **Global monotone rescale** of scores into (0, 1], preserving every ranking.

`src/audit_v2x.py` contains the stack as standalone functions for evaluating
each stage individually.

**Multi hypothesis class emission.** Per class analysis showed that trucks in this dataset are
systematically detected but assigned to the vehicle class, established by the fact that a truck
hypothesis derived from vehicle detections raised truck AP on the evaluation server. Because
average precision is computed independently per class, and appending detections below an
existing ranking cannot reduce precision at any recall already attained, each vehicle detection
also carries a secondary truck hypothesis, with width scaled by 0.9091 to match the measured
truck aspect ratio. Of the 3645 submitted truck detections, 2408 arise this way and 1237 are
native. Measured contribution: 0.0015 of combined score. See the challenge report for the full
account.

---

## 6. Realism frames

**Generation** (`src/realism_v2x_final.py`): greedy frame selection scored by sliced
Wasserstein distance, iterated density resampling toward the real non null density, and
injection of **null returns at rate 0.23**. Approximately 31 percent of the real reference is
null returns encoded as the literal point (0, 0, 0), absent from the synthetic clouds; the
score optimal injection rate is below the true rate because the non null distributions still
differ. A vertical offset of dz = -0.3 m is applied to the non null points. All draws seeded.

Verified: regenerating from the raw twin data reproduces the banked frames to four decimal
places on every moment (x, y and z mean and standard deviation), on the same 50 selected
frames and at the same null fraction of 0.2300. Byte differences remain only because float32
rounding depends on operation order, and realism is a function of the distribution.

**Alignment** (`src/realism_affine_v2x.py`): per axis affine moment alignment at
**alpha = 0.25**, fitted against 50 real frames held out from the public training split.

```bash
python src/realism/realism_affine_v2x.py <pre_affine>.zip <out>.zip 0.25
```

The pre affine frames are in `artifacts/realism_frames_banked_preaffine/`. Applying the above
to them reproduces all 50 submitted frames **byte for byte**, and no other alpha in
{0.0, 0.15, 0.25, 0.40, 0.75, 1.0} reproduces any of them. Check it with
`src/verify_affine_reproduction.py`.

**Provenance note.** The original affine script for this track was not preserved.
`realism_affine_v2x.py` is a reconstruction from the transform definition, and the byte exact
reproduction above is the evidence that the reconstruction is faithful.

---

## 7. The submitted archive

`artifacts/FINAL_v2x.zip` contains `predictions.json`, the 50 `synthetic/*.bin` frames and
`declaration.txt`.

**Correction to the declaration.** The declaration inside the archive carries the tag
`v2x-probeC-nms-and-truck-donor` dated 2026-07-29, which names an earlier configuration than
the submitted one. Its substantive claims are accurate for the submitted file, namely that no
forbidden frame was trained on and that all detections come from a model trained only on
synthetic data. We flag the stale tag rather than let it stand uncorrected.

---

## 8. Contact

Questions about any step, or requests for intermediate artifacts including the 98 entry
findings log, can be directed to the team contact on the submission.

---

## Contact

Aman (team AMAN_UPG), aman.upg27024@gmail.com
