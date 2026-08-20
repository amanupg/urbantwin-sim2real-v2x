#!/bin/bash
# Pod driver: V2X run9 = run2 proven base + ONE change: TrashCan synthesis.
#
# run8 FAILED (harness mAP 0.3815 vs run2's 0.434) because it silently used the
# WRONG pedestrian set: it pulled insomnia/ped_deltas_v2_v2x.tgz (19.0 peds per
# frame) while run2 used ped_deltas_v2x.tgz from the bucket ROOT (15.6/frame).
# That is +22% pedestrian density -- the known over-prediction failure mode --
# so run8 tested TrashCan AND a ped regression at once. run2's set is not in
# insomnia/ at all, so the driver's silent `[ -d X ] ||` fallback could never
# have found it. TrashCan itself was never actually evaluated.
#
# Fixed here: run2's exact ped source, plus a FINGERPRINT GATE that hard-fails
# if the composition does not match run2 + TrashCan.
# TrashCan = 42% of truck GT, absent from synthetic. This adds boxy TrashCan
# point clouds at real positions, labeled as 'truck' (SUPER_CLASS_MAP).
# No ped changes, no truck surgery, no anchor changes. Isolates TrashCan.
set -u
B=gs://eccv-data-bucket-amlc
WORK=/workspace
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
source "$WORK/venv/bin/activate"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PCD=$WORK/OpenPCDet
mark() { for i in 1 2 3; do echo "$2 @ $(date -u)" | gcloud storage cp - "$B/status/$1.txt" 2>/dev/null && break || sleep 5; done; }
last_ckpt() { ls -t "$PCD"/output/custom_models/pointpillar_v2x_run2/run9*/ckpt/checkpoint_epoch_*.pth 2>/dev/null | head -1 || true; }

if [ ! -f "$PCD/data/custom_v2x/.run9_ready" ]; then
  cd "$WORK"
  # 1. Clean pcd re-extract from bucket originals
  if [ ! -d "$WORK/UCF-DT-V2X-Real-Seq-1" ]; then
    mkdir -p v2x_parts && cd v2x_parts
    gcloud storage cp "$B/ut_v2x/*" . || exit 1
    rm -f full.zip
    for p in UCF-UT-V2X-Real-Seq-1.zip.0*; do cat "$p" >> full.zip && rm "$p"; done
    cd "$WORK"
    unzip -q v2x_parts/full.zip && rm -rf v2x_parts
  fi
  # 2. ped deltas (154K, run2's original) + rider-rescaled labels + TrashCan deltas
  # run2's EXACT ped set: bucket ROOT ped_deltas_v2x.tgz -> extracts to ped_deltas/
  if [ ! -d ped_deltas_run2 ]; then
    gcloud storage cp $B/ped_deltas_v2x.tgz . || { echo "FATAL: run2 ped set missing"; exit 1; }
    tar xzf ped_deltas_v2x.tgz && mv ped_deltas ped_deltas_run2
  fi
  [ -d lbl_prepped ] || { gcloud storage cp $B/lbl_prepped_v2x.tgz . && tar xzf lbl_prepped_v2x.tgz && mv lbl_prepped lbl_prepped 2>/dev/null || true; }
  [ -d lbl_prepped ] || { gcloud storage cp $B/insomnia/lbl_prepped_v2x.tgz . && tar xzf lbl_prepped_v2x.tgz; }
  # TrashCan deltas (new)
  [ -d trashcan_deltas ] || { gcloud storage cp $B/insomnia/trashcan_deltas_v2x.tgz . && tar xzf trashcan_deltas_v2x.tgz; }
  find ped_deltas_run2 lbl_prepped trashcan_deltas -name '._*' -delete 2>/dev/null || true
  # 3. Merge ped points + TrashCan points + build labels (run2 recipe + TrashCan)
  python - <<'PY'
import os, numpy as np
from pathlib import Path
work = Path('/workspace')
src = work/'UCF-DT-V2X-Real-Seq-1'
dst = work/'OpenPCDet/data/custom_v2x'
pts_dst = dst/'points'
for p in sorted((src/'pcd').glob('*.npy')):
    tgt = pts_dst/p.name
    if tgt.exists(): tgt.unlink()
    os.link(p, tgt)
# Merge run2's ped deltas (15.6/frame - verified)
marker = dst/'.ped_run2_merged'
if not marker.exists():
    d = work/'ped_deltas_run2'; n=0
    for f in sorted(pts_dst.glob('*.npy')):
        df = d/f.name
        if not df.exists(): continue
        add = np.load(df)
        if len(add):
            base=np.load(f); np.save(f, np.concatenate([base, add.astype(base.dtype)])); n+=len(add)
    marker.write_text('done'); print('merged ped v2 points:', n)
# Merge TrashCan deltas
marker_tc = dst/'.trashcan_merged'
if not marker_tc.exists():
    d = work/'trashcan_deltas'; n=0
    for f in sorted(pts_dst.glob('*.npy')):
        df = d/f.name
        if not df.exists(): continue
        add = np.load(df)
        if len(add):
            base=np.load(f); np.save(f, np.concatenate([base, add.astype(base.dtype)])); n+=len(add)
    marker_tc.write_text('done'); print('merged trashcan points:', n)
# Labels: run2 recipe (rider rescale, NO truck surgery) + TrashCan labels
SUPER = {'Car':'vehicle','Truck':'truck','Van':'truck','Bus':'truck',
         'BicycleRider':'pedestrian','MotorcyleRider':'pedestrian','Pedestrian':'pedestrian'}
n_ped=0; n_tc=0
lp_src = work/'lbl_prepped' if (work/'lbl_prepped').exists() else src/'lbl_prepped'
for lp in sorted(lp_src.glob('*.txt')):
    out=[]
    for line in lp.read_text().splitlines():
        f = line.split()
        if len(f)>=8 and f[7] in SUPER:
            out.append(' '.join(f[:7]+[SUPER[f[7]]]))
    d = work/'ped_deltas_run2'/lp.name
    if d.exists():
        for r in d.read_text().splitlines():
            if r.strip():
                f = r.split()
                out.append(' '.join(f[:7]+['pedestrian'])); n_ped+=1
    # TrashCan labels
    tc = work/'trashcan_deltas'/lp.name
    if tc.exists():
        for r in tc.read_text().splitlines():
            if r.strip():
                f = r.split()
                if len(f) >= 8:
                    # Keep as 'truck' (TrashCan maps to truck in SUPER_CLASS_MAP)
                    out.append(' '.join(f[:7]+['truck'])); n_tc+=1
    (dst/'labels'/lp.name).write_text('\n'.join(out)+'\n')
print('labels rebuilt, ped rows:', n_ped, 'trashcan rows:', n_tc)

# ---- FINGERPRINT GATE: composition must equal run2 + TrashCan, else abort ----
from collections import Counter
import sys as _s
cnt=Counter(); nfr=0
for lp in sorted((dst/'labels').glob('*.txt')):
    nfr+=1
    for line in lp.read_text().splitlines():
        f=line.split()
        if len(f)>=8: cnt[f[7]]+=1
per={k:v/max(nfr,1) for k,v in cnt.items()}
print('FINGERPRINT frames=%d per-frame=%s' % (nfr, {k:round(v,2) for k,v in per.items()}))
# run2 reference (measured from its inputs): ped 15.6/frame; TrashCan adds 1.0 truck/frame
# Gate on the DISCRIMINATING quantity: ped DELTA rows (n_ped), not the total
# 'pedestrian' class -- that class also contains riders from lbl_prepped via
# SUPER_CLASS_MAP, so its density (~46/frame) is identical for both ped sets and
# cannot distinguish them. run2's set = 154,035 rows; the wrong v2 set = ~187,883.
ok = (150000 <= n_ped <= 160000) and (9000 <= n_tc <= 11000) and (nfr >= 9000)
if not ok:
    print('FINGERPRINT_FAIL: ped_delta_rows=%d (want 150000-160000, run2=154035), '
          'trashcan_rows=%d (want ~10000), frames=%d' % (n_ped, n_tc, nfr)); _s.exit(1)
print('FINGERPRINT_OK')
PY
  [ $? -eq 0 ] || { mark v2x_run9 "FAILED_FINGERPRINT"; exit 1; }
  # 4. Config: run2 EXACT config (no changes)
  python - <<'PY'
from pathlib import Path
cfg = Path('/workspace/OpenPCDet/tools/cfgs/custom_models/pointpillar_v2x_run2.yaml')
s = cfg.read_text()
if '5.80, 2.30' in s or '3-mode' in s:
    print('WARNING: config was modified by run6 — need to restore from bucket')
    import subprocess
    subprocess.run(['gcloud','storage','cp','gs://eccv-data-bucket-amlc/checkpoints/custom_models/pointpillar_v2x_run2/run2/pointpillar_v2x_run2.yaml', str(cfg)])
    print('restored run2 config from bucket')
else:
    print('config OK (run2 original)')
PY
  # 5. Infos via symlink dance
  cd "$PCD/data"
  [ -L custom ] && rm custom
  [ -d custom_lumpi_park ] || mv custom custom_lumpi_park 2>/dev/null || true
  ln -s custom_v2x custom
  rm -f custom_v2x/custom_infos_train.pkl custom_v2x/custom_infos_val.pkl
  rm -rf custom_v2x/gt_database
  cd "$PCD"
  python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/v2x_ut.yaml
  ls data/custom_v2x/custom_infos_train.pkl || { echo INFOS_FAILED; exit 1; }
  rm custom && mv custom_lumpi_park custom
  touch "$PCD/data/custom_v2x/.run9_ready"
  echo SETUP_DONE
fi

cd "$PCD/tools"
gcloud storage rsync -r "$B/checkpoints" "$PCD/output" || true
setsid bash -c 'while true; do gcloud storage rsync -r "$PCD/output" "$B/checkpoints" >/dev/null 2>&1; sleep 300; done' >/dev/null 2>&1 &
mark v2x_run9 "RUNNING (run9, TrashCan synthesis)"
latest=$(last_ckpt)
resume=""
[ -n "$latest" ] && resume="--ckpt $latest"
echo "launching train.py resume='$resume'"
python train.py --cfg_file cfgs/custom_models/pointpillar_v2x_run2.yaml --extra_tag run9 $resume > "$WORK/train_v2x_run9.log" 2>&1
rc=$?
echo "train.py exited rc=$rc"
last=$(last_ckpt)
if [ -n "$last" ] && [[ "$last" == *"epoch_40.pth" ]]; then
  mark v2x_run9 DONE
  mkdir -p "$WORK/frames"
  for d in v2x_test_frames v2x_val_frames; do
    [ -d "$WORK/frames/$d" ] || gcloud storage cp -r "$B/$d" "$WORK/frames/"
  done
  V=$(last_ckpt)
  python "$PCD/infer_lumpi.py" --ckpt "$V" --cfg cfgs/custom_models/pointpillar_v2x_run2.yaml \
    --frames "$WORK/frames/v2x_test_frames" --out "$WORK/predictions_v2x_test_run9.json" --score_thresh 0.1
  python "$PCD/infer_lumpi.py" --ckpt "$V" --cfg cfgs/custom_models/pointpillar_v2x_run2.yaml \
    --frames "$WORK/frames/v2x_val_frames" --out "$WORK/predictions_v2x_val_run9.json" --score_thresh 0.1
  gcloud storage cp "$WORK"/predictions_v2x_*_run9.json "$B/preds/" || true
  gcloud storage rsync -r "$PCD/output" "$B/checkpoints" || true
  echo V2X_RUN8_COMPLETE
  exit 0
fi
CNT_FILE="$WORK/.v2x_run9_resubmit_count"
cnt=0; [ -f "$CNT_FILE" ] && cnt=$(cat "$CNT_FILE")
if [ "$rc" -ne 1 ] && [ "$cnt" -lt 3 ]; then
  echo $((cnt+1)) > "$CNT_FILE"
  nohup bash "$0" >/dev/null 2>&1 &
else
  mark v2x_run9 "FAILED_RC${rc}_NOEPOCH40 (runpod)"
fi
