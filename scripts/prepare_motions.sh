#!/bin/bash
# Turn every clip under ~/vollmont/誘導動作 into an R1 training motion.
#
#   video -> GVHMR -> GMR (R1) -> csv -> npz
#
# Each motion is named <category>_<clip id> (shinkou_0011, teishi_0007,
# habayose_0014) and that name is carried all the way through, so a training run
# directory always says which clip it came from.
#
#   setsid nohup scripts/prepare_motions.sh > ~/prepare_motions.log 2>&1 < /dev/null &
#
# Safe to re-run: anything that already has an .npz is skipped.
set -u

SRC="$HOME/vollmont/誘導動作"
ASSETS="$HOME/motion_assets"
REPO="$HOME/unitree_rl_mjlab"
NPZ_DIR="$REPO/src/assets/motions/r1"
STATUS="$REPO/logs/prepare_status.txt"
FPS=25   # every clip measured at 25 fps

mkdir -p "$ASSETS"/{videos,pkl,csv} "$NPZ_DIR" "$REPO/logs"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

# 進行 / 停止 / 幅寄せ -> romanised, because the toolchain writes these into
# file paths, conda env names and run directories.
declare -A CATEGORY=( ["進行"]="shinkou" ["停止"]="teishi" ["幅寄せ"]="habayose" )

source "$HOME/miniconda3/etc/profile.d/conda.sh"

total=0; done_n=0; skip=0; fail=0
for jp in "${!CATEGORY[@]}"; do
  for video in "$SRC/$jp"/*.MP4; do
    [ -e "$video" ] || continue
    total=$((total + 1))
    # DJI_20260904134225_0011_D.MP4 -> 0011
    id=$(basename "$video" | sed -E 's/.*_([0-9]{4})_D\.MP4/\1/')
    name="${CATEGORY[$jp]}_${id}"

    if [ -f "$NPZ_DIR/$name.npz" ]; then
      skip=$((skip + 1)); continue
    fi
    note "--- $name  ($jp/$(basename "$video"))"

    cp -n "$video" "$ASSETS/videos/$name.mp4"

    # 1. video -> human motion
    if [ ! -f "$HOME/GVHMR/outputs/demo/$name/hmr4d_results.pt" ]; then
      note "  GVHMR"
      ( conda activate gvhmr && cd "$HOME/GVHMR" && \
        python tools/demo/demo.py --video="$ASSETS/videos/$name.mp4" -s ) \
        >>"$STATUS.gvhmr" 2>&1 || { note "  FAIL GVHMR"; fail=$((fail+1)); continue; }
    fi

    # 2. human -> R1 joints. --input-fps matters: GVHMR emits one pose per video
    #    frame and carries no rate, so 25 fps read as 30 plays 1.2x fast.
    if [ ! -f "$ASSETS/pkl/$name.pkl" ]; then
      note "  GMR retarget"
      ( conda activate gmr && cd "$HOME/GMR" && \
        python scripts/gvhmr_to_robot.py \
          --gvhmr_pred_file "$HOME/GVHMR/outputs/demo/$name/hmr4d_results.pt" \
          --robot unitree_r1 --input-fps $FPS --headless \
          --save_path "$ASSETS/pkl/$name.pkl" ) \
        >>"$STATUS.gmr" 2>&1 || { note "  FAIL GMR"; fail=$((fail+1)); continue; }
    fi

    # 3. pkl -> csv  (one file at a time; the batch script walks a whole folder)
    if [ ! -f "$ASSETS/csv/$name.csv" ]; then
      ( conda activate gmr && cd "$HOME/GMR" && python - "$ASSETS/pkl/$name.pkl" "$ASSETS/csv/$name.csv" <<'PY'
import pickle, sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
m = pickle.load(open(src, "rb"))
dof = m["dof_pos"]
out = np.zeros((dof.shape[0], dof.shape[1] + 7), dtype=np.float32)
out[:, :3], out[:, 3:7], out[:, 7:] = m["root_pos"], m["root_rot"], dof
np.savetxt(dst, out, delimiter=",")
print(f"{dof.shape[0]} frames @ {m['fps']} fps -> {dst}")
PY
      ) >>"$STATUS.csv" 2>&1 || { note "  FAIL csv"; fail=$((fail+1)); continue; }
    fi

    # 4. csv -> npz (re-solves forward kinematics on the R1 model)
    note "  csv -> npz"
    ( conda activate unitree_rl_mjlab && cd "$REPO" && \
      WANDB_MODE=disabled python scripts/csv_to_npz.py --robot r1 \
        --input-file "$ASSETS/csv/$name.csv" --output-name "$name.npz" \
        --input-fps $FPS --output-fps 50 ) \
      >>"$STATUS.npz" 2>&1 || { note "  FAIL npz"; fail=$((fail+1)); continue; }

    if [ -f "$NPZ_DIR/$name.npz" ]; then
      note "  OK   $name"
      done_n=$((done_n + 1))
      # Publish as we go. /home/osone is 0750, so until this lands in the shared
      # area the other account cannot see any of it - and the queue behind this
      # runs for days.
      install -m 664 -g mjlab "$NPZ_DIR/$name.npz" /opt/mjlab-share/motions/npz/ 2>/dev/null
      install -m 664 -g mjlab "$ASSETS/videos/$name.mp4" /opt/mjlab-share/motions/video/ 2>/dev/null
      install -m 664 -g mjlab \
        "$HOME/GVHMR/outputs/demo/$name/${name}_3_incam_global_horiz.mp4" \
        "/opt/mjlab-share/motions/video/${name}_gvhmr.mp4" 2>/dev/null
    else
      note "  FAIL $name (no npz produced)"
      fail=$((fail + 1))
    fi
  done
done

note "prepare finished: $done_n new, $skip already present, $fail failed, of $total clips"
ls -1 "$NPZ_DIR" | sed 's/^/  npz: /' | tee -a "$STATUS"
