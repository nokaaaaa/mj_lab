#!/bin/bash
# Motion-imitation training for every clip prepared by prepare_motions.sh.
#
# One job per clip. The clip name goes into --agent.run-name, so the run
# directory reads logs/rsl_rl/r1_tracking/<timestamp>_<clip>/ and it is always
# obvious which video a result came from.
#
#   WAIT_FOR_PID=<pid> setsid nohup scripts/train_queue3_motions.sh \
#       > ~/train_queue3.log 2>&1 < /dev/null &
#
# Safe to re-run: a clip whose run directory already holds a final checkpoint is
# skipped, so an interrupted batch resumes where it stopped.
set -u

REPO="$HOME/unitree_rl_mjlab"
NPZ_DIR="$REPO/src/assets/motions/r1"
STATUS="$REPO/logs/queue_status.txt"
ENVS=4096
ITERS="${ITERS:-3000}"
TASK="Unitree-R1-Tracking-No-State-Estimation"
EXPERIMENT="r1_tracking"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

note() { echo "[$(date '+%F %T')] [q3] $*" | tee -a "$STATUS"; }

note "queue3 armed (motion imitation, ${ITERS} iterations per clip)"

# Wait for preprocessing and for whatever else holds the GPU.
if [ -n "$WAIT_FOR_PID" ]; then
  note "waiting for pid $WAIT_FOR_PID"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 60; done
fi
announced=0
while pgrep -f "bash .*scripts/prepare_motions\.sh" >/dev/null 2>&1; do
  [ "$announced" -eq 0 ] && { note "waiting for prepare_motions.sh"; announced=1; }
  sleep 120
done
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; do
  sleep 30
done

shopt -s nullglob
motions=("$NPZ_DIR"/*.npz)
if [ ${#motions[@]} -eq 0 ]; then
  note "no motions in $NPZ_DIR - did prepare_motions.sh run?"
  exit 1
fi
note "${#motions[@]} motion(s) to train"

for npz in "${motions[@]}"; do
  name="$(basename "$npz" .npz)"

  # Already trained? (a run directory for this clip holding a checkpoint)
  if compgen -G "$REPO/logs/rsl_rl/$EXPERIMENT/*_${name}/model_*.pt" >/dev/null; then
    note "SKIP  $name (already has a checkpoint)"
    continue
  fi

  note "START $name"
  cd "$REPO" || exit 1
  WANDB_MODE=disabled systemd-inhibit --what=sleep:idle \
      --why="mjlab motion training: $name" \
      unitree-rl-mjlab scripts/train.py "$TASK" \
      --motion_file "$npz" \
      --env.scene.num-envs=$ENVS \
      --agent.max-iterations="$ITERS" \
      --agent.experiment-name "$EXPERIMENT" \
      --agent.run-name "$name" \
      --agent.logger tensorboard 2>&1 | grep -vE "^Module |frame/s"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    note "DONE  $name"
  else
    note "FAIL  $name (exit $rc) - continuing"
  fi
done

note "queue3 finished"
