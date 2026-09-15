#!/bin/bash
# Run the queued trainings one after another.
#
# There is one GPU, so anything run in parallel just halves both. This walks the
# queue in order, skipping a job whose run directory already holds a final
# checkpoint, so it is safe to re-run after an interruption.
#
#   nohup scripts/train_queue.sh > ~/train_queue.log 2>&1 &
#
# Progress:  tail -f ~/train_queue.log
#            cat  ~/unitree_rl_mjlab/logs/queue_status.txt
set -u

REPO="$HOME/unitree_rl_mjlab"
STATUS="$REPO/logs/queue_status.txt"
ENVS=4096
mkdir -p "$REPO/logs"

# task_id | iterations | extra args
# Order matters. 1 is the deliverable; 2 and 3 together say whether the camera
# work is worth doing, and 3 is the one that counts because it is the only scan
# the robot could actually supply; 4 measures how much slack there is above the
# 20 cm the site has.
QUEUE=(
  "Unitree-R1-Stairs-Blind|15000|"
  "Unitree-R1-Stairs-Perceptive|15000|"
  "Unitree-R1-Stairs-Perceptive-Degraded|15000|"
  "Unitree-R1-Stairs-Blind-25cm|15000|"
)

note() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

wait_for_gpu() {
  # Do not start on top of a training that is already running. Ask the GPU what
  # is on it rather than grepping process tables - a pattern that matches
  # "scripts/train.py" also matches this script's own command line in some
  # shells, which deadlocks the queue against itself.
  local waited=0
  while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do
    if [ "$waited" -eq 0 ]; then note "GPU busy; waiting"; fi
    waited=$((waited + 1))
    sleep 30
  done
}

note "queue start: ${#QUEUE[@]} job(s)"
for entry in "${QUEUE[@]}"; do
  IFS='|' read -r task iters extra <<<"$entry"
  note "waiting for the GPU before $task"
  wait_for_gpu

  note "START $task  (${iters} iterations, ${ENVS} envs)"
  cd "$REPO" || exit 1
  # Hold off suspend for as long as a job runs; released when it exits.
  WANDB_MODE=disabled systemd-inhibit --what=sleep:idle \
      --why="mjlab training: $task" \
      unitree-rl-mjlab scripts/train.py "$task" \
      --env.scene.num-envs=$ENVS \
      --agent.max-iterations="$iters" \
      --agent.logger tensorboard $extra 2>&1 | grep -vE "^Module |frame/s"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    note "DONE  $task"
  else
    note "FAIL  $task (exit $rc) - continuing with the rest"
  fi
done
note "queue finished"
