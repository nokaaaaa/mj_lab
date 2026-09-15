#!/bin/bash
# Second half of the training queue.
#
# The first queue process read its job list into memory at startup, so appending
# to that script does not reach it. This picks up after it exits rather than
# racing it for the GPU - waiting on the process, not on the GPU being briefly
# idle between its own two jobs.
#
#   setsid nohup scripts/train_queue2.sh > ~/train_queue2.log 2>&1 < /dev/null &
set -u

REPO="$HOME/unitree_rl_mjlab"
STATUS="$REPO/logs/queue_status.txt"
ENVS=4096
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

QUEUE=(
  "Unitree-R1-Stairs-Perceptive-Degraded|15000|"
  "Unitree-R1-Stairs-Blind-25cm|15000|"
)

note() { echo "[$(date '+%F %T')] [q2] $*" | tee -a "$STATUS"; }

note "queue2 armed: ${#QUEUE[@]} job(s)"

if [ -n "$WAIT_FOR_PID" ]; then
  note "waiting for queue1 (pid $WAIT_FOR_PID) to finish"
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 60
  done
  note "queue1 finished"
fi

# Belt and braces: never start while something else holds the GPU.
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; do
  sleep 30
done

for entry in "${QUEUE[@]}"; do
  IFS='|' read -r task iters extra <<<"$entry"
  note "START $task  (${iters} iterations, ${ENVS} envs)"
  cd "$REPO" || exit 1
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
note "queue2 finished"
