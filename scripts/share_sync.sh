#!/bin/bash
# Keep /opt/mjlab-share in step with results as they are produced.
#
# /home/osone is 0750, so nothing the queues generate is visible to the other
# account until it lands in the shared area. The queues run for days, so waiting
# until the end would mean days of nothing to look at.
#
#   setsid nohup scripts/share_sync.sh > ~/share_sync.log 2>&1 < /dev/null &
#
# Exits once every queue has drained and one last sync has run.
set -u

REPO="$HOME/unitree_rl_mjlab"
SHARE="/opt/mjlab-share"
NPZ_DIR="$REPO/src/assets/motions/r1"
STATUS="$REPO/logs/queue_status.txt"
INTERVAL="${INTERVAL:-300}"

note() { echo "[$(date '+%F %T')] [sync] $*" | tee -a "$STATUS"; }

# The share is root:mjlab, and `usermod -aG mjlab` does not reach a session that
# was already open - so this user's running processes are not in the group yet
# and writing there fails. Go through sudo rather than depend on a re-login.
sh_install() { sudo -n install -m 664 -g mjlab "$1" "$2" 2>/dev/null; }
sh_mkdir()   { sudo -n mkdir -p "$1" 2>/dev/null && sudo -n chgrp mjlab "$1" 2>/dev/null && sudo -n chmod g+rwXs "$1" 2>/dev/null; }
sh_copy()    { sudo -n cp -r "$1" "$2" 2>/dev/null; }

sync_once() {
  sh_mkdir "$SHARE/motions/npz"; sh_mkdir "$SHARE/motions/video"; sh_mkdir "$SHARE/logs"

  # Reference motions
  for f in "$NPZ_DIR"/*.npz; do
    [ -e "$f" ] || continue
    [ -e "$SHARE/motions/npz/$(basename "$f")" ] && continue
    sh_install "$f" "$SHARE/motions/npz/" && note "npz -> $(basename "$f")"
  done

  # Source clip and the GVHMR overlay, so a result can be checked against what
  # it came from without file access to /home/osone.
  for f in "$HOME/motion_assets/videos"/*.mp4; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .mp4)
    [ -e "$SHARE/motions/video/$name.mp4" ] || sh_install "$f" "$SHARE/motions/video/"
    overlay="$HOME/GVHMR/outputs/demo/$name/${name}_3_incam_global_horiz.mp4"
    [ -e "$overlay" ] && [ ! -e "$SHARE/motions/video/${name}_gvhmr.mp4" ] && \
      sh_install "$overlay" "$SHARE/motions/video/${name}_gvhmr.mp4"
  done

  # Finished policies. Checkpoints are deliberately left out - 6 MB each and
  # only useful for resuming on this machine.
  for run in "$REPO"/logs/rsl_rl/*/*/; do
    [ -f "$run/policy.onnx" ] || continue
    dest="$SHARE/logs/$(basename "$(dirname "$run")")__$(basename "$run")"
    [ -d "$dest" ] && continue
    sh_mkdir "$dest" || continue
    for f in "$run"/policy.onnx*; do [ -e "$f" ] && sh_install "$f" "$dest/"; done
    sh_copy "$run/params" "$dest/"
    sudo -n chgrp -R mjlab "$dest" 2>/dev/null; sudo -n chmod -R g+rwX "$dest" 2>/dev/null
    [ -f "$dest/policy.onnx" ] && note "policy -> $(basename "$dest")"
  done
}

queues_running() {
  ps -eo args --no-headers \
    | awk '$1=="/bin/bash" && $2 ~ /^scripts\/(train_queue|prepare_motions)/ {n++} END {exit !n}'
}

note "started (every ${INTERVAL}s)"
while true; do
  sync_once
  queues_running || break
  sleep "$INTERVAL"
done
sync_once
note "queues drained; final sync done"
