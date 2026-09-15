"""Map a GMR Unitree G1 (29 dof) motion CSV onto the Unitree R1 (24 dof) joint set.

This exists so the R1 tracking task can be exercised before GMR gains a native R1
retarget target: the two robots share joint *names* for everything R1 has, so the
G1 columns can be selected by name. The joints R1 lacks (waist pitch, wrist pitch
and yaw) are dropped, and every column is clipped to R1's own limits.

The result is an approximation - it reuses joint angles solved for G1's link
lengths - so it is meant for pipeline validation, not for a motion you intend to
deploy. Use a real R1 retarget for that.

CSV layout (both in and out): [root_pos(3), root_rot wxyz(4), dof_pos(N)].

Usage:
  python scripts/g1_csv_to_r1_csv.py --input-file in.csv --output-file out.csv
"""

from pathlib import Path

import numpy as np
import tyro

# GMR's g1_mocap_29dof.xml joint order == the CSV dof column order.
G1_JOINTS = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

# r1.xml joint order, and R1's own limits (radians) read from the same file.
R1_JOINT_LIMITS = {
  "left_hip_pitch_joint": (-2.93215, 2.54818),
  "left_hip_roll_joint": (-1.0472, 1.74533),
  "left_hip_yaw_joint": (-2.7402, 2.7402),
  "left_knee_joint": (-0.174533, 2.42601),
  "left_ankle_pitch_joint": (-0.87266, 0.57596),
  "left_ankle_roll_joint": (-0.2618, 0.2618),
  "right_hip_pitch_joint": (-2.93215, 2.54818),
  "right_hip_roll_joint": (-1.74533, 1.0472),
  "right_hip_yaw_joint": (-2.7402, 2.7402),
  "right_knee_joint": (-0.17453, 2.42601),
  "right_ankle_pitch_joint": (-0.87266, 0.57596),
  "right_ankle_roll_joint": (-0.261799, 0.261799),
  "waist_roll_joint": (-0.5236, 0.5236),
  "waist_yaw_joint": (-2.618, 2.618),
  "left_shoulder_pitch_joint": (-3.1416, 2.0944),
  "left_shoulder_roll_joint": (-0.22689, 2.4784),
  "left_shoulder_yaw_joint": (-1.9199, 1.9199),
  "left_elbow_joint": (-0.97564, 2.1852),
  "left_wrist_roll_joint": (-1.9199, 1.9199),
  "right_shoulder_pitch_joint": (-3.1416, 2.0944),
  "right_shoulder_roll_joint": (-2.47849, 0.2268),
  "right_shoulder_yaw_joint": (-1.9199, 1.9199),
  "right_elbow_joint": (-0.97564, 2.1852),
  "right_wrist_roll_joint": (-1.9199, 1.9199),
}
R1_JOINTS = list(R1_JOINT_LIMITS)

# Home pelvis height of each model's MJCF, used to bring the root trajectory down
# to R1's shorter legs.
G1_PELVIS_HEIGHT = 0.793
R1_PELVIS_HEIGHT = 0.74


def main(
  input_file: str,
  output_file: str,
  scale_root_height: bool = True,
) -> None:
  """Convert a G1 motion CSV to an R1 motion CSV.

  Args:
    input_file: GMR-produced G1 CSV.
    output_file: Where to write the R1 CSV.
    scale_root_height: Scale root z by R1/G1 pelvis height so the feet stay near
      the ground. Disable to keep the original trajectory.
  """
  motion = np.loadtxt(input_file, delimiter=",", dtype=np.float32)
  if motion.ndim != 2 or motion.shape[1] != 7 + len(G1_JOINTS):
    raise ValueError(
      f"expected {7 + len(G1_JOINTS)} columns (root 7 + {len(G1_JOINTS)} dof), "
      f"got shape {motion.shape}"
    )

  g1_index = {name: i for i, name in enumerate(G1_JOINTS)}
  out = np.zeros((motion.shape[0], 7 + len(R1_JOINTS)), dtype=np.float32)
  out[:, :7] = motion[:, :7]

  if scale_root_height:
    out[:, 2] = motion[:, 2] * (R1_PELVIS_HEIGHT / G1_PELVIS_HEIGHT)

  clipped = []
  for i, name in enumerate(R1_JOINTS):
    col = motion[:, 7 + g1_index[name]]
    lo, hi = R1_JOINT_LIMITS[name]
    n_out = int(np.count_nonzero((col < lo) | (col > hi)))
    if n_out:
      clipped.append((name, n_out, float(col.min()), float(col.max())))
    out[:, 7 + i] = np.clip(col, lo, hi)

  Path(output_file).parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_file, out, delimiter=",")

  dropped = [n for n in G1_JOINTS if n not in R1_JOINT_LIMITS]
  print(f"{motion.shape[0]} frames: {len(G1_JOINTS)} dof -> {len(R1_JOINTS)} dof")
  print(f"dropped (absent on R1): {', '.join(dropped)}")
  if clipped:
    print("clipped to R1 limits:")
    for name, n_out, lo, hi in clipped:
      pct = 100.0 * n_out / motion.shape[0]
      print(f"  {name}: {n_out} frames ({pct:.1f}%), source range [{lo:.3f}, {hi:.3f}]")
  else:
    print("no joint needed clipping")
  print(f"wrote {output_file}")


if __name__ == "__main__":
  tyro.cli(main)
