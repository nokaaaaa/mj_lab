"""Contact and progress rewards for the R1 three-step motion."""

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_absolute_body_pos(
  env: "ManagerBasedRlEnv", command_name: str, std: float = 0.3
) -> torch.Tensor:
  """Track every body in true world space, not anchor-relative space.

  The base ``motion_body_pos`` reward compares body positions after
  re-centering both the reference and the robot on the robot's own current
  anchor position, so it only measures local pose *shape*: a robot that
  never walks forward can still score well on it by holding a joint
  configuration that coincidentally resembles the target's relative layout.
  For a task whose whole point is ending up in a specific place (partway up
  a staircase), that loophole makes "stand still" a viable strategy. This
  version compares true world positions instead, so standing still while
  the reference marches up the stairs necessarily drives every body's error
  up and the reward down -- there is no shape-only way to cheat it.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.body_pos_w - command.robot_body_pos_w).sum(dim=-1)
  return torch.exp(-error.mean(dim=-1) / std**2)


def stair_foot_landing(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  step_height: float = 0.15,
  tread: float = 0.24,
) -> torch.Tensor:
  """Reward the scheduled foot when it bears weight on the intended tread."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  robot = env.scene[asset_cfg.name]
  sensor = env.scene[sensor_name]
  feet = robot.data.site_pos_w[:, asset_cfg.site_ids]
  force = torch.linalg.vector_norm(sensor.data.force, dim=-1).reshape(env.num_envs, 2)
  local_x = feet[:, :, 0] - env.scene.env_origins[:, None, 0]
  local_z = feet[:, :, 2] - env.scene.env_origins[:, None, 2]
  score = torch.zeros(env.num_envs, device=env.device)
  # Original landing frames are at 30 Hz; the converted motion runs at 50 Hz
  # after interpreting those samples at 60 Hz.
  for step, side, frame in ((1, 1, 48), (1, 0, 81), (2, 0, 144),
                            (2, 1, 177), (3, 1, 210), (3, 0, 243)):
    center = round(frame * 50 / 60)
    phase = (command.time_steps >= center - 9) & (command.time_steps <= center + 16)
    x0 = 0.20 + (step - 1) * tread
    on_tread = (local_x[:, side] > x0 + 0.02) & (local_x[:, side] < x0 + tread - 0.02)
    height_ok = (local_z[:, side] - step * step_height).abs() < 0.08
    score += (phase & on_tread & height_ok & (force[:, side] > 10.0)).float()
  return score


def stair_swing_clearance(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lead_frames: int = 25,
  step_height: float = 0.15,
  ankle_height: float = 0.052,
  min_lift: float = 0.18,
  overshoot_scale: float = 0.12,
) -> torch.Tensor:
  """Give a strong lift gradient from a planted foot on every tread."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ankle_indexes = [
    command.cfg.body_names.index(name)
    for name in ("left_ankle_roll_link", "right_ankle_roll_link")
  ]
  actual_z = command.robot_body_pos_w[:, ankle_indexes, 2] - env.scene.env_origins[:, None, 2]
  reference_z = command.body_pos_w[:, ankle_indexes, 2] - env.scene.env_origins[:, None, 2]
  score = torch.zeros(env.num_envs, device=env.device)
  for step, side, frame in ((1, 1, 48), (1, 0, 81), (2, 0, 144),
                            (2, 1, 177), (3, 1, 210), (3, 0, 243)):
    center = round(frame * 50 / 60)
    phase = (command.time_steps >= center - lead_frames) & (command.time_steps <= center - 2)
    support_z = (step - 1) * step_height + ankle_height
    target_lift = (reference_z[:, side] - support_z).clamp_min(min_lift)
    actual_lift = actual_z[:, side] - support_z
    progress = (actual_lift / target_lift).clamp(max=1.0)
    overshoot = torch.relu(actual_lift - target_lift) / overshoot_scale
    score += phase.float() * (progress - overshoot)
  return score


def stair_swing_forward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lead_frames: int = 25,
  min_stride: float = 0.10,
  overshoot_scale: float = 0.12,
) -> torch.Tensor:
  """Give a strong forward gradient from the start of each swing."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ankle_indexes = [
    command.cfg.body_names.index(name)
    for name in ("left_ankle_roll_link", "right_ankle_roll_link")
  ]
  actual_x = command.robot_body_pos_w[:, ankle_indexes, 0] - env.scene.env_origins[:, None, 0]
  reference_x = command.body_pos_w[:, ankle_indexes, 0] - env.scene.env_origins[:, None, 0]
  score = torch.zeros(env.num_envs, device=env.device)
  for _, side, frame in ((1, 1, 48), (1, 0, 81), (2, 0, 144),
                            (2, 1, 177), (3, 1, 210), (3, 0, 243)):
    center = round(frame * 50 / 60)
    phase = (command.time_steps >= center - lead_frames) & (command.time_steps <= center - 2)
    start_x = command.motion.body_pos_w[center - lead_frames, ankle_indexes[side], 0]
    target_stride = (reference_x[:, side] - start_x).clamp_min(min_stride)
    actual_stride = actual_x[:, side] - start_x
    progress = (actual_stride / target_stride).clamp(max=1.0)
    overshoot = torch.relu(actual_stride - target_stride) / overshoot_scale
    score += phase.float() * (progress - overshoot)
  return score


def stair_climb_progress(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Provide dense progress only after the contact pause has completed."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  x = command.robot_anchor_pos_w[:, 0] - env.scene.env_origins[:, 0]
  z = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return command._startup_ready.float() * (
    torch.clamp(x / 0.9, 0.0, 1.0) * torch.clamp((z - 0.6) / 0.6, 0.0, 1.0)
  )


def stair_foot_target(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Densely shape both ankle trajectories with a tighter kernel than the
  whole-body tracking reward, so foot placement gets a continuous precision
  gradient every step rather than only within the narrow stair_foot_landing
  contact windows.

  No longer gated on ``_startup_ready``: that flag only means something for
  the fixed frame-0-plus-settle "start" sampling path used in play/eval.
  Under RSI (adaptive/uniform sampling, used for training) the robot is
  teleported directly onto the sampled reference frame's pose, so there is
  no settle phase to gate on, and ``_startup_ready`` would simply stay False
  forever, silently zeroing this reward for the entire training run.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  indexes = [
    command.cfg.body_names.index(name)
    for name in ("left_ankle_roll_link", "right_ankle_roll_link")
  ]
  target = command.body_pos_w[:, indexes]
  actual = command.robot_body_pos_w[:, indexes]
  error = torch.square(target - actual).sum(dim=-1)
  return torch.exp(-error.mean(dim=-1) / 0.10**2)


def arm_excess_motion_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_margin: float = 0.15,
) -> torch.Tensor:
  """Penalize excess linear speed across the complete arm.

  The reference arm movement remains allowed.  Only the excess speed of the
  actual shoulder, elbow, and wrist links over their corresponding reference
  speeds (plus a small margin) is penalized.  Averaging all six links prevents
  fast motion at the shoulder or elbow from being hidden by a wrist-only term.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  arm_indexes = [
    command.cfg.body_names.index(name)
    for name in (
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_roll_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_roll_link",
    )
  ]
  actual_speed = torch.linalg.vector_norm(
    command.robot_body_lin_vel_w[:, arm_indexes], dim=-1
  )
  reference_speed = torch.linalg.vector_norm(
    command.body_lin_vel_w[:, arm_indexes], dim=-1
  )
  excess_speed = torch.relu(actual_speed - reference_speed - speed_margin)
  return torch.square(excess_speed).mean(dim=-1)


def arm_position_error_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float = 0.10,
) -> torch.Tensor:
  """Penalize absolute position error across all arm links.

  This complements the velocity penalty: an arm that moves at the correct
  speed but remains at a constant offset from the reference is still penalized.
  The squared error is normalized by ``std`` so the weight has a predictable
  scale (an RMS error of ``std`` contributes approximately one).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  arm_indexes = [
    command.cfg.body_names.index(name)
    for name in (
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_roll_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_roll_link",
    )
  ]
  error = torch.square(
    command.robot_body_pos_w[:, arm_indexes]
    - command.body_pos_w[:, arm_indexes]
  ).sum(dim=-1)
  return error.mean(dim=-1) / std**2


def arm_joint_velocity_error_penalty(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float = 1.0,
) -> torch.Tensor:
  """Penalize arm joint velocities that differ from the reference.

  Link linear velocity does not fully capture fast shoulder/elbow/wrist
  rotations.  This term directly constrains the eight arm joints, while still
  allowing the velocity prescribed by the reference motion.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  robot = command.robot
  arm_joint_names = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
  )
  joint_ids, _ = robot.find_joints(arm_joint_names, preserve_order=True)
  actual = command.robot_joint_vel[:, joint_ids]
  target = command.joint_vel[:, joint_ids]
  return torch.square((actual - target) / std).mean(dim=-1)


def stair_initial_stillness(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  """Prefer quiet double support throughout the required one-second pause."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  speed = torch.linalg.vector_norm(command.robot_anchor_lin_vel_w, dim=-1)
  return (command._startup_contact & ~command._startup_ready).float() * torch.exp(
    -torch.square(speed / 0.15)
  )
