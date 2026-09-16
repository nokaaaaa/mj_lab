from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def stair_forward_progress(
  env: ManagerBasedRlEnv,
  target_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward reaching the fixed staircase landing along its +x axis."""
  asset: Entity = env.scene[asset_cfg.name]
  distance = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return torch.clamp(distance / target_distance, min=0.0, max=1.0)


def stair_forward_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward actual forward motion while the stair-climb command is active.

  Unlike the Gaussian velocity-tracking reward, this term is exactly zero when
  the robot stands still.  That prevents a conservative standing policy from
  collecting most of the tracking reward without attempting the staircase.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  commanded_forward_speed = command[:, 0]
  moving_command = commanded_forward_speed > 0.1
  speed_ratio = asset.data.root_link_lin_vel_b[:, 0] / torch.clamp(
    commanded_forward_speed, min=0.1
  )
  return moving_command.float() * torch.clamp(speed_ratio, min=0.0, max=1.0)


def stair_top_stop(
  env: ManagerBasedRlEnv,
  target_distance: float,
  max_distance: float,
  min_height: float,
  command_name: str,
  velocity_std: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward standing still on the landing after the motion command stops."""
  asset: Entity = env.scene[asset_cfg.name]
  distance = asset.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  on_landing = (
    (distance >= target_distance)
    & (distance <= max_distance)
    & (height >= min_height)
  )
  command = env.command_manager.get_command(command_name)
  stopped_command = torch.linalg.norm(command, dim=1) < 0.1
  lin_speed_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
  ang_speed_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  still = torch.exp(-(lin_speed_sq + 0.25 * ang_speed_sq) / velocity_std**2)
  return on_landing.float() * stopped_command.float() * still


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class stair_alternating_foot_lift:
  """Reward right-first alternating foot lifts while the other foot supports.

  Clearance is measured from each foot's height immediately before its swing,
  so the reward remains valid as the robot climbs onto higher stair treads.
  Site and contact order must be ``(left, right)``.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.swing_start_pos = torch.zeros(
      (env.num_envs, 2, 3), device=env.device, dtype=torch.float32
    )
    self.swing_start_relative_z = torch.zeros(
      (env.num_envs, 2), device=env.device, dtype=torch.float32
    )
    self.was_swing = torch.zeros(
      (env.num_envs, 2), device=env.device, dtype=torch.bool
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    period: float,
    target_lift: float,
    target_forward: float,
    first_target_forward: float,
    first_step_position: float,
    step_tread: float,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    assert sensor.data.found is not None

    foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    assert foot_pos.shape[1] == 2
    pelvis_z = asset.data.root_link_pos_w[:, 2].unsqueeze(1)
    foot_relative_z = foot_pos[:, :, 2] - pelvis_z

    phase = (env.episode_length_buf * env.step_dt / period) % 1.0
    desired_swing = torch.zeros_like(self.was_swing)
    # A short double-support interval precedes each swing. Right foot is index
    # 1 and always swings first; left foot (index 0) follows half a cycle later.
    desired_swing[:, 1] = (phase >= 0.05) & (phase < 0.45)
    desired_swing[:, 0] = (phase >= 0.55) & (phase < 0.95)

    reset = env.episode_length_buf <= 1
    starting_swing = desired_swing & ~self.was_swing
    update_baseline = ~desired_swing | starting_swing | reset.unsqueeze(1)
    self.swing_start_pos = torch.where(
      update_baseline.unsqueeze(2), foot_pos, self.swing_start_pos
    )
    self.swing_start_relative_z = torch.where(
      update_baseline, foot_relative_z, self.swing_start_relative_z
    )

    displacement = foot_pos - self.swing_start_pos
    # Pelvis-relative height prevents pitching the whole body from faking lift.
    clearance = torch.clamp(
      foot_relative_z - self.swing_start_relative_z, min=0.0
    )
    lift_score = torch.clamp(clearance / target_lift, min=0.0, max=1.0)
    unit_x = torch.zeros((env.num_envs, 3), device=env.device)
    unit_x[:, 0] = 1.0
    forward_w = quat_apply(asset.data.root_link_quat_w, unit_x)
    forward_displacement = torch.sum(
      displacement * forward_w.unsqueeze(1), dim=2
    )
    elapsed = env.episode_length_buf * env.step_dt
    forward_target = torch.where(
      elapsed < period,
      torch.full_like(elapsed, first_target_forward),
      torch.full_like(elapsed, target_forward),
    ).unsqueeze(1)
    forward_score = torch.clamp(
      forward_displacement / forward_target, min=0.0, max=1.0
    )
    # Do not move toward the riser until the foot has cleared it vertically.
    forward_score *= torch.clamp(clearance / 0.10, min=0.0, max=1.0)

    # Each swing has a fixed landing tread.  Relative displacement alone lets
    # the policy put the first right foot on the floor and the following left
    # foot on the second tread, so score the absolute site position as well.
    # Site x is measured from the terrain/environment origin in the robot's
    # current forward and lateral directions.
    origin = env.scene.env_origins.unsqueeze(1)
    relative_pos = foot_pos - origin
    forward_position = torch.sum(relative_pos * forward_w.unsqueeze(1), dim=2)
    right_unit = torch.zeros((env.num_envs, 3), device=env.device)
    right_unit[:, 1] = 1.0
    right_w = quat_apply(asset.data.root_link_quat_w, right_unit)
    lateral_displacement = torch.sum(displacement * right_w.unsqueeze(1), dim=2)

    # Right is index 1 and starts at t=0; left is index 0 and starts half a
    # period later.  Clamp at the third tread so no fourth-step target exists.
    right_stage = torch.floor(elapsed / period).clamp(min=0.0, max=2.0)
    left_stage = torch.floor((elapsed - 0.5 * period) / period).clamp(min=0.0, max=2.0)
    stage = torch.stack((left_stage, right_stage), dim=1)
    target_position = first_step_position + stage * step_tread
    landing_position_score = torch.exp(
      -torch.square(forward_position - target_position) / (0.08**2)
    )
    straight_forward_score = torch.exp(
      -torch.square(lateral_displacement) / (0.05**2)
    )

    contact = sensor.data.found > 0
    # When left swings, right must support; when right swings, left must support.
    support_contact = torch.stack((contact[:, 1], contact[:, 0]), dim=1)
    # Height alone is not sufficient: multiplying by forward placement makes
    # pulling a foot backwards worth exactly zero.
    per_foot_score = (
      lift_score
      * forward_score
      * landing_position_score
      * straight_forward_score
      * support_contact.float()
      * desired_swing.float()
    )
    score = torch.sum(per_foot_score, dim=1)

    active_command = torch.linalg.norm(command[:, :2], dim=1) > 0.1
    self.was_swing.copy_(desired_swing)
    return score * active_command.float()


def stair_swing_foot_upward_velocity(
  env: ManagerBasedRlEnv,
  period: float,
  target_upward_velocity: float,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward immediate upward motion during each right-first swing onset."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.found is not None

  phase = (env.episode_length_buf * env.step_dt / period) % 1.0
  raising = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
  # Site order is (left, right). Reward only the rising half of each swing.
  raising[:, 1] = (phase >= 0.05) & (phase < 0.25)
  raising[:, 0] = (phase >= 0.55) & (phase < 0.75)

  contact = sensor.data.found > 0
  support_contact = torch.stack((contact[:, 1], contact[:, 0]), dim=1)
  foot_upward_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2]
  upward_score = torch.clamp(
    foot_upward_velocity / target_upward_velocity, min=0.0, max=1.0
  )
  score = torch.sum(
    upward_score * raising.float() * support_contact.float(), dim=1
  )
  active_command = torch.linalg.norm(command[:, :2], dim=1) > 0.1
  return score * active_command.float()


def stair_swing_foot_forward_velocity(
  env: ManagerBasedRlEnv,
  period: float,
  target_forward_velocity: float,
  command_name: str,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward moving the scheduled swing foot in the robot's forward direction."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.found is not None

  phase = (env.episode_length_buf * env.step_dt / period) % 1.0
  swinging = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
  swinging[:, 1] = (phase >= 0.25) & (phase < 0.45)
  swinging[:, 0] = (phase >= 0.75) & (phase < 0.95)

  contact = sensor.data.found > 0
  support_contact = torch.stack((contact[:, 1], contact[:, 0]), dim=1)
  unit_x = torch.zeros((env.num_envs, 3), device=env.device)
  unit_x[:, 0] = 1.0
  forward_w = quat_apply(asset.data.root_link_quat_w, unit_x)
  foot_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :]
  foot_forward_velocity = torch.sum(foot_velocity * forward_w.unsqueeze(1), dim=2)
  forward_score = torch.clamp(
    foot_forward_velocity / target_forward_velocity, min=0.0, max=1.0
  )
  score = torch.sum(
    forward_score * swinging.float() * support_contact.float(), dim=1
  )
  active_command = torch.linalg.norm(command[:, :2], dim=1) > 0.1
  return score * active_command.float()


def stair_swing_leg_flexion(
  env: ManagerBasedRlEnv,
  period: float,
  command_name: str,
  sensor_name: str,
  hip_neutral: float,
  hip_target: float,
  knee_neutral: float,
  knee_target: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward forward hip flexion and deep knee flexion on the swing leg.

  Joint order must be left hip, left knee, right hip, right knee.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  assert sensor.data.found is not None

  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  assert joint_pos.shape[1] == 4
  hip_pos = torch.stack((joint_pos[:, 0], joint_pos[:, 2]), dim=1)
  knee_pos = torch.stack((joint_pos[:, 1], joint_pos[:, 3]), dim=1)
  hip_score = torch.clamp(
    (hip_neutral - hip_pos) / (hip_neutral - hip_target), min=0.0, max=1.0
  )
  knee_score = torch.clamp(
    (knee_pos - knee_neutral) / (knee_target - knee_neutral), min=0.0, max=1.0
  )

  phase = (env.episode_length_buf * env.step_dt / period) % 1.0
  swinging = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
  swinging[:, 1] = (phase >= 0.05) & (phase < 0.45)
  swinging[:, 0] = (phase >= 0.55) & (phase < 0.95)
  contact = sensor.data.found > 0
  support_contact = torch.stack((contact[:, 1], contact[:, 0]), dim=1)
  flexion_score = 0.5 * (hip_score + knee_score)
  score = torch.sum(
    flexion_score * swinging.float() * support_contact.float(), dim=1
  )
  active_command = torch.linalg.norm(command[:, :2], dim=1) > 0.1
  return score * active_command.float()


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
