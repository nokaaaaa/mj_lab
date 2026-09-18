"""Unitree R1 flat tracking environment configurations."""

from dataclasses import fields

from src.assets.robots import (
  R1_ACTION_SCALE,
  get_r1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.tasks.tracking.mdp.commands import MotionCommandCfg

from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_r1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree R1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  # R1 carries more collision geoms than the G1 the tracking defaults were tuned
  # for; the flat-terrain velocity task raises the same budgets.
  cfg.sim.njmax = 300
  cfg.sim.nconmax = 48
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.entities = {"robot": get_r1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = R1_ACTION_SCALE

  base_motion_cmd = cfg.commands["motion"]
  motion_cmd = MotionCommandCfg(
    **{
      field.name: getattr(base_motion_cmd, field.name)
      for field in fields(base_motion_cmd) if field.name != "viz"
    },
    viz=MotionCommandCfg.VizCfg(
      mode=base_motion_cmd.viz.mode,
      ghost_color=base_motion_cmd.viz.ghost_color,
    ),
  )
  cfg.commands["motion"] = motion_cmd
  motion_cmd.anchor_body_name = "torso_link"
  # R1's arms end at the wrist roll link (no wrist pitch/yaw), so it takes the
  # place of the G1's wrist yaw link as the hand body.
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_roll_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Smoothness penalties. The shared tracking config only damps the action rate,
  # which leaves the policy free to chatter at high torque - fine in sim, hard on
  # hardware. unitree_rl_lab's mimic task carries both of these alongside
  # action_rate_l2; the weights are taken from there.
  cfg.rewards["joint_acc"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-2.5e-7,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["joint_torque"] = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-1e-5,
    params={"asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",))},
  )

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
