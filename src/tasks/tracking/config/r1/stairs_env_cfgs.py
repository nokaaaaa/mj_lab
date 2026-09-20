"""R1 motion tracking on the three-step sim2sim staircase."""

from mjlab.envs.mdp import bad_orientation, root_height_below_minimum
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains import TerrainEntityCfg

from src.tasks.tracking.config.r1.env_cfgs import unitree_r1_flat_tracking_env_cfg
from src.tasks.tracking.mdp.stair_rewards import (
  motion_absolute_body_pos,
  stair_foot_landing,
  stair_foot_target,
  stair_swing_clearance,
  stair_swing_forward,
  arm_excess_motion_penalty,
  arm_position_error_penalty,
  arm_joint_velocity_error_penalty,
)
from src.tasks.tracking.mdp.terminations import bad_anchor_pos
from src.tasks.velocity.config.r1.stairs_env_cfgs import (
  BoxR1ThreeStepTerrainCfg,
  MOTION_STAIR_HEIGHT,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from src.tasks.velocity import mdp as velocity_mdp
from src.assets.robots import R1_ACTION_SCALE
from src.tasks.tracking.mdp.actions import MotionJointPositionActionCfg


def unitree_r1_stairs_tracking_env_cfg(play: bool = False):
  cfg = unitree_r1_flat_tracking_env_cfg(play=play)
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      size=(2.0, 2.0),
      border_width=10.0,
      num_rows=1,
      num_cols=1,
      curriculum=False,
      difficulty_range=(1.0, 1.0),
      sub_terrains={
        "three_steps": BoxR1ThreeStepTerrainCfg(
          proportion=1.0,
          step_height_range=(MOTION_STAIR_HEIGHT, MOTION_STAIR_HEIGHT),
        )
      },
    ),
  )
  cfg.sim.nconmax = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.scene.num_envs = 1 if play else 512
  cfg.events.pop("push_robot", None)
  # The reference swing needs hip/knee changes around 1.5-2 rad, while the
  # ordinary R1 action scale moves those joint targets only 0.15 rad per unit.
  # Follow the reference with the position actuators and let the policy learn
  # a residual correction using the same scale on all six swings.
  cfg.actions["joint_pos"] = MotionJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=R1_ACTION_SCALE,
    use_default_offset=False,
    command_name="motion",
  )
  motion = cfg.commands["motion"]
  # Reference State Initialization (RSI). Starting every training episode at
  # frame 0 forced the policy to survive the *entire* climb sequentially
  # before ever getting gradient from later frames, and combined with falls
  # being true (non-bootstrapped) terminations while a frozen frame-0 target
  # is trivially matched by standing still, made "never move" a safer, higher
  # -return strategy than attempting the climb -- the policy learned to stand
  # still. Sampling random start frames -- weighted toward frames the policy
  # currently fails on -- gives direct gradient on the hard, later parts of
  # the motion and removes the long free ride a stationary policy got.
  motion.sampling_mode = "start" if play else "adaptive"
  # Adaptive sampling reprioritizes bins by *termination* rate, so a phase
  # the robot survives without falling -- like the very first stair mount,
  # a cold weight-shift out of a dead stop that's dynamically unlike every
  # later, already-moving footstep -- never accumulates failure weight and
  # stays chronically under-sampled even while later phases get hammered
  # (observed: sampling_top1_bin pinned near the top of the motion for the
  # whole run, and the first-footstep precision check in
  # scripts/eval_r1_stairs_2x.py stayed at 0 despite otherwise-improving
  # tracking). Raise the uniform floor so early frames keep getting real
  # practice regardless of the failure-bin curriculum.
  motion.adaptive_uniform_ratio = 0.35
  # Even with the uniform floor above, landing on the *literal* frame 0 is a
  # single sample out of the whole motion, weighted by chance within
  # whichever bin gets drawn -- so the true cold-start-to-first-step
  # transition (zero velocity, must initiate the whole climb from a dead
  # stop) stays a near-zero-probability training sample even though every
  # play/eval rollout starts there. Force a slice of resets to real frame 0
  # so this specific transition gets guaranteed, regular practice.
  motion.cold_start_ratio = 0.0 if play else 0.45
  motion.pose_range = (
    {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.05, 0.05)}
    if not play
    else {}
  )
  motion.velocity_range = (
    {} if play else {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}
  )
  motion.joint_position_range = (0.0, 0.0) if play else (-0.05, 0.05)
  # The fixed contact-then-pause startup exists so the always-frame-0 "start"
  # mode gets a moment to settle from a dead stop before the reference clock
  # is allowed to move. Under RSI the robot is teleported directly onto the
  # pose/velocity the sampled reference frame already has, so it doesn't need
  # that settle -- and keeping it enabled would let a stationary policy match
  # every freshly sampled frame for up to a second, once per episode, for
  # free, reintroducing the same exploit at a smaller scale.
  motion.startup_contact_sensor = "stair_feet_contact" if play else ""
  motion.startup_pause_s = 1.0
  motion.hold_last_frame = True
  motion.debug_vis = play

  foot_sensor = ContactSensorCfg(
    name="stair_feet_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  nonfoot_sensor = ContactSensorCfg(
    name="stair_nonfoot_contact",
    primary=ContactMatch(
      mode="geom", entity="robot", pattern=r".*_collision$",
      exclude=tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right") for i in range(1, 8)
      ),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (foot_sensor, nonfoot_sensor)
  cfg.rewards["nonfoot_ground"] = RewardTermCfg(
    func=velocity_mdp.illegal_contact,
    weight=-1.0,
    params={"sensor_name": nonfoot_sensor.name, "force_threshold": 2.0},
  )
  # The default anchor_pos check only looks at height, so a policy that never
  # walks forward never breaches it once the climb is under 3 * step_height
  # (0.45 m here) -- it can stand still for the whole episode for free. Use
  # the full 3D distance instead so forward drift toward the stairs is
  # actually required. Keep the threshold loose: this exists only to rule
  # out standing completely still, not to police tracking accuracy -- that's
  # the reward's job.
  cfg.terminations["anchor_pos"].func = bad_anchor_pos
  cfg.terminations["anchor_pos"].params["threshold"] = 0.6
  # Drop the per-body shape-mismatch termination entirely: any attempt at the
  # dynamic stepping motion transiently mismatches the reference shape, so a
  # tight threshold here punishes trying to move at all and reinforces
  # standing still, the opposite of the desired "fall while attempting the
  # motion, then fall less over training" curve. Falling is instead caught
  # directly below.
  del cfg.terminations["ee_body_pos"]
  # Direct, target-independent "did it fall over" checks, so a collapse is
  # always caught even if anchor_pos above is lenient. These -- not tracking
  # error -- should be the terminations that actually fire and shrink over
  # training.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=bad_orientation, params={"limit_angle": 0.9}
  )
  cfg.terminations["root_height"] = TerminationTermCfg(
    func=root_height_below_minimum, params={"minimum_height": 0.35}
  )
  # The base motion_body_pos reward re-centers both the reference and the
  # robot on the robot's own current anchor position before comparing, so it
  # only measures local pose *shape* and is blind to whether the robot ever
  # actually moved. A policy can therefore score well on it -- and on
  # motion_body_ori, motion_body_lin/ang_vel, which are similarly anchor-
  # relative -- by twitching its joints in place without ever climbing.
  # Swap in an absolute (world-space) version so every body's target visibly
  # marches away when the robot doesn't follow, making "stand still" a
  # strictly losing strategy instead of a free partial-credit one. This is
  # now the dominant reward term by weight.
  cfg.rewards["motion_body_pos"] = RewardTermCfg(
    func=motion_absolute_body_pos, weight=6.0, params={"command_name": "motion", "std": 0.3}
  )
  # Root position/orientation are already absolute; keep them salient too.
  cfg.rewards["motion_global_root_pos"].weight = 4.0
  cfg.rewards["motion_global_root_ori"].weight = 1.5
  # De-weight the remaining anchor-relative terms so they shape local joint
  # coordination without being able to outweigh the absolute tracking above.
  cfg.rewards["motion_body_ori"].weight = 0.5
  cfg.rewards["motion_body_lin_vel"].weight = 0.5
  cfg.rewards["motion_body_ang_vel"].weight = 0.5
  # The whole-body tracking terms above use a fairly wide std (0.3) averaged
  # over 14 bodies, so a policy can score well on them while still missing
  # the tread by a few centimeters -- exactly what happened: low average
  # tracking error, but the robot never actually plants a foot solidly on a
  # step (first_step/success both stayed at 0 in scripts/eval_r1_stairs_2x.py
  # even once the policy reliably survived the full episode without
  # collapsing). Layer in the sparse, high-precision foot-landing bonus now
  # that the plain-tracking behavior is reliable, per this file's original
  # plan (see module docstring).
  cfg.rewards["stair_foot_landing"] = RewardTermCfg(
    func=stair_foot_landing,
    weight=6.0,
    params={
      "command_name": "motion",
      "sensor_name": foot_sensor.name,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )
  # stair_foot_landing only fires inside six narrow ~0.5s contact windows, so
  # most of the episode still gets zero precision gradient on the feet
  # specifically. Add the dense per-step ankle tracking term (tighter std
  # than the whole-body reward) so foot placement is shaped continuously,
  # not just at the moment of contact.
  cfg.rewards["stair_foot_target"] = RewardTermCfg(
    func=stair_foot_target, weight=8.0, params={"command_name": "motion"}
  )
  # Keep the wrists close to the reference speed.  The margin preserves the
  # intended arm movement while discouraging additional hand flailing caused
  # by residual exploration.
  cfg.rewards["arm_excess_motion"] = RewardTermCfg(
    func=arm_excess_motion_penalty,
    weight=-1.0,
    params={"command_name": "motion", "speed_margin": 0.15},
  )
  cfg.rewards["arm_position_error"] = RewardTermCfg(
    func=arm_position_error_penalty,
    weight=-1.0,
    params={"command_name": "motion", "std": 0.10},
  )
  cfg.rewards["arm_joint_velocity_error"] = RewardTermCfg(
    func=arm_joint_velocity_error_penalty,
    weight=-1.0,
    params={"command_name": "motion", "std": 1.0},
  )
  # Strong, independent lift and forward gradients are needed from a planted
  # foot; a narrow symmetric tracking kernel left the scratch policy still.
  # The same weights apply to every scheduled swing.
  cfg.rewards["stair_swing_clearance"] = RewardTermCfg(
    func=stair_swing_clearance,
    weight=8.0,
    params={"command_name": "motion"},
  )
  cfg.rewards["stair_swing_forward"] = RewardTermCfg(
    func=stair_swing_forward,
    weight=6.0,
    params={"command_name": "motion"},
  )
  cfg.episode_length_s = 8.0
  return cfg
