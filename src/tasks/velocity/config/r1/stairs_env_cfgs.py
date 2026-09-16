"""Unitree R1 blind stair-traversal configurations.

Target: do not fall on the 20 cm steps measured at the vollmont site
(`~/vollmont/段差`). The robot is teleoperated - the operator supplies the
velocity command and can see the step - so the policy does not have to find the
step, only to survive it. That lets us drop the height scan from the actor and
run purely on proprioception, which in turn means nothing new is needed on the
robot: no depth camera, no elevation map, no new deploy observation.

For reference, R1's leg (hip to ankle) is 0.597 m standing and its foot is
0.163 m long, so a 20 cm step is 33% of leg length on a 20 cm tread - hard, and
roughly the ceiling of what blind locomotion manages. The terrain curriculum
starts far easier and works up, and `STAIR_HEIGHT_MAX` is the knob to sweep when
measuring where it actually breaks.
"""

from dataclasses import dataclass, replace

import mujoco
import mjlab.terrains as terrain_gen
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

from src.tasks.velocity.config.r1.env_cfgs import unitree_r1_rough_env_cfg
from src.assets.robots import R1_ACTION_SCALE
from src.tasks.velocity import mdp

# Measured on site: riser ~0.20 m. Tread read as 0.20-0.25 m from the photos;
# 0.20 is used because a narrower tread is the harder case, so training stays
# conservative with respect to the real steps.
STAIR_HEIGHT_MAX = 0.20
STAIR_TREAD = 0.20

# Exact dimensions of src/assets/robots/unitree_r1/xmls/stair.xml.
MOTION_STAIR_HEIGHT = 0.15
MOTION_STAIR_TREAD = 0.24
MOTION_STAIR_COUNT = 3
MOTION_STAIR_FRONT = 0.20
MOTION_LANDING_LENGTH = 0.50
# Six half-swings (right/left on each of the three treads), then command zero.
MOTION_CLIMB_DURATION = 3.0
MOTION_FORWARD_SPEED = 0.25
MOTION_TARGET_DISTANCE = 1.05
MOTION_GAIT_PERIOD = 1.0


@dataclass(kw_only=True)
class BoxR1ThreeStepTerrainCfg(SubTerrainCfg):
  """Three-step staircase matching the standalone R1 sim2sim scene."""

  step_height_range: tuple[float, float] = (0.05, MOTION_STAIR_HEIGHT)
  step_width: float = MOTION_STAIR_TREAD
  num_steps: int = MOTION_STAIR_COUNT
  stair_front: float = MOTION_STAIR_FRONT
  landing_length: float = MOTION_LANDING_LENGTH

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )

    # The output origin is the robot spawn. Geometry is expressed relative to
    # it with the same coordinates as stair.xml: first riser at x=0.20 m.
    spawn_x = 0.50
    center_y = self.size[1] / 2
    geometries: list[TerrainGeometry] = []

    floor = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.size[0] / 2, self.size[1] / 2, 0.05),
      pos=(self.size[0] / 2, center_y, -0.05),
    )
    geometries.append(TerrainGeometry(geom=floor, color=(0.25, 0.30, 0.35, 1.0)))

    for index in range(1, self.num_steps + 1):
      height = index * step_height
      center_x = (
        spawn_x + self.stair_front + (index - 0.5) * self.step_width
      )
      step = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(self.step_width / 2, 0.75, height / 2),
        pos=(center_x, center_y, height / 2),
      )
      geometries.append(TerrainGeometry(geom=step, color=(0.55, 0.55, 0.58, 1.0)))

    landing_height = self.num_steps * step_height
    landing_start = spawn_x + self.stair_front + self.num_steps * self.step_width
    landing = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.landing_length / 2, 0.75, landing_height / 2),
      pos=(landing_start + self.landing_length / 2, center_y, landing_height / 2),
    )
    geometries.append(TerrainGeometry(geom=landing, color=(0.55, 0.55, 0.58, 1.0)))

    return TerrainOutput(
      origin=np.array((spawn_x, center_y, 0.0)), geometries=geometries
    )


def stair_terrains_cfg(
  height_max: float = STAIR_HEIGHT_MAX,
  tread: float = STAIR_TREAD,
) -> TerrainGeneratorCfg:
  """Terrain weighted towards steps, from flat up to `height_max`.

  The curriculum interpolates each sub-terrain's difficulty from the low end of
  its range to the high end, so `step_height_range=(0.02, height_max)` means
  levels start near-flat and end at the target step.
  """
  # Patch size and column count are set for simulation cost, not variety. A
  # 0.20 m tread means a pyramid patch is built from a lot of boxes, and the
  # collision broadphase across 4096 envs scales with the total. Measured:
  # 8 m x 20 cols = 21084 geoms (15.9 s/iter), 6 m x 10 cols + merge = 3910.
  # Shrinking the flat platform at the centre from 2 m to 1 m then lets a 5 m
  # patch carry the same 10 steps for 3236 geoms - the wasted middle goes, the
  # climb stays.
  return TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=20.0,
    num_rows=10,  # difficulty levels
    num_cols=10,  # variations per level
    curriculum=True,
    sub_terrains={
      # Somewhere to recover and to learn plain walking.
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.1),
      # Ascending and descending stairs, equally weighted. Descending is the
      # harder direction - a missed step becomes a drop - so it gets the same
      # share rather than less.
      "stairs_up": terrain_gen.BoxPyramidStairsTerrainCfg(
        proportion=0.3,
        step_height_range=(0.02, height_max),
        step_width=tread,
        platform_width=1.0,
        border_width=1.0,
      ),
      "stairs_down": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
        proportion=0.3,
        step_height_range=(0.02, height_max),
        step_width=tread,
        platform_width=1.0,
        border_width=1.0,
      ),
      # Irregular steps - the "段差だらけ" case. Height range is +/-, so half
      # the step height gives the same worst-case rise between two cells.
      "random_steps": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.3,
        grid_width=0.45,
        grid_height_range=(0.01, height_max / 2),
        platform_width=1.0,
        merge_similar_heights=True,  # collapses adjacent equal cells into one box
      ),
    },
  )


def unitree_r1_stairs_blind_env_cfg(
  play: bool = False,
  height_max: float = STAIR_HEIGHT_MAX,
  tread: float = STAIR_TREAD,
  scan_resolution: float | None = 0.2,
) -> ManagerBasedRlEnvCfg:
  """Blind stair traversal: rough-terrain locomotion with no height scan."""
  cfg = unitree_r1_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(
    stair_terrains_cfg(height_max=height_max, tread=tread)
  )

  # Blind: the actor never sees the terrain. The critic keeps its height scan -
  # it only runs in training, and the extra information makes the value estimate
  # far better without leaking into the deployed policy.
  if "height_scan" in cfg.observations["actor"].terms:
    new_actor_terms = {
      k: v for k, v in cfg.observations["actor"].terms.items() if k != "height_scan"
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=cfg.observations["actor"].enable_corruption,
    )

  # Once height_scan is out of the actor, the raycast grid only feeds the critic.
  # It is not cheap: the default 1.6 x 1.0 grid at 0.1 m is 187 rays per env,
  # ~766k rays per step against the terrain. Coarsening it keeps the critic's
  # terrain awareness at a fraction of the cost; None drops the sensor entirely.
  if scan_resolution is None:
    cfg.scene.sensors = tuple(
      s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    for group in ("actor", "critic"):
      if "height_scan" in cfg.observations[group].terms:
        cfg.observations[group] = ObservationGroupCfg(
          terms={
            k: v
            for k, v in cfg.observations[group].terms.items()
            if k != "height_scan"
          },
          concatenate_terms=True,
          enable_corruption=cfg.observations[group].enable_corruption,
        )
  else:
    for sensor in cfg.scene.sensors or ():
      if sensor.name == "terrain_scan":
        sensor.pattern = replace(sensor.pattern, resolution=scan_resolution)

  if play:
    # Start at the hardest level so play shows the actual target, not level 0.
    cfg.scene.terrain.max_init_terrain_level = None
    # Drop the flat patches. `randomize_terrain` drops the robot on a random
    # sub-terrain each reset, and landing on flat ground (or worse, the border)
    # makes it look like the steps are missing. For inspection we always want to
    # be standing on them.
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    gen.sub_terrains = {k: v for k, v in gen.sub_terrains.items() if k != "flat"}

  return cfg


def unitree_r1_three_step_motion_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fixed blind motion: climb the sim2sim staircase, then stand still."""
  cfg = unitree_r1_stairs_blind_env_cfg(play=play)
  # The non-foot contact sensor keeps all body/stair pairs available. A single
  # inspection environment can therefore have more simultaneous contacts than
  # the rough-task default buffer.
  cfg.sim.nconmax = 128

  assert cfg.scene.terrain is not None
  height_range = (
    (MOTION_STAIR_HEIGHT, MOTION_STAIR_HEIGHT)
    if play
    else (0.05, MOTION_STAIR_HEIGHT)
  )
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(2.0, 2.0),
    border_width=10.0,
    # One patch is enough for visual inspection; training still uses a grid of
    # difficulty levels and independent copies for thousands of robots.
    num_rows=1 if play else 10,
    num_cols=1 if play else 10,
    curriculum=not play,
    difficulty_range=(1.0, 1.0) if play else (0.0, 1.0),
    sub_terrains={
      "r1_three_steps": BoxR1ThreeStepTerrainCfg(
        proportion=1.0,
        step_height_range=height_range,
      )
    },
  )
  cfg.scene.terrain.max_init_terrain_level = None if play else 2

  # The motion starts aligned with the staircase. Small perturbations prevent
  # the policy from depending on one exact floating-point spawn pose.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.02, 0.02),
    "y": (-0.03, 0.03),
    "z": (0.0, 0.0),
    "yaw": (-0.03, 0.03),
  }

  cfg.commands["twist"] = mdp.StairMotionCommandCfg(
    resampling_time_range=(1.0e9, 1.0e9),
    forward_speed=MOTION_FORWARD_SPEED,
    climb_duration=MOTION_CLIMB_DURATION,
  )
  # Keep the upper body quiet while the legs execute the scripted climb.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    **R1_ACTION_SCALE,
    "waist_.*": 0.04,
    ".*_shoulder_pitch.*": 0.06,
    ".*_shoulder_roll.*": 0.06,
    ".*_shoulder_yaw.*": 0.05,
    ".*_elbow.*": 0.05,
    ".*_wrist_roll.*": 0.04,
  }
  cfg.curriculum.pop("command_vel", None)
  for observation_group in ("actor", "critic"):
    cfg.observations[observation_group].terms["phase"].params[
      "period"
    ] = MOTION_GAIT_PERIOD

  # Only the fourteen sole collision geoms may touch the floor or stairs.
  foot_geom_names = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in range(1, 8)
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="stair_nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision$",
      exclude=foot_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (nonfoot_ground_cfg,)
  cfg.terminations["nonfoot_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 1.0},
  )

  # World-frame foot height is not a valid clearance target on elevated steps.
  cfg.rewards.pop("foot_clearance", None)

  # A broad Gaussian gives a stationary robot 78% of the velocity reward for a
  # 0.25 m/s command.  Tighten it for this fixed motion, and add an explicit
  # reward that is zero at rest so the policy must actually attempt to advance.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=mdp.track_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": 0.15},
  )
  cfg.rewards["stair_forward_velocity"] = RewardTermCfg(
    func=mdp.stair_forward_velocity,
    weight=2.0,
    params={"command_name": "twist"},
  )
  # The generic gait term already schedules the right foot first. Strengthen
  # it and require an 18 cm lift while the opposite foot is supporting. The
  # extra 3 cm clears the 15 cm stair lip without an exaggerated high step.
  cfg.rewards["foot_gait"].weight = 3.0
  cfg.rewards["foot_gait"].params["period"] = MOTION_GAIT_PERIOD
  # The generic pose reward favors the default standing pose. Keep a small
  # stabilizing contribution without letting it dominate swing-leg flexion.
  cfg.rewards["pose"].weight = 0.25
  cfg.rewards["body_orientation_l2"].weight = -3.0
  cfg.rewards["stair_alternating_foot_lift"] = RewardTermCfg(
    func=mdp.stair_alternating_foot_lift,
    weight=12.0,
    params={
      "period": MOTION_GAIT_PERIOD,
      "target_lift": 0.18,
      "target_forward": 0.24,
      "first_target_forward": 0.35,
      "first_step_position": 0.30,
      "step_tread": MOTION_STAIR_TREAD,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )
  cfg.rewards["stair_swing_upward_velocity"] = RewardTermCfg(
    func=mdp.stair_swing_foot_upward_velocity,
    weight=4.0,
    params={
      "period": MOTION_GAIT_PERIOD,
      "target_upward_velocity": 0.5,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )
  cfg.rewards["stair_swing_forward_velocity"] = RewardTermCfg(
    func=mdp.stair_swing_foot_forward_velocity,
    weight=5.0,
    params={
      "period": MOTION_GAIT_PERIOD,
      "target_forward_velocity": 0.8,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=("left_foot", "right_foot")
      ),
    },
  )
  cfg.rewards["stair_swing_leg_flexion"] = RewardTermCfg(
    func=mdp.stair_swing_leg_flexion,
    weight=6.0,
    params={
      "period": MOTION_GAIT_PERIOD,
      "command_name": "twist",
      "sensor_name": "feet_ground_contact",
      "hip_neutral": -0.1,
      "hip_target": -0.55,
      "knee_neutral": 0.3,
      "knee_target": 0.95,
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          "left_hip_pitch_joint",
          "left_knee_joint",
          "right_hip_pitch_joint",
          "right_knee_joint",
        ),
        preserve_order=True,
      ),
    },
  )
  cfg.rewards["stair_progress"] = RewardTermCfg(
    func=mdp.stair_forward_progress,
    weight=3.0,
    params={"target_distance": MOTION_TARGET_DISTANCE},
  )
  cfg.rewards["stair_top_stop"] = RewardTermCfg(
    func=mdp.stair_top_stop,
    weight=5.0,
    params={
      "target_distance": MOTION_TARGET_DISTANCE,
      "max_distance": 1.38,
      "min_height": 0.85,
      "command_name": "twist",
      "velocity_std": 0.2,
    },
  )

  # Five seconds to climb and three seconds to settle on the landing.
  cfg.episode_length_s = 8.0
  return cfg


def unitree_r1_stairs_perceptive_env_cfg(
  play: bool = False,
  height_max: float = STAIR_HEIGHT_MAX,
  tread: float = STAIR_TREAD,
) -> ManagerBasedRlEnvCfg:
  """Same terrain as the blind task, but the actor keeps its height scan.

  This is the control for the blind experiment, not a deployable policy: nothing
  on the robot produces a height scan today (see the camera repo). It answers
  "how much does seeing the step actually buy?" on identical terrain, which is
  what decides whether building the perception stack is worth it.

  Note the scan here is the ideal one - every ray, every step, no occlusion. The
  real camera is 10 Hz, head-mounted and cannot see under the feet, so treat the
  gap this measures as an upper bound on what perception could give.
  """
  cfg = unitree_r1_stairs_blind_env_cfg(
    play=play, height_max=height_max, tread=tread, scan_resolution=0.1
  )

  # Put height_scan back into the actor, matching the critic's term.
  if "height_scan" not in cfg.observations["actor"].terms:
    terms = dict(cfg.observations["actor"].terms)
    terms["height_scan"] = cfg.observations["critic"].terms["height_scan"]
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=terms,
      concatenate_terms=True,
      enable_corruption=cfg.observations["actor"].enable_corruption,
    )

  return cfg
