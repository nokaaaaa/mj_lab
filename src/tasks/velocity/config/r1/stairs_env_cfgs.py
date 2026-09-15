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

from dataclasses import replace

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from src.tasks.velocity.config.r1.env_cfgs import unitree_r1_rough_env_cfg

# Measured on site: riser ~0.20 m. Tread read as 0.20-0.25 m from the photos;
# 0.20 is used because a narrower tread is the harder case, so training stays
# conservative with respect to the real steps.
STAIR_HEIGHT_MAX = 0.20
STAIR_TREAD = 0.20


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
