from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_r1_flat_env_cfg,
  unitree_r1_rough_env_cfg,
)
from .stairs_env_cfgs import (
  unitree_r1_stairs_blind_env_cfg,
  unitree_r1_stairs_perceptive_env_cfg,
  unitree_r1_three_step_motion_env_cfg,
)
from .rl_cfg import unitree_r1_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-R1-Rough",
  env_cfg=unitree_r1_rough_env_cfg(),
  play_env_cfg=unitree_r1_rough_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Flat",
  env_cfg=unitree_r1_flat_env_cfg(),
  play_env_cfg=unitree_r1_flat_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Stairs-Blind",
  env_cfg=unitree_r1_stairs_blind_env_cfg(),
  play_env_cfg=unitree_r1_stairs_blind_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Stairs-Perceptive",
  env_cfg=unitree_r1_stairs_perceptive_env_cfg(),
  play_env_cfg=unitree_r1_stairs_perceptive_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Three-Step-Motion",
  env_cfg=unitree_r1_three_step_motion_env_cfg(),
  play_env_cfg=unitree_r1_three_step_motion_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
