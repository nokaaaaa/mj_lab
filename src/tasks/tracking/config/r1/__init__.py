from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_r1_flat_tracking_env_cfg
from .stairs_env_cfgs import unitree_r1_stairs_tracking_env_cfg
from .rl_cfg import unitree_r1_tracking_ppo_runner_cfg


def unitree_r1_stairs_tracking_ppo_runner_cfg():
  cfg = unitree_r1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "r1_stairs_tracking_2x"
  cfg.save_interval = 100
  # The action is now a correction around the reference joint pose. Keep
  # initial exploration small enough to preserve the demonstrated swing.
  cfg.actor.distribution_cfg["init_std"] = 0.5
  return cfg

register_mjlab_task(
  task_id="Unitree-R1-Tracking",
  env_cfg=unitree_r1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_r1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_r1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Tracking-No-State-Estimation",
  env_cfg=unitree_r1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_r1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_r1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-R1-Stairs-Tracking-2x",
  env_cfg=unitree_r1_stairs_tracking_env_cfg(),
  play_env_cfg=unitree_r1_stairs_tracking_env_cfg(play=True),
  rl_cfg=unitree_r1_stairs_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
