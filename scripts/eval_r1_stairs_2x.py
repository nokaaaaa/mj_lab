"""Evaluate a stair policy by contact, motion progress, and landing stability."""

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.tracking.config.r1 import (
  unitree_r1_stairs_tracking_env_cfg,
  unitree_r1_stairs_tracking_ppo_runner_cfg,
)
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint")
  parser.add_argument("--motion-file", default="src/assets/motions/r1/r1_stairs_step_by_step_2x.npz")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--no-startup-pause", action="store_true")
  args = parser.parse_args()

  cfg = unitree_r1_stairs_tracking_env_cfg(play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.commands["motion"].motion_file = args.motion_file
  cfg.commands["motion"].debug_vis = False
  if args.no_startup_pause:
    cfg.commands["motion"].startup_contact_sensor = ""
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cuda:0"))
  runner = MotionTrackingOnPolicyRunner(
    env, asdict(unitree_r1_stairs_tracking_ppo_runner_cfg()), device="cuda:0"
  )
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  policy = runner.get_inference_policy(device="cuda:0")
  obs = env.get_observations()
  command = env.unwrapped.command_manager.get_term("motion")
  robot = env.unwrapped.scene["robot"]
  sensor = env.unwrapped.scene["stair_feet_contact"]
  origins = env.unwrapped.scene.env_origins
  best_frame = torch.zeros(args.num_envs, device="cuda:0", dtype=torch.long)
  first_step = torch.zeros(args.num_envs, device="cuda:0", dtype=torch.bool)
  first_swing_z = torch.full((args.num_envs,), -float("inf"), device="cuda:0")
  first_swing_x = torch.full((args.num_envs,), -float("inf"), device="cuda:0")
  success = torch.zeros(args.num_envs, device="cuda:0", dtype=torch.bool)
  alive = torch.ones(args.num_envs, device="cuda:0", dtype=torch.bool)
  first_swing_trace = []
  swings = ((1, 1, 40), (1, 0, 68), (2, 0, 120), (2, 1, 148), (3, 1, 175), (3, 0, 202))
  swing_reached = torch.zeros((6, args.num_envs), device="cuda:0", dtype=torch.bool)
  swing_max_z = torch.full((6, args.num_envs), -float("inf"), device="cuda:0")
  swing_max_x = torch.full((6, args.num_envs), -float("inf"), device="cuda:0")
  swing_sample_x = torch.full((6, args.num_envs), float("nan"), device="cuda:0")
  swing_sample_z = torch.full((6, args.num_envs), float("nan"), device="cuda:0")
  landed = torch.zeros((6, args.num_envs), device="cuda:0", dtype=torch.bool)
  for _ in range(env.max_episode_length):
    with torch.no_grad():
      actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    frame = command.time_steps
    best_frame = torch.maximum(best_frame, torch.where(alive, frame, 0))
    sites = robot.data.site_pos_w
    left = sites[:, robot.site_names.index("left_foot")]
    right = sites[:, robot.site_names.index("right_foot")]
    force = torch.linalg.vector_norm(sensor.data.force, dim=-1).reshape(args.num_envs, 2)
    x_r = right[:, 0] - origins[:, 0]
    z_r = right[:, 2] - origins[:, 2]
    for sample_frame in (20, 25, 30, 35, 40, 45):
      sample = alive & (frame == sample_frame)
      if sample.any():
        first_swing_trace.append((sample_frame, x_r[sample].mean().item(), z_r[sample].mean().item(), int(sample.sum().item())))
    swing = alive & (frame >= 15) & (frame <= 38)
    first_swing_z = torch.maximum(first_swing_z, torch.where(swing, z_r, -float("inf")))
    first_swing_x = torch.maximum(first_swing_x, torch.where(swing, x_r, -float("inf")))
    first_step |= alive & (frame >= 31) & (frame <= 56) & (x_r > 0.22) & (x_r < 0.42) & ((z_r - 0.15).abs() < 0.08) & (force[:, 1] > 10)
    x_l = left[:, 0] - origins[:, 0]
    z_l = left[:, 2] - origins[:, 2]
    foot_x = torch.stack((x_l, x_r), dim=1)
    foot_z = torch.stack((z_l, z_r), dim=1)
    for i, (step, side, center) in enumerate(swings):
      in_swing = alive & (frame >= center - 25) & (frame <= center - 2)
      swing_reached[i] |= in_swing
      swing_max_z[i] = torch.maximum(swing_max_z[i], torch.where(in_swing, foot_z[:, side], -float("inf")))
      swing_max_x[i] = torch.maximum(swing_max_x[i], torch.where(in_swing, foot_x[:, side], -float("inf")))
      at_sample = alive & (frame == center - 10)
      swing_sample_x[i] = torch.where(at_sample, foot_x[:, side], swing_sample_x[i])
      swing_sample_z[i] = torch.where(at_sample, foot_z[:, side], swing_sample_z[i])
      x0 = 0.20 + (step - 1) * 0.24
      in_landing = alive & (frame >= center - 9) & (frame <= center + 16)
      landed[i] |= in_landing & (foot_x[:, side] > x0 + 0.02) & (foot_x[:, side] < x0 + 0.22) & ((foot_z[:, side] - step * 0.15).abs() < 0.08) & (force[:, side] > 10)
    root_x = robot.data.root_link_pos_w[:, 0] - origins[:, 0]
    success |= alive & (frame >= 250) & (root_x > 0.72) & (x_l > 0.68) & (x_r > 0.68) & ((z_l - 0.45).abs() < 0.08) & ((z_r - 0.45).abs() < 0.08) & (force.min(dim=-1).values > 10)
    alive &= dones == 0
    if not alive.any():
      break
  print(f"first_step={first_step.float().mean().item():.3f} success={success.float().mean().item():.3f} best_frame_mean={best_frame.float().mean().item():.1f}/{command.motion.time_step_total}")
  print(f"first_swing_max_z={first_swing_z.nan_to_num(neginf=0).mean().item():.3f}m first_swing_max_x={first_swing_x.nan_to_num(neginf=0).mean().item():.3f}m")
  print("first_swing_trace(frame,x,z,n)=", first_swing_trace)
  print("swing step side reached max_z max_x x_at_center-10 z_at_center-10 landed")
  for i, (step, side, _) in enumerate(swings):
    reached = swing_reached[i]
    n = int(reached.sum().item())
    if n:
      print(step, "left" if side == 0 else "right", n,
            f"{swing_max_z[i, reached].mean().item():.3f}",
            f"{swing_max_x[i, reached].mean().item():.3f}",
            f"{swing_sample_x[i, reached].nanmean().item():.3f}",
            f"{swing_sample_z[i, reached].nanmean().item():.3f}",
            f"{landed[i].sum().item()}/{n}")
    else:
      print(step, "left" if side == 0 else "right", 0, "unreached")
  env.close()


if __name__ == "__main__":
  main()
