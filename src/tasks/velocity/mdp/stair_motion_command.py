"""Open-loop command used by the fixed three-step R1 motion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


class StairMotionCommand(CommandTerm):
  """Command forward for a fixed time, then command a complete stop."""

  cfg: "StairMotionCommandCfg"

  def __init__(self, cfg: "StairMotionCommandCfg", env):
    super().__init__(cfg, env)
    self._command = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._command[env_ids] = 0.0

  def _update_command(self) -> None:
    elapsed = self._env.episode_length_buf * self._env.step_dt
    moving = elapsed < self.cfg.climb_duration
    self._command.zero_()
    self._command[:, 0] = torch.where(
      moving,
      torch.full_like(elapsed, self.cfg.forward_speed),
      torch.zeros_like(elapsed),
    )


@dataclass(kw_only=True)
class StairMotionCommandCfg(CommandTermCfg):
  """Configuration for the fixed three-step command profile."""

  forward_speed: float = 0.25
  climb_duration: float = 5.0

  def build(self, env) -> StairMotionCommand:
    return StairMotionCommand(self, env)
