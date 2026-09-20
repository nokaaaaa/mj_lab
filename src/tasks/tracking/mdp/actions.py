"""Position actions expressed as corrections to the tracked motion."""

from dataclasses import dataclass
from typing import cast

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .commands import MotionCommand


@dataclass(kw_only=True)
class MotionJointPositionActionCfg(JointPositionActionCfg):
  command_name: str = "motion"

  def build(self, env) -> "MotionJointPositionAction":
    return MotionJointPositionAction(self, env)


class MotionJointPositionAction(JointPositionAction):
  """Let zero action follow the reference; learn small joint corrections."""

  def __init__(self, cfg: MotionJointPositionActionCfg, env):
    super().__init__(cfg, env)
    self._motion = cast(MotionCommand, env.command_manager.get_term(cfg.command_name))

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._processed_actions = (
      self._motion.joint_pos[:, self._target_ids]
      + self._raw_actions * self._scale
    )
