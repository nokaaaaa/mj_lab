"""Summarise every finished training run into a markdown table.

Reads the tensorboard event files rather than the console logs, so it works the
same whether a run was launched by hand or by one of the queues.

  python scripts/collect_results.py --out ~/traffic-officer-humanoid/docs/08-results.md
"""

import glob
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import tyro
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# tag -> (column heading, higher-is-better)
METRICS = {
  "Train/mean_reward": ("報酬", True),
  "Train/mean_episode_length": ("エピソード長", True),
  "Metrics/motion/error_body_pos": ("位置誤差[m]", False),
  "Episode_Termination/fell_over": ("転倒", False),
  "Episode_Termination/time_out": ("完走", True),
}

# Where a run came from, for grouping. Order matters: first match wins.
GROUPS = (
  ("r1_tracking", "誘導動作（モーション模倣）"),
  ("r1_velocity", "段差・歩行"),
)


@dataclass
class Run:
  experiment: str
  run_dir: str
  steps: int = 0
  values: dict = field(default_factory=dict)
  has_onnx: bool = False
  motion: str = ""


def read_run(events_path: str) -> Run | None:
  run_dir = os.path.dirname(events_path)
  experiment = os.path.basename(os.path.dirname(run_dir))
  ea = EventAccumulator(events_path, size_guidance={"scalars": 0})
  try:
    ea.Reload()
  except Exception:
    return None
  tags = set(ea.Tags()["scalars"])
  if not tags:
    return None

  run = Run(experiment=experiment, run_dir=run_dir)
  run.has_onnx = os.path.exists(os.path.join(run_dir, "policy.onnx"))
  # Run directories are <timestamp> or <timestamp>_<clip>. Match the timestamp
  # explicitly - splitting on "_" mangles any directory renamed by hand
  # (baseline_no_smoothness_<timestamp> would otherwise read as a clip name).
  base = os.path.basename(run_dir)
  m = re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(.+)$", base)
  if m:
    run.motion = m.group(1)
  elif not re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", base):
    run.motion = f"({base})"   # hand-renamed directory; show it as-is

  for tag, _ in METRICS.items():
    if tag in tags:
      scalars = ea.Scalars(tag)
      if scalars:
        run.values[tag] = scalars[-1].value
        run.steps = max(run.steps, scalars[-1].step)
  return run if run.values else None


def main(
  logs: str = "~/unitree_rl_mjlab/logs/rsl_rl",
  out: str = "~/traffic-officer-humanoid/docs/08-results.md",
  min_steps: int = 50,
) -> None:
  """Write a results table for every run with at least `min_steps` iterations."""
  logs = os.path.expanduser(logs)
  out = os.path.expanduser(out)

  runs = []
  for events in sorted(glob.glob(os.path.join(logs, "*", "*", "events.out.tfevents.*"))):
    r = read_run(events)
    if r and r.steps >= min_steps:
      runs.append(r)

  lines = [
    "# 08. 学習結果",
    "",
    f"`scripts/collect_results.py` が自動生成（{datetime.now():%Y-%m-%d %H:%M}）。",
    "**手で編集しないこと** — 次回の集計で上書きされる。",
    "",
    "各行は tensorboard のイベントから読んだ最終値。`onnx` 列が ✅ のものは",
    "デプロイ可能な `policy.onnx` が書き出されている。",
    "",
  ]

  for key, title in GROUPS:
    group = [r for r in runs if r.experiment.startswith(key)]
    if not group:
      continue
    lines += [f"## {title}", ""]
    heads = ["実験", "元動画", "iter"] + [h for h, _ in METRICS.values()] + ["onnx"]
    lines.append("| " + " | ".join(heads) + " |")
    lines.append("|" + "---|" * len(heads))
    for r in sorted(group, key=lambda x: (x.motion, x.run_dir)):
      cells = [
        f"`{r.experiment}`",
        f"**{r.motion}**" if r.motion else "—",
        str(r.steps),
      ]
      for tag in METRICS:
        v = r.values.get(tag)
        cells.append("—" if v is None else (f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"))
      cells.append("✅" if r.has_onnx else "—")
      lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

  lines += [
    "## 読み方",
    "",
    "| 列 | 意味 |",
    "|---|---|",
    "| 報酬 | 高いほど良い。タスク間で比較しても意味はない（報酬の定義が違う） |",
    "| エピソード長 | 転倒せず続いた長さ。模倣タスクは上限500（10秒） |",
    "| 位置誤差 | 参照モーションとの差。模倣タスクのみ |",
    "| 転倒 / 完走 | 1イテレーションあたりの終了回数。**転倒が減り完走が増えていれば良い** |",
    "",
    "元動画の命名は `進行`→`shinkou` / `停止`→`teishi` / `幅寄せ`→`habayose` +",
    "DJIの4桁ID。動画・pkl・csv・npz・学習結果まで同じ名前で一貫している。",
    "",
    f"総run数: {len(runs)}",
  ]

  os.makedirs(os.path.dirname(out), exist_ok=True)
  with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  print(f"wrote {out}  ({len(runs)} runs)")


if __name__ == "__main__":
  tyro.cli(main)
