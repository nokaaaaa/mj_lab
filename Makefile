# Prefer a project-local venv; fall back to the environment used for the
# current training runs if none exists yet.
FALLBACK_PYTHON := /home/yuki/.local/share/Trash/files/unitree_rl_mjlab/.venv/bin/python
PYTHON := $(firstword $(wildcard .venv/bin/python) $(wildcard $(FALLBACK_PYTHON)))

# Task whose checkpoints `make check` should play. A checkpoint's run
# directory only records `experiment_name`, which several tasks share, so
# this can't be auto-detected reliably; override it per invocation instead.
TASK ?= Unitree-R1-Stairs-Tracking-2x

# Tracking tasks need an explicit reference motion (play.py leaves it unset
# otherwise); override alongside TASK when checking a different task.
MOTION_FILE ?= src/assets/motions/r1/r1_stairs_step_by_step_2x.npz

# GUI file picker used to choose the checkpoint (Ubuntu's standard dialog).
ZENITY := $(shell command -v zenity 2>/dev/null)

# TensorBoard settings for `make tensor`. Ports 6006/6007 are already used
# by other long-running TensorBoard instances on this machine, so default
# to a free one; override if it's taken too.
TENSORBOARD_PORT ?= 6008
TENSORBOARD_LOGDIR ?= logs/rsl_rl

.PHONY: check tensor

# Open a GUI file chooser (rooted at logs/rsl_rl) to pick a model_*.pt
# checkpoint, then play it in the MuJoCo viewer, e.g.:
#   make check
#   make check TASK=Unitree-R1-Stairs-Blind MOTION_FILE=path/to/motion.npz
check:
	@if [ -z "$(PYTHON)" ]; then \
		echo "No virtualenv python found (looked for .venv/bin/python and $(FALLBACK_PYTHON))." >&2; \
		echo "Pass PYTHON=/path/to/python to override." >&2; \
		exit 1; \
	fi
	@if [ -z "$(ZENITY)" ]; then \
		echo "zenity not found; install it with 'sudo apt install zenity' for the file picker." >&2; \
		exit 1; \
	fi
	@LATEST_RUN=$$(find logs/rsl_rl -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null \
		| sort -rn | head -n1 | cut -d' ' -f2-); \
	START_DIR=$${LATEST_RUN:-logs/rsl_rl}; \
	CKPT=$$("$(ZENITY)" --file-selection \
		--title="Select a checkpoint (model_*.pt)" \
		--filename="$(CURDIR)/$$START_DIR/" 2>/dev/null); \
	if [ -z "$$CKPT" ]; then \
		echo "No checkpoint selected." >&2; \
		exit 1; \
	fi; \
	echo "Checkpoint: $$CKPT"; \
	echo "Task: $(TASK)"; \
	echo "Motion file: $(MOTION_FILE)"; \
	"$(PYTHON)" scripts/play.py "$(TASK)" \
		--checkpoint-file="$$CKPT" \
		--motion-file="$(MOTION_FILE)" \
		--num-envs=1 \
		--viewer=native

# Start (or reuse) a TensorBoard server over all training runs under
# logs/rsl_rl/ (all experiments, including r1_stairs_tracking_2x) and open
# it in the browser, e.g.:
#   make tensor
#   make tensor TENSORBOARD_PORT=6009 TENSORBOARD_LOGDIR=logs/rsl_rl/r1_stairs_tracking_2x
tensor:
	@if [ -z "$(PYTHON)" ]; then \
		echo "No virtualenv python found (looked for .venv/bin/python and $(FALLBACK_PYTHON))." >&2; \
		echo "Pass PYTHON=/path/to/python to override." >&2; \
		exit 1; \
	fi
	@URL="http://localhost:$(TENSORBOARD_PORT)"; \
	if ss -ltn 2>/dev/null | grep -q ":$(TENSORBOARD_PORT) "; then \
		echo "TensorBoard already running at $$URL"; \
	else \
		mkdir -p logs; \
		nohup "$(PYTHON)" -m tensorboard.main \
			--logdir "$(TENSORBOARD_LOGDIR)" --port $(TENSORBOARD_PORT) --bind_all \
			> logs/tensorboard_$(TENSORBOARD_PORT).log 2>&1 & \
		echo "Starting TensorBoard on $(TENSORBOARD_LOGDIR) at $$URL ..."; \
		sleep 3; \
	fi; \
	command -v xdg-open >/dev/null 2>&1 && xdg-open "$$URL" >/dev/null 2>&1 & \
	echo "Open $$URL if the browser didn't launch automatically."
