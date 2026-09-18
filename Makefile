# Prefer a project-local venv; fall back to the environment used for the
# current training runs if none exists yet.
FALLBACK_PYTHON := /home/yuki/.local/share/Trash/files/unitree_rl_mjlab/.venv/bin/python
PYTHON := $(firstword $(wildcard .venv/bin/python) $(wildcard $(FALLBACK_PYTHON)))

# Task whose checkpoints `make check` should play. A checkpoint's run
# directory only records `experiment_name`, which several tasks share, so
# this can't be auto-detected reliably; override it per invocation instead.
TASK ?= Unitree-R1-Stairs-Tracking-2x

# GUI file picker used to choose the checkpoint (Ubuntu's standard dialog).
ZENITY := $(shell command -v zenity 2>/dev/null)

.PHONY: check

# Open a GUI file chooser (rooted at logs/rsl_rl) to pick a model_*.pt
# checkpoint, then play it in the MuJoCo viewer, e.g.:
#   make check
#   make check TASK=Unitree-R1-Stairs-Blind   # fallback task override
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
	"$(PYTHON)" scripts/play.py "$(TASK)" \
		--checkpoint-file="$$CKPT" \
		--num-envs=1 \
		--viewer=native
