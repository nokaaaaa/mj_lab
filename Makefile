# Prefer a project-local venv; fall back to the environment used for the
# current training runs if none exists yet.
FALLBACK_PYTHON := /home/yuki/.local/share/Trash/files/unitree_rl_mjlab/.venv/bin/python
PYTHON := $(firstword $(wildcard .venv/bin/python) $(wildcard $(FALLBACK_PYTHON)))

# Task whose checkpoints `make check` should look for and play.
TASK ?= Unitree-R1-Three-Step-Motion
EXPERIMENT ?= r1_velocity

.PHONY: check

# Play the most recently saved checkpoint for TASK in the MuJoCo viewer, e.g.:
#   make check
#   make check TASK=Unitree-R1-Stairs-Blind
check:
	@if [ -z "$(PYTHON)" ]; then \
		echo "No virtualenv python found (looked for .venv/bin/python and $(FALLBACK_PYTHON))." >&2; \
		echo "Pass PYTHON=/path/to/python to override." >&2; \
		exit 1; \
	fi
	@CKPT=$$(find logs/rsl_rl/$(EXPERIMENT) -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
		| sort -rn | head -n1 | cut -d' ' -f2-); \
	if [ -z "$$CKPT" ]; then \
		echo "No checkpoint (model_*.pt) found under logs/rsl_rl/$(EXPERIMENT)/." >&2; \
		exit 1; \
	fi; \
	echo "Checkpoint: $$CKPT"; \
	echo "Task: $(TASK)"; \
	"$(PYTHON)" scripts/play.py $(TASK) \
		--checkpoint-file="$$CKPT" \
		--num-envs=1 \
		--viewer=native
