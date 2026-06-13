# Machine-specific overrides live in local.mk (git-ignored).
# Copy local.mk.example → local.mk and fill in for your machine.
-include local.mk

# ── Machine-specific variables (set in local.mk) ──────────────────────────────
NXF     ?= nextflow/main.nf
WORKDIR ?=                       # e.g. -w /scratch/nextflow_work
PBS_BIN ?= /opt/pbs/bin          # override if qsub is not on PATH

# ── Training hyperparameters (override on CLI: make train-transformer EPOCHS=200) ──
EPOCHS      ?= 100
WORKERS     ?= 8
LR          ?= 1e-3
PATIENCE    ?= 10
LOSS        ?= mse
GRAD_ACCUM  ?= 1
NUM_GPUS    ?= 4

# VAE defaults
VAE_BATCH   ?= 256

# Transformer defaults (smaller batch — VIIRS 1024×1024 tiles are memory-heavy)
TR_BATCH    ?= 4
TR_LR       ?= 5e-4
MED_IN_CH   ?= 18

# ── Local smoke tests ─────────────────────────────────────────────────────────
train-vae:
	PYTHONPATH=src uv run python src/models/training.py \
		--model       vae \
		--batch_size  2 \
		--num_epochs  3 \
		--num_workers 2 \
		--patience    2 \
		--loss_fn     $(LOSS) \
		2>&1 | tee training_vae.log

train-transformer:
	PYTHONPATH=src uv run python src/models/training.py \
		--model       transformer \
		--batch_size  1 \
		--num_epochs  3 \
		--num_workers 2 \
		--patience    2 \
		--med_in_ch   $(MED_IN_CH) \
		--loss_fn     $(LOSS) \
		2>&1 | tee training_transformer.log

# ── HPC — VAE 4-GPU DDP ───────────────────────────────────────────────────────
train-vae-hpc-ddp:
	PATH=$(PBS_BIN):$$PATH nextflow run $(NXF) \
		--model            vae \
		--ddp \
		--num_gpus         $(NUM_GPUS) \
		--num_epochs       $(EPOCHS) \
		--batch_size       $(VAE_BATCH) \
		--num_workers      $(WORKERS) \
		--learning_rate    $(LR) \
		--patience         $(PATIENCE) \
		--loss_fn          $(LOSS) \
		--grad_accum_steps $(GRAD_ACCUM) \
		$(WORKDIR) \
		-resume \
		-profile spacehpc

# ── HPC — Transformer 4-GPU DDP ──────────────────────────────────────────────
train-transformer-hpc-ddp:
	PATH=$(PBS_BIN):$$PATH nextflow run $(NXF) \
		--model            transformer \
		--ddp \
		--num_gpus         $(NUM_GPUS) \
		--num_epochs       $(EPOCHS) \
		--batch_size       $(TR_BATCH) \
		--num_workers      $(WORKERS) \
		--learning_rate    $(TR_LR) \
		--patience         $(PATIENCE) \
		--loss_fn          $(LOSS) \
		--grad_accum_steps $(GRAD_ACCUM) \
		--med_in_ch        $(MED_IN_CH) \
		$(WORKDIR) \
		-profile spacehpc

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=src uv run python -m pytest src/ -v

.PHONY: train-vae train-transformer train-vae-hpc-ddp train-transformer-hpc-ddp test
