#!/usr/bin/env nextflow

/*
 * Dengue-Africa training pipeline — supports VAE and Transformer.
 *
 * Steps:
 *   1. prepare_data  (optional) — rasterise Argentina weekly cases onto netCDF grid
 *   2. train_model              — train VAE or Transformer (--model vae|transformer)
 *
 * Usage (local):
 *   nextflow run main.nf -profile local --model vae --num_epochs 5 --batch_size 2
 *
 * Usage (HPC — 4-GPU DDP, VAE):
 *   nextflow run main.nf -profile spacehpc \
 *       --model vae --ddp --num_gpus 4 --batch_size 256 --num_workers 8 --num_epochs 100
 *
 * Usage (HPC — 4-GPU DDP, Transformer):
 *   nextflow run main.nf -profile spacehpc \
 *       --model transformer --ddp --num_gpus 4 --batch_size 4 --num_workers 8 --num_epochs 100
 */

venv_path = "${params.project_root}/.venv/bin/activate"


// ── Process 1: prepare Argentina dengue data (optional) ──────────────────────
process prepare_data {
    label 'cpu'
    tag 'argentina_dengue'

    beforeScript "source ${venv_path}"

    output:
    val true, emit: done

    script:
    """
    cd ${params.project_root}
    PYTHONPATH=${params.project_root}/src \
        uv run python ${params.project_root}/src/data/prepare_argentina_dengue.py
    """
}


// ── Process 2: train model ────────────────────────────────────────────────────
process train_model {
    label 'gpu'
    tag "${params.model}_training"

    beforeScript """
        module load PrgEnv-nvidia
    """

    input:
    val ready   // gate: waits for prepare_data if it ran, else fires immediately

    script:
    def ddp_flag     = params.ddp     ? '--ddp'     : ''
    def kl_flag      = params.use_kl  ? '--use_kl'  : ''
    def titok_flag   = params.use_titok ? '--use_titok' : ''
    def sm_flag      = params.add_sm  ? "--add_sm --sm_data_path ${params.sm_data_path}" : ''
    def lc_flag      = params.add_lc  ? "--add_lc --lc_data_path ${params.lc_data_path}" : ''

    // --no-sync: use the pre-built venv as-is; compute nodes have no internet
    def launch = params.ddp
        ? "uv run --no-sync torchrun --standalone --nproc_per_node=${params.num_gpus}"
        : "uv run --no-sync python"

    // Model-specific flags — only passed when relevant to avoid argparse conflicts
    def vae_flags = params.model == 'vae' ? """\
            --beta_kl           ${params.beta_kl} \\
            --latent_channels   ${params.latent_channels} \\
            --layers_per_block  ${params.layers_per_block} \\
            --norm_num_groups   ${params.norm_num_groups} \\
            ${kl_flag}""" : ""

    def transformer_flags = params.model == 'transformer' ? """\
            --med_in_ch                 ${params.med_in_ch} \\
            --swin_model                ${params.swin_model} \\
            --titok_backbone            ${params.titok_backbone} \\
            --titok_num_latent_tokens   ${params.titok_num_latent_tokens} \\
            ${titok_flag}""" : ""

    """
    cd ${params.project_root}

    # NCCL/HDF5 env vars are set globally in nextflow.config env{} block.

    PYTHONPATH=${params.project_root}/src \\
        ${launch} ${params.project_root}/src/models/training.py \\
            --model             ${params.model} \\
            --batch_size        ${params.batch_size} \\
            --num_epochs        ${params.num_epochs} \\
            --num_workers       ${params.num_workers} \\
            --learning_rate     ${params.learning_rate} \\
            --patience          ${params.patience} \\
            --train_split       ${params.train_split} \\
            --loss_fn           ${params.loss_fn} \\
            --grad_accum_steps  ${params.grad_accum_steps} \\
            --cleanup_every     ${params.cleanup_every} \\
            ${ddp_flag} \\
            ${vae_flags} \\
            ${transformer_flags} \\
            ${sm_flag} \\
            ${lc_flag}
    """
}


// ── Workflow ──────────────────────────────────────────────────────────────────
workflow {

    if (params.prepare_data) {
        prepare_data()
        ready = prepare_data.out.done
    } else {
        ready = Channel.value(true)
    }

    train_model(ready)
}
