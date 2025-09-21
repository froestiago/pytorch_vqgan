#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=CIACD

OUTPUT_NAME="eq_vae_2"

OUTPUT_ROOT="./outputs/$OUTPUT_NAME"
LOG_DIR="$OUTPUT_ROOT/logs"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/slurm_%j.out")
exec 2> >(tee -a "$LOG_DIR/slurm_%j.err" >&2)

# Conda init (non-interactive)
__conda_setup="$('/home/tiagofroes/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/tiagofroes/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/tiagofroes/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/tiagofroes/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate bio_ldm

# Launch script
python3 -W ignore train_full_rec_eq_vae.py \
    --batch_size 2 \
    --num_workers 16 \
    --kl_loss_weight 1.e-6 \
    --n_epochs 100 \
    --n_epochs_warm_up 0 \
    --val_interval 1 \
    --save_interval 1 \
    --output_dir "$OUTPUT_NAME" \
    --learning_rate 1.e-4 \
    --accumulation_steps 64
