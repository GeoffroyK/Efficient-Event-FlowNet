#!/bin/bash
# Qualitative DSEC validation figure of a train_fast_flow.py checkpoint (args passed through):
#   sbatch plot_fast_flow.slurm --ckpt checkpoints/fast_flow/h200_p3_int8_D_j33273.pt
#SBATCH --job-name=plot_flow
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home/geoffroyk/miniconda3/etc/profile.d/conda.sh
conda activate eventvpr
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /home/geoffroyk/StateSpace_SNN
python plot_fast_flow.py "$@"
