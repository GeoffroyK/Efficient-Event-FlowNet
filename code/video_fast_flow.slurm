#!/bin/bash
# Validation video (visualize_dsec_flow_video.py) of a train_fast_flow.py checkpoint on ONE val recording:
#   sbatch video_fast_flow.slurm <ckpt> <val recording> <out stem>
#SBATCH --job-name=video_flow
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home/geoffroyk/miniconda3/etc/profile.d/conda.sh
conda activate eventvpr
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /home/geoffroyk/StateSpace_SNN
nvidia-smi --query-gpu=name --format=csv,noheader
python visualize_dsec_flow_video.py --fast_ckpt "$1" --val_seq "$2" --out_mp4 "$3.mp4" --out_gif "$3.gif"
