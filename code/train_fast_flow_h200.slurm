#!/bin/bash
# Fully spiking SpikingFastFlowNet experiment on the one-off H200 (gpu-h200-test).
# Trainings run strictly ONE AFTER ANOTHER inside this single job:
#   phase 1: 3 architectures x 120 epochs (EMA + h/v flips, light rate penalty)
#   phase 2: best phase-1 architecture (lowest best val EPE) retrained for 500 epochs
#   phase 3: from the phase-2 model: sparsity fine-tune (rate penalty 2), INT8 QAT fine-tune
#   sbatch train_fast_flow_h200.slurm
#SBATCH --job-name=spk_h200
#SBATCH --partition=gpu-h200-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home/geoffroyk/miniconda3/etc/profile.d/conda.sh
conda activate eventvpr   # spikingjelly 0.0.0.0.15: accepts the 'triton' SiLIF backend
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /home/geoffroyk/StateSpace_SNN
export MLFLOW_TRACKING_URI=file:$PWD/mlruns
export PYTHONUNBUFFERED=1
J=$SLURM_JOB_ID
echo "Job $J on $SLURMD_NODENAME, GPU $CUDA_VISIBLE_DEVICES, start $(date)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

COMMON="--arch spiking --T 5 --amp --vflip --num_workers 16"
declare -A CFG
CFG[B]="--channels 16 32 64 128 --dec_channels 16 32 64 128"                           # 394k
CFG[C]="--channels 32 48 96 112 --dec_channels 24 48 96 112 --res_blocks 0 0 1 1"      # 727k
CFG[D]="--channels 32 48 96 128 --dec_channels 24 48 96 128 --res_blocks 0 0 1 2"      # 973k

python -c "import train_fast_flow as t; from dsec_optical.flow_cache import CachedFlowDataset as D, FLOW_TRAIN_RECORDINGS as R; D(R)[0]; print('preflight ok')" \
    || { echo "preflight failed"; exit 1; }

echo "=== phase 1 $(date)"
for k in B C D; do
    python train_fast_flow.py $COMMON ${CFG[$k]} --ema 0.999 --rate_penalty 0.5 --epochs 120 --run_name h200_p1_$k
done

BEST=$(python - <<EOF
import torch
best = {}
for k in "BCD":
    try:
        best[k] = torch.load(f"checkpoints/fast_flow/h200_p1_{k}_j$J.pt", map_location="cpu", weights_only=False)["val"]["epe"]
    except FileNotFoundError:
        pass
print(min(best, key=best.get))
EOF
)
echo "=== phase 2 $(date): best phase-1 config = $BEST"
python train_fast_flow.py $COMMON ${CFG[$BEST]} --ema 0.999 --rate_penalty 0.5 --epochs 500 --run_name h200_p2_$BEST
P2=checkpoints/fast_flow/h200_p2_${BEST}_j$J.pt

echo "=== phase 3 $(date)"
python train_fast_flow.py $COMMON ${CFG[$BEST]} --ema 0.999 --rate_penalty 2 --epochs 40 --lr 1e-4 \
    --warmup_epochs 0 --init $P2 --run_name h200_p3_sparse_$BEST
python train_fast_flow.py $COMMON ${CFG[$BEST]} --rate_penalty 0.5 --epochs 20 --lr 1e-4 \
    --warmup_epochs 0 --init $P2 --qat_bits 8 --run_name h200_p3_int8_$BEST
echo "End $(date)"
