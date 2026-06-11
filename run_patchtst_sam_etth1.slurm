#!/bin/bash
#===============================================================================
# SLURM SCRIPT FOR CT-PatchTST (channel-attention) + SAM ON ETTh1
#
# Differences from the previous PatchTST+SAM script:
#   --channel_attn 1     temporal attention replaced by channel attention
#   --n_heads      1     7 channels is too few to split across multiple heads
#   --revin        1     mandatory — without it, channel attention rank-collapses
#                        even with SAM
#   --rho          0.5   fixed (no sweep). Override with RHO=... sbatch ... if
#                        you want to experiment later.
#
# SAM still doubles per-iteration work; ~2× wall-clock vs vanilla.
#
# USAGE:
#   sbatch run_ctpatchtst_sam_etth1.slurm                 # default (rho=0.5)
#   USE_SAM=0 sbatch run_ctpatchtst_sam_etth1.slurm       # ablation: SAM off
#   RHO=0.3   sbatch run_ctpatchtst_sam_etth1.slurm       # override if needed
#===============================================================================

#SBATCH --job-name=ctpatchtst-sam-etth1
#SBATCH --mail-user=shivansh@prl.res.in
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04-00:00:00
#SBATCH --output=logs/ctpatchtst_sam_etth1_%j.log
#SBATCH --error=logs/ctpatchtst_sam_etth1_%j.error

#===============================================================================
# HEADER
#===============================================================================
echo "=========================================================================="
echo "    CT-PatchTST (channel-attention) + SAM — ETTh1"
echo "=========================================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start time: $(date)"
echo "=========================================================================="

#===============================================================================
# SETUP
#===============================================================================
mkdir -p logs/LongForecasting

WORK_DIR="/home/akashg/time-series/python-files/PatchTST/PatchTST_supervised"

cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }
echo "Working directory: $(pwd)"

source /home/akashg/miniconda3/etc/profile.d/conda.sh
conda activate informerworking

echo "Python:  $(which python)"
echo "Version: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs:    $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================================================="

if [ ! -f "dataset/ETTh1.csv" ]; then
    echo "ERROR: dataset/ETTh1.csv not found."
    exit 1
fi
echo "Data file found: dataset/ETTh1.csv"
echo "=========================================================================="

#===============================================================================
# SAM CONFIGURATION
#===============================================================================
USE_SAM="${USE_SAM:-1}"
RHO="${RHO:-0.5}"               # FIXED at 0.5 per recommendation (no sweep)
ADAPTIVE="${ADAPTIVE:-0}"

SAM_FLAGS=""
if [ "$USE_SAM" -eq 1 ]; then
    SAM_FLAGS="--use_sam --rho $RHO"
    if [ "$ADAPTIVE" -eq 1 ]; then
        SAM_FLAGS="$SAM_FLAGS --adaptive_sam"
    fi
    echo "SAM:          ENABLED"
    echo "  rho:        $RHO"
    echo "  adaptive:   $ADAPTIVE"
else
    echo "SAM:          DISABLED (ablation run)"
fi

#===============================================================================
# HYPERPARAMETERS
#
# Differences from the vanilla PatchTST recipe for ETTh1:
#   - n_heads: 4 → 1   (7 channel tokens; don't split across heads)
#   - channel_attn = 1 (the whole point of this script)
#
# Kept small (d_model=16, d_ff=128, e_layers=3) to match the paper's ETTh1
# config; channel attention over 7 tokens doesn't need a bigger model.
#===============================================================================
model_name=PatchTST          # file/class; --channel_attn 1 makes it CT-PatchTST
seq_len=336
random_seed=2021

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1_CT
data_name=ETTh1

if [ "$USE_SAM" -eq 1 ]; then
    TRAIN_EPOCHS=150
    PATIENCE=20
else
    TRAIN_EPOCHS=100
    PATIENCE=100
fi

echo "Model:           CT-PatchTST  (channel_attn=1)"
echo "Look-back:       seq_len=$seq_len"
echo "Patch / stride:  16 / 8"
echo "Enc layers:      3"
echo "Heads:           1    (channel attention over 7 tokens — single head)"
echo "d_model / d_ff:  16 / 128"
echo "RevIN:           ON   (required for stable channel attention)"
echo "Horizons:        96 192 336 720"
echo "Epochs/patience: $TRAIN_EPOCHS / $PATIENCE,  batch 128,  lr 1e-4"
echo "=========================================================================="

#===============================================================================
# TRAINING LOOP — 4 prediction horizons
#===============================================================================
for pred_len in 96 192 336 720; do

    echo ""
    echo ">>> Starting pred_len=$pred_len  at $(date)"
    echo "------------------------------------------------------------"

    python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path  $root_path_name \
        --data_path  $data_path_name \
        --model_id   ${model_id_name}_${seq_len}_${pred_len} \
        --model      $model_name \
        --data       $data_name \
        --features   M \
        --seq_len    $seq_len \
        --pred_len   $pred_len \
        --enc_in     7 \
        --e_layers   3 \
        --n_heads    1 \
        --d_model    16 \
        --d_ff       128 \
        --dropout    0.3 \
        --fc_dropout 0.3 \
        --head_dropout 0 \
        --patch_len  16 \
        --stride     8 \
        --channel_attn 1 \
        --revin      1 \
        --affine     0 \
        --des        'CT_SAM' \
        --train_epochs $TRAIN_EPOCHS \
        --patience   $PATIENCE \
        --itr        1 \
        --batch_size 128 \
        --learning_rate 0.0001 \
        $SAM_FLAGS \
        2>&1 | tee logs/LongForecasting/${model_name}_CT_SAM_${model_id_name}_${seq_len}_${pred_len}.log

    echo ">>> Finished pred_len=$pred_len  at $(date)"

done

#===============================================================================
# SUMMARY
#===============================================================================
EXIT_CODE=$?
echo ""
echo "=========================================================================="
echo "JOB SUMMARY"
echo "=========================================================================="
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "Model:     CT-PatchTST (channel attention)"
echo "SAM:       $([ "$USE_SAM" -eq 1 ] && echo "rho=$RHO adaptive=$ADAPTIVE" || echo "DISABLED")"

if [ $EXIT_CODE -eq 0 ]; then
    echo "ALL 4 HORIZONS COMPLETED SUCCESSFULLY"
    echo ""
    echo "Results are appended to:  $WORK_DIR/result.txt"
    echo "Per-run logs:             $WORK_DIR/logs/LongForecasting/"
    echo "Checkpoints:              $WORK_DIR/checkpoints/"
    echo ""
    echo "Sanity-check attention rank:"
    echo "  python inspect_rank.py --checkpoint <ckpt_path> --data_path ETTh1.csv --enc_in 7 --n_heads 1"
else
    echo "TRAINING FAILED — check logs/ctpatchtst_sam_etth1_${SLURM_JOB_ID}.error"
fi
echo "=========================================================================="

exit $EXIT_CODE
