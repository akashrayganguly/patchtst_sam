#!/bin/bash
#===============================================================================
# SLURM SCRIPT FOR PatchTST + SAM SUPERVISED TRAINING ON ETTh1
#
# PatchTST is a single-process training job — no FSDP, no torchrun.
# We run 4 sequential prediction horizons (96, 192, 336, 720) on one GPU.
#
# SAM doubles the forward-backward passes per iteration, so expect roughly
# 2× wall-clock time per epoch compared to the baseline.
#
# USAGE:
#   sbatch run_patchtst_sam_etth1.slurm              # default rho=0.2
#   RHO=0.5 sbatch run_patchtst_sam_etth1.slurm      # override rho
#   USE_SAM=0 sbatch run_patchtst_sam_etth1.slurm    # disable SAM (baseline)
#===============================================================================

#SBATCH --job-name=patchtst-sam-etth1
#SBATCH --mail-user=shivansh@prl.res.in
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpulong
#SBATCH --gres=gpu:1                      # 1 GPU is sufficient — PatchTST on ETTh1
                                          # is lightweight (7 channels, small model)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                 # A few workers for DataLoader
#SBATCH --mem=32G
#SBATCH --time=04-00:00:00                # SAM ~2× slower: allow 4 days
#SBATCH --output=logs/patchtst_sam_etth1_%j.log
#SBATCH --error=logs/patchtst_sam_etth1_%j.error

#===============================================================================
# HEADER
#===============================================================================
echo "=========================================================================="
echo "    PatchTST + SAM SUPERVISED TRAINING — ETTh1"
echo "=========================================================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start time: $(date)"
echo "=========================================================================="

#===============================================================================
# SETUP
#===============================================================================
mkdir -p logs/LongForecasting

# ---- Edit this path to wherever you cloned yuqinie98/PatchTST ---------------
WORK_DIR="/home/akashg/time-series/python-files/PatchTST/PatchTST_supervised"
# -----------------------------------------------------------------------------

cd "$WORK_DIR" || { echo "ERROR: Cannot cd to $WORK_DIR"; exit 1; }
echo "Working directory: $(pwd)"

# ---- Conda environment -------------------------------------------------------
source /home/akashg/miniconda3/etc/profile.d/conda.sh
conda activate informerworking          # reuse your existing environment; it
                                        # already has PyTorch. If deps are missing
                                        # run:  pip install -r requirements.txt

echo "Python:  $(which python)"
echo "Version: $(python --version 2>&1)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs:    $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================================================="

# ---- Data --------------------------------------------------------------------
if [ ! -f "dataset/ETTh1.csv" ]; then
    echo "ERROR: dataset/ETTh1.csv not found."
    echo "Download it from the Autoformer Google Drive link in the PatchTST README"
    echo "and place it at: $WORK_DIR/dataset/ETTh1.csv"
    exit 1
fi
echo "Data file found: dataset/ETTh1.csv"
echo "=========================================================================="

#===============================================================================
# SAM CONFIGURATION
#
# Override via environment:  RHO=0.3 sbatch ...
#                            USE_SAM=0 sbatch ...   (baseline run)
#                            ADAPTIVE=1 sbatch ...  (ASAM variant)
#===============================================================================
USE_SAM="${USE_SAM:-1}"        # 1 = SAM enabled (default), 0 = baseline
RHO="${RHO:-0.2}"              # Perturbation radius (start low for deep PatchTST)
ADAPTIVE="${ADAPTIVE:-0}"      # 0 = SAM, 1 = ASAM (adaptive)

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
    echo "SAM:          DISABLED (baseline run)"
fi

#===============================================================================
# HYPERPARAMETERS  (directly from the official etth1.sh in the repo)
#
# Model: PatchTST/42  (seq_len=336 → N = (336-16)/8 + 2 = 42 patches)
# These match the paper's Table 3 ETTh1 numbers.
#
# Small model size (H=4, D=16, F=128) is intentional — the paper uses a reduced
# model for ETTh1/ETTh2 to avoid overfitting on this smaller dataset.
#===============================================================================
model_name=PatchTST
seq_len=336
random_seed=2021

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

# SAM converges slower per epoch — allow more epochs and patience
if [ "$USE_SAM" -eq 1 ]; then
    TRAIN_EPOCHS=150
    PATIENCE=20
else
    TRAIN_EPOCHS=100
    PATIENCE=100    # original default (effectively no early stop with 100 epochs)
fi

echo "Model:        $model_name"
echo "Look-back:    seq_len=$seq_len"
echo "Patch length: 16, stride 8  → ~42 patches"
echo "Enc layers:   3,  heads: 4,  d_model: 16,  d_ff: 128"
echo "Horizons:     96 192 336 720"
echo "Epochs:       $TRAIN_EPOCHS,  patience: $PATIENCE,  batch: 128,  lr: 0.0001"
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
        --n_heads    4 \
        --d_model    16 \
        --d_ff       128 \
        --dropout    0.3 \
        --fc_dropout 0.3 \
        --head_dropout 0 \
        --patch_len  16 \
        --stride     8 \
        --des        'Exp' \
        --train_epochs $TRAIN_EPOCHS \
        --patience   $PATIENCE \
        --itr        1 \
        --batch_size 128 \
        --learning_rate 0.0001 \
        $SAM_FLAGS \
        2>&1 | tee logs/LongForecasting/${model_name}_SAM_${model_id_name}_${seq_len}_${pred_len}.log

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
echo "SAM:       $([ "$USE_SAM" -eq 1 ] && echo "rho=$RHO adaptive=$ADAPTIVE" || echo "DISABLED")"

if [ $EXIT_CODE -eq 0 ]; then
    echo "ALL 4 HORIZONS COMPLETED SUCCESSFULLY"
    echo ""
    echo "Results are appended to:  $WORK_DIR/result.txt"
    echo "Per-run logs:             $WORK_DIR/logs/LongForecasting/"
    echo "Checkpoints:              $WORK_DIR/checkpoints/"
else
    echo "TRAINING FAILED — check logs/patchtst_sam_etth1_${SLURM_JOB_ID}.error"
fi
echo "=========================================================================="

exit $EXIT_CODE
