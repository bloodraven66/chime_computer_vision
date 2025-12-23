#!/bin/bash

set -e

echo "" 
echo "--------------------------------------------------" 
echo "" 
echo "Current time: $(date)"
start_time=$(date +%s)

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HOME="/mnt/matylda4/udupa/hugging-face"
export WANDB_MODE=offline

source /mnt/matylda4/udupa/miniconda3/etc/profile.d/conda.sh


conda activate mcorec

export CUDA_VISIBLE_DEVICES=$(/mnt/matylda4/udupa/exps/archive/ctc_spec_decoding/runs/sge_utils/free-gpus.sh 1) || {
   echo "Could not obtain GPU."
   exit 1
}
#export CUDA_VISIBLE_DEVICES=-1
export CUDA_HOME=/usr/local/share/cuda-12.1
export CUDA_LAUNCH_BLOCKING=1

#for cudnn error - do not use the version from /usr/local/share/cuda/lib64 for pytorch versions < 2.4 - https://github.com/pytorch/pytorch/issues/119989
unset LD_LIBRARY_PATH
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "hostname: $(hostname)"
echo ""

# /mnt/matylda4/udupa/exps/asr/madasr_exps/run.sh
# /mnt/matylda4/udupa/exps/asr/mta/asr1/run.sh

cd /mnt/matylda4/udupa/exps/asr/chime2026/mcorec_baseline


python3 script/train.py

echo "" 
echo "Job finished at: $(date)"
end_time=$(date +%s)
time_taken_minutes=$(echo "scale=2; ($end_time - $start_time) / 60" | bc)
echo "Time taken: $time_taken_minutes minutes"
echo "" 
echo "--------------------------------------------------" 
echo "--------------------------------------------------" 
