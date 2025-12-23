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

cd /mnt/matylda4/udupa/exps/asr/chime2026/mcorec_baseline

model_type=avsr_cocktail
session_dir="data-bin/dev/session_57"

###pretrained baseline
# checkpoint_path=./model-bin/avsr_cocktail
# output_dir_name="output_avsr_cocktail"
# Speaker to WER: {'spk_0': 0.3867, 'spk_1': 0.7906, 'spk_2': 0.4286, 'spk_3': 0.5553, 'spk_4': 0.5239}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.19335, 'spk_1': 0.3953, 'spk_2': 0.2143, 'spk_3': 0.27765, 'spk_4': 0.26195}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 0.53702
# Average Joint ASR-Clustering Error Rate: 0.26851

##audio-visual finetune
checkpoint_path=./model-bin/mcorec_finetuning/checkpoint-6500/
# output_dir_name="output_avsr_cocktail_finetune_6500"
# Speaker to WER: {'spk_0': 0.4183, 'spk_1': 0.4729, 'spk_2': 0.5382, 'spk_3': 0.6004, 'spk_4': 0.5522}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.20915, 'spk_1': 0.23645, 'spk_2': 0.2691, 'spk_3': 0.3002, 'spk_4': 0.2761}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 0.5164
# Average Joint ASR-Clustering Error Rate: 0.2582

##template video finetune
# otput_dir_name="output_avsr_cocktail_finetune_6500-infer-templatevideo"
# Average Speaker WER: 1.0

##zero video finetune
# otput_dir_name="output_avsr_cocktail_finetune_6500-infer-zerovideo"
# Average Speaker WER: 1.0

##zero audio finetune
otput_dir_name="output_avsr_cocktail_finetune_6500-infer-zeroaudio"



##all augs
# checkpoint_path=./model-bin/mcorec_finetuning_vidaug_{rc_hf_cj_rg}/checkpoint-2000
# output_dir_name="output_avsr_cocktail_finetune_2000_allaugs"
# Conversation clustering F1 score: 1.0
# Speaker to WER: {'spk_0': 0.3617, 'spk_1': 0.3755, 'spk_2': 0.5748, 'spk_3': 0.5922, 'spk_4': 0.5522}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.18085, 'spk_1': 0.18775, 'spk_2': 0.2874, 'spk_3': 0.2961, 'spk_4': 0.2761}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 0.49128
# Average Joint ASR-Clustering Error Rate: 0.24564

# checkpoint_path=./model-bin/mcorec_finetuning_vidaug_{rc_rg}/checkpoint-2900
# output_dir_name="output_avsr_cocktail_finetune_2000_gray"
# Conversation clustering F1 score: 1.0
# Speaker to WER: {'spk_0': 0.3767, 'spk_1': 0.4404, 'spk_2': 0.5947, 'spk_3': 0.584, 'spk_4': 0.5478}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.18835, 'spk_1': 0.2202, 'spk_2': 0.29735, 'spk_3': 0.292, 'spk_4': 0.2739}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 0.5087200000000001
# Average Joint ASR-Clustering Error Rate: 0.25436000000000003


# checkpoint_path=./model-bin/mcorec_finetuning_vidaug_{rc_hf}/checkpoint-1000
# output_dir_name="output_avsr_cocktail_finetune_1000_hflip"
# Conversation clustering F1 score: 1.0
# Speaker to WER: {'spk_0': 0.475, 'spk_1': 0.6715, 'spk_2': 0.8405, 'spk_3': 0.8074, 'spk_4': 0.6717}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.2375, 'spk_1': 0.33575, 'spk_2': 0.42025, 'spk_3': 0.4037, 'spk_4': 0.33585}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 0.69322
# Average Joint ASR-Clustering Error Rate: 0.34661


# checkpoint_path=./model-bin/mcorec_finetuning_vidaug_{rc_cj}/checkpoint-3000
# output_dir_name="output_avsr_cocktail_finetune_3000_cj"
# Conversation clustering F1 score: 1.0
# Speaker to WER: {'spk_0': 0.36, 'spk_1': 0.3682, 'spk_2': 0.5648, 'spk_3': 0.5779, 'spk_4': 0.5304}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.18, 'spk_1': 0.1841, 'spk_2': 0.2824, 'spk_3': 0.28895, 'spk_4': 0.2652}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 0.48026
# Average Joint ASR-Clustering Error Rate: 0.24013

##----zero audio----

# checkpoint_path=./model-bin/mcorec_finetuning_zeroaudio/checkpoint-7900
# output_dir_name="output_avsr_cocktail_finetune_7900_zeroaudio"
## default default
# Speaker to WER: {'spk_0': 0.5983, 'spk_1': 0.6245, 'spk_2': 0.7542, 'spk_3': 0.7807, 'spk_4': 0.6913}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.29915, 'spk_1': 0.31225, 'spk_2': 0.3771, 'spk_3': 0.39035, 'spk_4': 0.34565}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 0.6898
# Average Joint ASR-Clustering Error Rate: 0.3449

##template video
# output_dir_name="output_avsr_cocktail_finetune_7900_zeroaudio_infer-templatevideo"
# Average Speaker WER: 0.96212

##zero video
# output_dir_name="output_avsr_cocktail_finetune_7900_zeroaudio_infer-zerovideo"
# Average Speaker WER: 0.9995

##zero audio
# output_dir_name="output_avsr_cocktail_finetune_7900_zeroaudio_infer-zeroaudio"
# Average Speaker WER: 0.7456



##----zero video----

# checkpoint_path=./model-bin/mcorec_finetuning_zerovideo/checkpoint-8600
# output_dir_name="output_avsr_cocktail_finetune_8600_zerovideo"
## default default
# Conversation clustering F1 score: 1.0
# Speaker to WER: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Speaker clustering F1 score: {'spk_0': 1.0, 'spk_1': 1.0, 'spk_2': 1.0, 'spk_3': 1.0, 'spk_4': 1.0}
# Joint ASR-Clustering Error Rate: {'spk_0': 0.5, 'spk_1': 0.5, 'spk_2': 0.5, 'spk_3': 0.5, 'spk_4': 0.5}
# Average Conversation Clustering F1 score: 1.0
# Average Speaker WER: 1.0
# Average Joint ASR-Clustering Error Rate: 0.5

##template video
# output_dir_name="output_avsr_cocktail_finetune_8600_zerovideo_infer-templatevideo"
# Average Speaker WER: 1.0

##zero video
# output_dir_name="output_avsr_cocktail_finetune_8600_zerovideo_infer-zerovideo"
# Average Speaker WER: 0.8505

##zero audio
# output_dir_name="output_avsr_cocktail_finetune_8600_zerovideo_infer-zeroaudio"
#1.0
##----template video----

# checkpoint_path=./model-bin/mcorec_finetuning_templatevideo/checkpoint-30000
# output_dir_name="output_avsr_cocktail_finetune_30000_templatevideo_infer-templatevideo"
##default
# Average Speaker WER: 0.9632800000000001
##zero audio
# Average Speaker WER: 1.0
##zero video
# Average Speaker WER: 0.9958600000000001
##template video
# Average Speaker WER: 0.96692



args=(
  --model_type "$model_type"
  --session_dir "$session_dir"
  --verbose
  --checkpoint_path "$checkpoint_path"
  --output_dir_name "$output_dir_name"
)

echo "Running with args: ${args[*]}"
python script/inference.py "${args[@]}"
# 
python script/evaluate.py --session_dir "$session_dir" --output_dir_name "$output_dir_name"     


echo "" 
echo "Job finished at: $(date)"
end_time=$(date +%s)
time_taken_minutes=$(echo "scale=2; ($end_time - $start_time) / 60" | bc)
echo "Time taken: $time_taken_minutes minutes"
echo "" 
echo "--------------------------------------------------" 
echo "--------------------------------------------------" 
