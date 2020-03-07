#!/bin/bash

# Use the baseline source separation model to separate the mixtures from the different datasets
# Weak, unlabel_in_domain, validation and synthetic are separated.

BASE_DATASET=../dataset
AUDIO_PATH_WEAK=${BASE_DATASET}/audio/train/weak
AUDIO_PATH_UNLABEL=${BASE_DATASET}/audio/train/unlabel_in_domain
AUDIO_PATH_VALIDATION=${BASE_DATASET}/audio/validation
GENERATED_PATTERN_RECORDED=_ss/ss_computed  # The pattern to add after the audio path of where to put the generated data

AUDIO_PATH_SYNTH=${BASE_DATASET}/audio/train/synthetic20_reverb/soundscapes
GENERATED_SYNTH=${BASE_DATASET}/audio/train/synthetic20_reverb/ss_computed

MODEL_DIR=../baseline_model  # Path pointing to google folder model
SCRIPTS_PATH=../data_generation

######## Under this line you should not have to change anything ###########

# Recorded data
declare -a arr=(${AUDIO_PATH_WEAK} ${AUDIO_PATH_UNLABEL} ${AUDIO_PATH_VALIDATION})

for audio_path in "${arr[@]}"
do
   echo "${audio_path}"
   python ${SCRIPTS_PATH}/separate_wavs.py --audio_path=${audio_path} \
   --output_folder=${audio_path}${GENERATED_PATTERN_RECORDED} --model_dir=${MODEL_DIR}
   # or do whatever with individual element of the array
done

# Synthetic (generated) data
python ${SCRIPTS_PATH}/separate_wavs.py --audio_path=${AUDIO_PATH_SYNTH} \
--output_folder=${GENERATED_SYNTH} --model_dir=${MODEL_DIR}

