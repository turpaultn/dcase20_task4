#!/bin/bash

# Use the baseline source separation model to separate the mixtures from the different datasets
# Weak, unlabel_in_domain, validation and synthetic are separated.

BASE_DATASET=../dataset
AUDIO_PATH_WEAK=${BASE_DATASET}/audio/train/weak
AUDIO_PATH_UNLABEL=${BASE_DATASET}/audio/train/unlabel_in_domain
AUDIO_PATH_VALIDATION=${BASE_DATASET}/audio/validation
GENERATED_PATTERN_RECORDED=_ss/separated_sources  # The pattern to add after the audio path of where to put the generated data

# Change synthetic20 to synthetic20_reverb if you want to separate the reverbed data
AUDIO_PATH_SYNTH=${BASE_DATASET}/audio/train/synthetic20/soundscapes
GENERATED_SYNTH=${BASE_DATASET}/audio/train/synthetic20/separated_sources
SCRIPTS_PATH=../data_generation

# Download the checkpoint model from FUSS baseline
# If you want to use another model from Google, just change the path of the CHECKPOINT_MODEL and INFERENCE_META
wget -O FUSS_DESED_baseline_dry_2_model.tar.gz https://zenodo.org/record/3743844/files/FUSS_DESED_baseline_dry_2_model.tar.gz
tar -xzf FUSS_DESED_baseline_dry_2_model.tar.gz
rm FUSS_DESED_baseline_dry_2_model.tar.gz
mv fuss_desed_baseline_dry_2_model ../fuss_desed_baseline_dry_2_model
CHECKPOINT_MODEL=../fuss_desed_baseline_dry_2_model/fuss_desed_baseline_dry_2_model  # Path pointing to google checkoint
INFERENCE_META=../fuss_desed_baseline_dry_2_model/fuss_desed_baseline_dry_2_inference.meta  # Path pointing to google metagraph

######## Under this line you should not have to change anything ###########
# Recorded data
declare -a arr=(${AUDIO_PATH_WEAK} ${AUDIO_PATH_UNLABEL} ${AUDIO_PATH_VALIDATION})

for audio_path in "${arr[@]}"
do
   echo "${audio_path}"
   python ${SCRIPTS_PATH}/separate_wavs.py --audio_path=${audio_path} \
   --output_folder=${audio_path}${GENERATED_PATTERN_RECORDED} --checkpoint=${CHECKPOINT_MODEL} --inference=${INFERENCE_META}
   # or do whatever with individual element of the array
done

# Synthetic (generated) data
python ${SCRIPTS_PATH}/separate_wavs.py --audio_path=${AUDIO_PATH_SYNTH} \
--output_folder=${GENERATED_SYNTH} --checkpoint=${CHECKPOINT_MODEL} --inference=${INFERENCE_META}

