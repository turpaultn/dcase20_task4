#!/usr/bin/env bash
# Generate the audio files associated with jams files

# To be changed
# DATA
DIR_PATH=..
JAMS_PATH=${DIR_PATH}/dataset/audio/train/synthetic20
SOUNDBANK_PATH=${DIR_PATH}/synthetic/audio/train/soundbank
OUT_AUDIO=${JAMS_PATH}

SCRIPTS_PATH=../data_generation

python ${SCRIPTS_PATH}/generate_wav_from_jams --jams_folder=${JAMS_PATH} --soundbank=${SOUNDBANK_PATH} \
--out_audio_dir=${OUT_AUDIO} --save_isolated --save_jams