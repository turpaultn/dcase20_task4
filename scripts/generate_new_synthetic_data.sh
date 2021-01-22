#!/usr/bin/env bash
# Script to reproduce the synthetic data for dcase 2020 task 4 for sound event detection
# Change the parameters in each part of the if depending the case you are in

# We assume you're in the conda envrionment specified

# Change paths by yours
#DIR_PATH=../synthetic
DIR_PATH=..
SB_BASE=../synthetic
SUBSET=train  # The soundbank set to use (+ event_occurences values)
SOUNDBANK_PATH=${SB_BASE}/audio/${SUBSET}/soundbank
OUT_PATH=${DIR_PATH}/dataset/audio/${SUBSET}/synthetic20_${SUBSET}

# These parameters should reproduce the dataset
JSON_PATH=${DIR_PATH}/dataset/metadata/event_occurences/event_occurences_${SUBSET}.json
NUMBER=4500
NPROC=8         # Be careful, if you do not use the same number of processors, you won't reproduce the baseline data.
######## Under this line you should not have to change anything ###########
SCRIPTS_PATH=../data_generation

python ${SCRIPTS_PATH}/generate_synth_dcase20.py -sb ${SOUNDBANK_PATH} -o ${OUT_PATH} -jp ${JSON_PATH} \
-n=${NUMBER} --nproc=${NPROC}


SUBSET=validation  # The soundbank set to use (+ event_occurences values)
SOUNDBANK_PATH=${SB_BASE}/audio/${SUBSET}/soundbank
OUT_PATH=${DIR_PATH}/dataset/audio/${SUBSET}/synthetic20_${SUBSET}

# These parameters should reproduce the dataset
JSON_PATH=${DIR_PATH}/dataset/metadata/event_occurences/event_occurences_${SUBSET}.json
NUMBER=500
NPROC=2         # Be careful, if you do not use the same number of processors, you won't reproduce the baseline data.
######## Under this line you should not have to change anything ###########
SCRIPTS_PATH=../data_generation

python ${SCRIPTS_PATH}/generate_synth_dcase20.py -sb ${SOUNDBANK_PATH} -o ${OUT_PATH} -jp ${JSON_PATH} \
-n=${NUMBER} --nproc=${NPROC}