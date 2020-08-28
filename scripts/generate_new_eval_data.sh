#!/usr/bin/env bash
# Script to reproduce the synthetic data for dcase 2020 task 4 for sound event detection
# Change the parameters in each part of the if depending the case you are in

# We assume you're in the conda envrionment specified

# Change paths by yours
DIR_PATH=..
SB_BASE=../../../walle/soundbank

# Normal set
SUBSET=eval  # The soundbank set to use (+ event_occurences values)
SOUNDBANK_PATH=${SB_BASE}/audio/${SUBSET}/soundbank
OUT_PATH=${DIR_PATH}/dataset/audio/${SUBSET}/synthetic20_${SUBSET}

# These parameters should reproduce the dataset
JSON_PATH=${DIR_PATH}/dataset/metadata/event_occurences/event_occurences_${SUBSET}.json
NUMBER=1500
NPROC=8         # Be careful, if you do not use the same number of processors, you won't reproduce the baseline data.
######## Under this line you should not have to change anything ###########
SCRIPTS_PATH=../data_generation

python ${SCRIPTS_PATH}/generate_synth_dcase20.py -sb ${SOUNDBANK_PATH} -o ${OUT_PATH} -jp ${JSON_PATH} \
-n=${NUMBER} --nproc=${NPROC} -sr 44100


BASE_DATASET=${DIR_PATH}/dataset
subset=eval
## One event per file (study of weak labels)
NUMBER=1000
python ${SCRIPTS_PATH}/generate_one_event.py -n ${NUMBER} \
-sb ${SOUNDBANK_PATH} \
-o ${BASE_DATASET}/audio/${subset}/one_event_${subset} \
-s ${BASE_DATASET}/metadata/${subset}/one_event_${subset}.tsv \
-sr 44100 -j 5

# One event, varying onset
NUMBER=1000
python ${SCRIPTS_PATH}/generate_var_onset.py -n ${NUMBER} \
-o ${BASE_DATASET}/audio/${subset}/var_onset_${subset} \
-sf ${BASE_DATASET}/metadata/${subset}/var_onset_${subset} \
-fg ${SOUNDBANK_PATH}/foreground_on_off \
-bg ${SOUNDBANK_PATH}/background \
-sr 44100 -j 5

## Long events
OUT_PATH=${DIR_PATH}/dataset/audio/${SUBSET}/synthetic20_${SUBSET}_long
NUMBER=150
DURATION=300  # 5 min
MAX_EVENTS=15
NPROC=3

python ${SCRIPTS_PATH}/generate_synth_dcase20.py -d ${DURATION} -m ${MAX_EVENTS} \
-sb ${SOUNDBANK_PATH} \
-o ${OUT_PATH} -jp ${JSON_PATH} \
-n=${NUMBER} --nproc=${NPROC} -sr 44100