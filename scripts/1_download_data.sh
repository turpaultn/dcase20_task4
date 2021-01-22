#!/usr/bin/env bash
# This files download the soundbank, the RIR dat and the sound-separation model needed to create the data with further
# scripts

# We assume you're in the right conda envrionment

DATASET_DIR=../dataset

########### Shouldn't be changed after that (except if you want TUT background too)

SCRIPTS_PATH=../data_generation

#############
# DESED recorded data
#############

# If not already done
echo "download real data audio files ... ~23GB"
python ${SCRIPTS_PATH}/download_recorded_soundscapes.py --basedir=${DATASET_DIR}

#############
# DESED Soundbank
#############
echo "Download and extract soundbank"
python ${SCRIPTS_PATH}/download_synthetic_soundbank.py --basedir=../synthetic
echo "Done"