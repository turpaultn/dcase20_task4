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
wget -O DESED_synth_soundbank.tar.gz https://zenodo.org/record/3702397/files/DESED_synth_soundbank.tar.gz?download=1
tar -xzf DESED_synth_soundbank.tar.gz
#rm DESED_synth_soundbank.tar.gz
mv synthetic ../synthetic
echo "Done"

# If you did not download the synthetic training background yet
echo "Download SINS background... (to add TUT, add the option --TUT)"
python ${SCRIPTS_PATH}/get_background_training.py --basedir=../synthetic
echo "Done"



############
# Source separation
#########
cd ..
wget -O FUSS_baseline_model.tar.gz https://zenodo.org/record/3694384/files/FUSS_baseline_model.tar.gz?download=1
tar -xzf FUSS_baseline_model.tar.gz
#rm FUSS_baseline_model.tar.gz

wget -O FUSS_rir_data.tar.gz https://zenodo.org/record/3694384/files/FUSS_rir_data.tar.gz?download=1
tar -xzf FUSS_rir_data.tar.gz
#rm FUSS_rir_data.tar.gz
