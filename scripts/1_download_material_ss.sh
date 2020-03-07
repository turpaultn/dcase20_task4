#!/usr/bin/env bash
# This files download the soundbank, the RIR dat and the sound-separation model needed to create the data with further
# scripts

# We assume you're in the right conda envrionment

BASE_PATH=..
cd ${BASE_PATH}

########### Shouldn't be changed after that (except if you want TUT background too)

#############
# DESED Soundbank
#############
# If not already installed, install DESED
pip install desed@git+https://github.com/turpaultn/DESED
mkdir synthetic
cd synthetic
echo "Download and extract soundbank"
wget -O DESED_synth_soundbank.tar.gz https://zenodo.org/record/3571305/files/DESED_synth_soundbank.tar.gz?download=1
tar -xzf DESED_synth_soundbank.tar.gz
#rm DESED_synth_soundbank.tar.gz
echo "Done"

cd data_generation
# If you did not download the synthetic training background yet
echo "Download SINS background... (to add TUT, add the option --TUT)"
python get_background_training.py --basedir=../synthetic
cd ..
echo "Done"



############
# Source separation
#########
wget -O FUSS_baseline_model.tar.gz https://zenodo.org/record/3694384/files/FUSS_baseline_model.tar.gz?download=1
tar -xzf FUSS_baseline_model.tar.gz
#rm FUSS_baseline_model.tar.gz

wget -O $FUSS_rir_data.tar.gz https://zenodo.org/record/3694384/files/FUSS_rir_data.tar.gz?download=1
tar -xzf FUSS_rir_data.tar.gz
#rm FUSS_rir_data.tar.gz
