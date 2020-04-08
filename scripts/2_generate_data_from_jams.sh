#!/usr/bin/env bash
# Generate the audio files associated with jams files

# Can be changed
# DATA
DIR_PATH=..
DIR_JAMS_DATASET=${DIR_PATH}/dataset
JAMS_PATH=${DIR_JAMS_DATASET}/audio/train/synthetic20
SOUNDBANK_PATH=${DIR_PATH}/synthetic/audio/train/soundbank
OUT_AUDIO=${JAMS_PATH}/soundscapes

SCRIPTS_PATH=../data_generation


############# Shouldn't be changed after this line
WORKDIR=$(pwd -P)

cd ${DIR_JAMS_DATASET}
echo "Getting jams for dcase 2020"
# Get jams file
wget -O DESED_synth_dcase20jams.tar.gz https://zenodo.org/record/3713328/files/DESED_synth_dcase20_train_jams.tar.gz?download=1
tar -xzf DESED_synth_dcase20jams.tar.gz
rm DESED_synth_dcase20jams.tar.gz
cd ${WORKDIR}
echo "Done"

echo "Generating wav from jams ..."
python ${SCRIPTS_PATH}/generate_wav_from_jams.py --jams_folder=${JAMS_PATH} --soundbank=${SOUNDBANK_PATH} \
--out_audio_dir=${OUT_AUDIO} --save_isolated --save_jams