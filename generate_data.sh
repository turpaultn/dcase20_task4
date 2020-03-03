#!/bin/bash
#SBATCH -p p100
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -J nturpaul
#SBATCH -t 10:00:00
#SBATCH --output=res.txt
TRYING_EXT=_try

source activate ssenv
# Script to reproduce the synthetic data for dcase 2020 task 4 for sound event detection
# Change the parameters in each part of the if depending the case you are in
# To launch this code you need to install Tensorflow.

REPRODUCE=$1 # If a parameter is given after the bash script, will try to reproduce an existing dataset
NPROC=8
SCRIPTS_PATH=baseline/data_generation
# ####################
# Launching new experiments (should reproduce the dataset), change the parameters by yours
# ##################
if [[ -z ${REPRODUCE} ]]; then
	DIR_PATH=.
	# Generate synthetic data from soundbank (saving isolated events)
	SUBSET=train
	SOUNDBANK_PATH=${DIR_PATH}/synthetic/audio/${SUBSET}/soundbank
	OUT_PATH=${DIR_PATH}/dataset/audio/${SUBSET}/synthetic20${TRYING_EXT}
	JSON_PATH=${DIR_PATH}/dataset/metadata/event_occurences/event_occurences_${SUBSET}.json
	NUMBER=5000

	# Reverb default path
	RIR=rir_data  # Could be replaced by wget and then the path
	REVERB_PATH=${OUT_PATH}_reverb


	######## Under this line you should not have to change anything ###########

	# Reverberate data using same RIR as Google baseline
	MIX_INFO=${REVERB_PATH}/mix_info.txt
	SRC_LIST=${REVERB_PATH}/src_list.txt
	RIR_LIST=${REVERB_PATH}/rir_list.txt
	python ${SCRIPTS_PATH}/generate_synth_dcase20.py -sb ${SOUNDBANK_PATH} -o ${OUT_PATH} -jp ${JSON_PATH} \
	-n=${NUMBER} --nproc=${NPROC}

	python ${SCRIPTS_PATH}/reverberate_data.py --rir_folder=${RIR} --input_folder=${OUT_PATH} --reverb_out_folder=${REVERB_PATH} \
	--rir_subset=${SUBSET} --mix_info_file=${MIX_INFO}  --src_list_file=${SRC_LIST} --rir_list_file=${RIR_LIST} \
	--nproc=${NPROC}

# ###################
# Launching existing config, parameters to change
# ###################
else
	# To be changed
	# DATA
	DIR_PATH=.
	JAMS_PATH=${DIR_PATH}/dataset/audio/train/synthetic20${TRYING_EXT}
	SOUNDBANK_PATH=${DIR_PATH}/synthetic/audio/train/soundbank/soundscapes
	OUT_AUDIO=${JAMS_PATH}
	# REVERB
	RIR=rir_data  # Could be replaced by wget and then the path
	REVERB_PATH=${JAMS_PATH}_reverb
	MIX_INFO=${REVERB_PATH}/mix_info.txt
	SRC_LIST=${REVERB_PATH}/src_list.txt
	RIR_LIST=${REVERB_PATH}/rir_list.txt

	python ${SCRIPTS_PATH}/generate_wav_from_jams --jams_folder=${JAMS_PATH} --soundbank=${SOUNDBANK_PATH} --out_audio_dir=${OUT_AUDIO}\
	--save_isolated --save_jams

	python ${SCRIPTS_PATH}/reverberate_data.py --rir_folder=${RIR} --input_folder=${OUT_AUDIO} --reverb_out_folder=${REVERB_PATH} \
	--rir_subset=${SUBSET} --mix_info_file=${MIX_INFO}  --src_list_file=${SRC_LIST} --rir_list_file=${RIR_LIST} \
	--nproc=${NPROC}
fi

AUDIO_PATH=${REVERB_PATH}/soundscapes/
GENERATED=${REVERB_PATH}/ss_computed
MODEL_DIR=model_ckpt
python separate_wavs.py --audio_path=${AUDIO_PATH} --output_folder=${GENERATED} --model_dir=${MODEL_DIR}