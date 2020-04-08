#!/bin/bash
conda create -y -n dcase2020 python=3.6
source activate dcase2020
conda install -y pandas h5py scipy
conda install -y pytorch torchvision cudatoolkit=9.1 -c pytorch # for gpu install (or cpu in MAC)
# conda install pytorch-cpu torchvision-cpu -c pytorch (cpu linux)
conda install -y pysoundfile librosa youtube-dl tqdm -c conda-forge
conda install -y ffmpeg -c conda-forge

pip install dcase_util
pip install sed-eval
pip install --upgrade psds_eval
pip install scaper

# Should be done only if you did not already installed it to download the data
pip install --upgrade desed@git+https://github.com/turpaultn/DESED

# Source separation:
pip install tensorflow