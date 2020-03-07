#!/usr/bin/env bash

BASE_PATH=..
cd ${BASE_PATH}

wget -O FUSS_baseline_model.tar.gz https://zenodo.org/record/3694384/files/FUSS_baseline_model.tar.gz?download=1
tar -xzf FUSS_baseline_model.tar.gz

wget -O $FUSS_rir_data.tar.gz https://zenodo.org/record/3694384/files/FUSS_rir_data.tar.gz?download=1
tar -xzf FUSS_rir_data.tar.gz
