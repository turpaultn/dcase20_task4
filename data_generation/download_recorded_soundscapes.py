# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import argparse
import os
import pandas as pd
from pprint import pformat
import time
from desed.logger import create_logger
from desed.download import download_audioset_files


LOG = create_logger("DESED_real")


def download_from_csv(csv_path, result_dir, missing_files_folder):
    LOG.info(f"downloading data from: {csv_path}")
    # read metadata file and get only one filename once
    df = pd.read_csv(csv_path, header=0, sep='\t')
    filenames_test = df["filename"].drop_duplicates()
    missing_tsv = os.path.join(missing_files_folder, "missing_files_" + os.path.basename(csv_path))
    download_audioset_files(filenames_test, result_dir, n_jobs=N_JOBS, chunk_size=CHUNK_SIZE,
                            missing_files_tsv=missing_tsv)
    LOG.info("###### DONE #######")


if __name__ == "__main__":
    # To be changed for your root folder if needed (if dcase2019 used)
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str, default="../dataset",
                        help="the base folder of the dataset (will create subset automatically)")
    parser.add_argument('--n_jobs', type=int, default=3,
                        help="The number of parallel jobs to download the data")
    parser.add_argument('--chunk_size', type=int, default=10,
                        help="The number of chunks given to each parallel process at a time")
    args = parser.parse_args()
    pformat(vars(args))

    base_missing_files_folder = args.basedir
    dataset_folder = args.basedir

    LOG.info("Download_data")
    LOG.info("\n\nOnce database is downloaded, do not forget to check your missing_files\n\n")

    LOG.info("You can change N_JOBS and CHUNK_SIZE to increase the download with more processes.")
    # Modify it with the number of process you want, but be careful, youtube can block you if you put too many.
    N_JOBS = 3

    # Only useful when multiprocessing,
    # if chunk_size is high, download is faster. Be careful, progress bar only update after each chunk.
    CHUNK_SIZE = 10

    download_from_csv(
        os.path.join(dataset_folder, "metadata", "validation", "validation.tsv"),
        os.path.join(dataset_folder, "audio", "validation"),
        base_missing_files_folder
    )

    download_from_csv(
        os.path.join(dataset_folder, "metadata", "train", "weak.tsv"),
        os.path.join(dataset_folder, "audio", "train", "weak"),
        base_missing_files_folder
    )

    download_from_csv(
        os.path.join(dataset_folder, "metadata", "train", "unlabel_in_domain.tsv"),
        os.path.join(dataset_folder, "audio", "train", "unlabel_in_domain"),
        base_missing_files_folder
    )

    LOG.info(f"time of the program: {time.time() - t}")
