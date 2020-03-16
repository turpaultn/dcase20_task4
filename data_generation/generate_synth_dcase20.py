# -*- coding: utf-8 -*-
"""
File used to generate synthetic20
"""
import json
import logging
import multiprocessing
import warnings

import numpy as np
import time
import argparse
import os.path as osp
from pprint import pformat

from desed.generate_synthetic import SoundscapesGenerator
from desed.logger import create_logger
from desed.post_process import rm_high_polyphony, post_process_txt_labels
from desed.utils import create_folder
import config_data as cfg


def parse_args():
    absolute_dir_path = osp.abspath(osp.dirname(__file__))
    default_soundbank_path = osp.join(absolute_dir_path, "..", "..", "synthetic", "audio", "train", "soundbank")
    out_default_path = osp.join(absolute_dir_path, "..", "..", "dataset", "audio", "train", "synthetic20")
    default_json_path = osp.join(absolute_dir_path, "..", "..", "dataset", "metadata",
                                 "event_occurences", "event_occurences_train.json")

    parser = argparse.ArgumentParser()
    # generating files
    parser.add_argument('-sb', '--soundbank', type=str, default=default_soundbank_path, required=True,
                        help="path to the soundbank folder (should contain foreground and background subfolders)")
    parser.add_argument('-o', '--out_folder', type=str, default=out_default_path, required=True,
                        help="the output folder in which to save the generated examples")
    parser.add_argument('-jp', '--json_path', type=str, default=default_json_path, required=True,
                        help=f"path to the json path with co-occurences, see {default_json_path} for an example of"
                        f"parameters defined (the structure has to be similar)")
    parser.add_argument('-n', '--number', type=int, default=1000, help="the number of data to be generated")
    parser.add_argument('--out_tsv', type=str, default=None,
                        help="output metadata file, if not defined the folder will be taken from --out_folder and "
                             "audio replaced to metadata in the path")
    parser.add_argument('--nproc', type=int, default=8,
                        help="the number of parallel processes to be used if parallelizing")
    parser.add_argument('--random_seed', type=int, default=2020)

    args = parser.parse_args()
    pformat(vars(args))
    return args


if __name__ == '__main__':
    LOG = create_logger(__name__, terminal_level=logging.INFO)
    LOG.info(__file__)
    t = time.time()
    args = parse_args()

    # General parameters
    out_folder = args.out_folder
    soundbank_path = args.soundbank
    n_soundscapes = args.number
    random_state = args.random_seed

    subset = "soundscapes"  # Needed for RIR, so used everywhere (need to be the same in reverberate_data.py)
    full_out_folder = osp.join(out_folder, subset)
    out_tsv = args.out_tsv
    if out_tsv is None:
        out_tsv = full_out_folder.replace("audio", "metadata") + ".tsv"
    create_folder(full_out_folder)
    create_folder(osp.dirname(out_tsv))

    # ############
    # Generate soundscapes
    # ############
    # Parameters (default)
    clip_duration = cfg.clip_duration
    sample_rate = cfg.sample_rate
    ref_db = cfg.ref_db
    pitch_shift = cfg.pitch_shift

    # Defined
    fg_folder = osp.join(soundbank_path, "foreground")
    bg_folder = osp.join(soundbank_path, "background")
    with open(args.json_path) as json_file:
        co_occur_dict = json.load(json_file)

    # Check for multiprocessing
    nproc = args.nproc
    if nproc > 1:
        if n_soundscapes // nproc < 200:
            nproc = n_soundscapes // 200
            if nproc < 2:
                nproc = 1
                warnings.warn(f"Not enough files to generate (minimum 200 per processor, "
                              f"having {n_soundscapes // nproc}), using only one processor")
            else:
                warnings.warn(f"Be careful, less than 200 files generated per jobs can have an impact "
                              f"on the distribution of the classes, changing nproc to {nproc}")

    # Generate the data in single or multi processing
    if nproc > 1:
        rest = n_soundscapes % nproc
        multiple = n_soundscapes - rest  # Get a number we can divide by n_jobs
        list_start = np.arange(0, multiple, multiple // nproc)  # starting numbers of each process
        numbers = [multiple // nproc for i in range(len(list_start) - 1)]  # numbers per processes except last one
        numbers.append(multiple // nproc + rest)  # Last process having the rest added

        multiprocessing.log_to_stderr()
        logger = multiprocessing.get_logger()
        logger.setLevel(logging.INFO)


        def generate_multiproc(start_from, number, rand_state):
            # We need to give a different random state to each object otherwise we just replicate X times the same data
            # except if it is None
            sg = SoundscapesGenerator(duration=clip_duration,
                                      fg_folder=fg_folder,
                                      bg_folder=bg_folder,
                                      ref_db=ref_db,
                                      samplerate=sample_rate,
                                      random_state=rand_state,
                                      logger=logger)
            sg.generate_by_label_occurence(label_occurences=co_occur_dict,
                                           number=number,
                                           out_folder=full_out_folder,
                                           save_isolated_events=True,
                                           start_from=start_from,
                                           pitch_shift=pitch_shift)
        if random_state is None:
            random_states = [None for i in range(nproc)]
        else:
            random_states = [random_state + i for i in range(nproc)]
        print(random_states)
        with multiprocessing.Pool(nproc) as p:
            p.starmap(generate_multiproc, zip(list_start, numbers, random_states))
    # Single process
    else:
        sg = SoundscapesGenerator(duration=clip_duration,
                                  fg_folder=fg_folder,
                                  bg_folder=bg_folder,
                                  ref_db=ref_db,
                                  samplerate=sample_rate,
                                  random_state=random_state)
        sg.generate_by_label_occurence(label_occurences=co_occur_dict,
                                       number=n_soundscapes,
                                       out_folder=full_out_folder,
                                       save_isolated_events=True,
                                       pitch_shift=pitch_shift)

    # ##
    # Post processing
    rm_high_polyphony(full_out_folder, max_polyphony=2)
    # concat same labels overlapping
    post_process_txt_labels(full_out_folder,
                            output_folder=full_out_folder,
                            output_tsv=out_tsv, rm_nOn_nOff=True)

