# -*- coding: utf-8 -*-
"""
File used to generate synthetic20
"""
import functools
import glob
import json
import logging
import multiprocessing
import os
import re
import sys
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

import config as cfg


absolute_dir_path = osp.abspath(osp.dirname(__file__))

base_dir_repo = os.path.abspath(os.path.join("..", ".."))
sys.path.append(os.path.join(base_dir_repo, "dcase2020_inference_source_separation", "dcase2020_review"))
print(sys.path)

from reverberate_and_mix import reverberate_and_mix, make_rir_dict_from_folder, make_mix_info_subsources
from utils import make_example_dict_from_folder, check_and_correct_example


def make_example_list(base_dir_soundscapes, base_dir_isolated_events=None, pattern_sources="_events"):
    """Make a list of files as needed per the code in sound-separation/dcase2020
    Args:
        base_dir_soundscapes: str, the path of the base directory where to find the mixtures
        base_dir_isolated_events: str, the path where to find subfolders (associated with mixtures)
        pattern_sources: str, the pattern to match a mixture and isolated events folders
            (pattern is what is added to the mixture filename)
    Returns:

    """
    if base_dir_isolated_events is None:
        base_dir_isolated_events = base_dir_soundscapes
    examples_list = []
    list_soundscapes = glob.glob(osp.join(base_dir_soundscapes, "*.wav"))
    for sc_fname in list_soundscapes:
        bname = osp.basename(sc_fname)
        example_list = [bname]

        bname_no_ext, ext = osp.splitext(bname)
        list_wav_events = glob.glob(osp.join(base_dir_isolated_events, bname_no_ext, pattern_sources))
        for ev_fname in list_wav_events:
            example_list.append(ev_fname)
        examples_list.append("\t".join(example_list))

    return examples_list


if __name__ == '__main__':
    LOG = create_logger(__name__, terminal_level=logging.INFO)
    LOG.info(__file__)
    t = time.time()
    default_soundbank_path = osp.join(absolute_dir_path, "..", "..", "synthetic", "audio", "train", "soundbank")
    # out_default_path = osp.join(absolute_dir_path, "..", "..", "dataset", "audio", "train", "synthetic20")
    out_default_path = osp.join("/Volumes/TOSHIBA EXT/All/dcase20", "dataset", "audio", "train", "synthetic20")
    out_default_reverb_path = osp.join("/Volumes/TOSHIBA EXT/All/dcase20", "dataset", "audio", "train",
                                       "synthetic20_reverbed")
    # out_default_reverb_path = osp.join(absolute_dir_path, "..", "..", "dataset", "audio", "train",
    #                                       "synthetic20_reverbed")
    default_json_path = osp.join(absolute_dir_path, "..", "..", "dataset", "metadata",
                                 "event_occurences", "event_occurences_train.json")

    parser = argparse.ArgumentParser()
    # generating files
    parser.add_argument('--out_folder', type=str, default=out_default_path,
                        help="the output folder in which to save the generated examples")
    parser.add_argument('--out_tsv', type=str, default=None,
                        help="output metadata file, if not defined the folder will be taken from --out_folder and "
                             "audio replaced to metadata in the path")
    parser.add_argument('--number', type=int, default=1000, help="the number of data to be generated")
    parser.add_argument('--soundbank', type=str, default=default_soundbank_path,
                        help="path to the soundbank folder (should contain foreground and background subfolders)")
    parser.add_argument('--json_path', type=str, default=default_json_path,
                        help=f"path to the json path with co-occurences, see {default_json_path} for an example of"
                        f"parameters defined (the structure has to be similar)")
    parser.add_argument('--nproc', type=int, default=8,
                        help="the number of parallel processes to be used if parallelizing")
    parser.add_argument('--random_seed', type=int, default=2020)

    # Apply reverb
    parser.add_argument('--rir_folder', type=str, default=None,
                        help="the Room impulse responses folder base path")
    parser.add_argument('--reverb_out_folder', type=str, default=None,
                        help="the output folder in which to save the reverberated generated examples")
    parser.add_argument('--reverb_out_tsv', type=str, default=None,
                        help="output metadata file, if not defined the folder will be taken from --reverb_out_folder "
                             "and audio replaced to metadata in the path")
    parser.add_argument('--rir_subset', type=str, default="train",
                        help="Choice between train, validation and eval")
    args = parser.parse_args()
    pformat(vars(args))

    # General parameters
    out_folder = args.out_folder
    soundbank_path = args.soundbank
    n_soundscapes = args.number
    random_state = args.random_seed
    rir_subset = args.rir_subset

    subset = "soundscapes"
    full_out_folder = osp.join(out_folder, subset)
    out_tsv = args.out_tsv
    if out_tsv is None:
        out_tsv = full_out_folder.replace("audio", "metadata") + ".tsv"
    create_folder(full_out_folder)
    create_folder(osp.dirname(out_tsv))
    # ############
    # Generate soundscapes
    # ############
    clip_duration = 10.0
    fg_folder = osp.join(soundbank_path, "foreground")
    bg_folder = osp.join(soundbank_path, "background")
    with open(args.json_path) as json_file:
        co_occur_dict = json.load(json_file)

    # Check for multiprocessing
    nproc = args.nproc
    if nproc > 1:
        if n_soundscapes // nproc < 200:
            nproc = n_soundscapes // 200
            if nproc == 0:
                nproc = 1
                warnings.warn(f"Not enough files to generate, using only one processor")
            else:
                warnings.warn(f"Be careful, less than 200 files generated per jobs can have an impact "
                              f"on the distribution of the classes, changing nproc to {nproc}")
    # ###########
    # Generate the data
    # ###########
    # Multiprocessing
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
            # We need to give a different random state to each object otherwise we just replicate 4 times the same data
            # except if it is None
            sg = SoundscapesGenerator(duration=clip_duration,
                                      fg_folder=fg_folder,
                                      bg_folder=bg_folder,
                                      ref_db=cfg.ref_db,
                                      samplerate=cfg.sample_rate,
                                      random_state=rand_state,
                                      logger=logger)
            sg.generate_by_label_occurence(label_occurences=co_occur_dict,
                                           number=number,
                                           out_folder=full_out_folder,
                                           save_isolated_events=True,
                                           start_from=start_from)
        if random_state is None:
            random_states = [None for i in range(nproc)]
        else:
            random_states = [random_state + i for i in range(nproc)]
        with multiprocessing.Pool(nproc) as p:
            p.starmap(generate_multiproc, zip(list_start, numbers, random_states))
    # Single process
    else:
        sg = SoundscapesGenerator(duration=clip_duration,
                                  fg_folder=fg_folder,
                                  bg_folder=bg_folder,
                                  ref_db=cfg.ref_db,
                                  samplerate=cfg.sample_rate,
                                  random_state=random_state)
        sg.generate_by_label_occurence(label_occurences=co_occur_dict,
                                       number=n_soundscapes,
                                       out_folder=full_out_folder,
                                       save_isolated_events=True)


    # ## Post processing
    rm_high_polyphony(full_out_folder, max_polyphony=2)
    # concat same labels overlapping
    post_process_txt_labels(full_out_folder,
                            output_folder=full_out_folder,
                            output_tsv=out_tsv)

    # ############
    # Apply RIR on the generated data (comes from source separation folder from Google code)
    # ############
    rir_folder = args.rir_folder
    if rir_folder is None:
        warnings.warn("Not generating reverbed version")
    else:
        source_dict = make_example_dict_from_folder(out_folder, subset=subset,
                                                    ss_regex=re.compile(r".*_events"), pattern="_events",
                                                    subfolder_events=None)
        rir_dict = make_rir_dict_from_folder(rir_folder)
        mix_info = make_mix_info_subsources({}, source_dict[subset], rir_dict[rir_subset])
        reverb_folder = args.reverb_out_folder
        if reverb_folder is None:
            reverb_folder = out_folder + "_reverb"
        if args.reverb_out_tsv is None:
            out_tsv = reverb_folder.replace("audio", "metadata") + ".tsv"

        if nproc > 1:
            def reverb_mix_partial(part):
                reverberate_and_mix(reverb_folder, out_folder, args.rir_folder,
                                    mix_info, scale_rirs=10.0, part=part, nparts=nproc,
                                    chat=True)
            with multiprocessing.Pool(nproc) as p:
                p.map(reverb_mix_partial, range(nproc))
        else:
            reverberate_and_mix(reverb_folder, out_folder, args.rir_folder,
                                mix_info, part=0, nparts=1,
                                chat=True)

        examples_list = make_example_list(reverb_folder)
        for example in examples_list:
            check_and_correct_example(example, reverb_folder,
                                      check_length=True, fix_length=True, check_mix=True, fix_mix=True,
                                      sample_rate=cfg.sample_rate, duration=clip_duration)

        print(time.time() - t)

