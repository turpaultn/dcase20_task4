# -*- coding: utf-8 -*-
import functools
import glob
import logging
import multiprocessing
import time
import argparse
import numpy as np
import os.path as osp
import pandas as pd
from pprint import pformat

from desed.generate_synthetic import SoundscapesGenerator, generate_files_from_jams
from desed.utils import create_folder, modify_fg_onset, modify_jams
from desed.post_process import rm_high_polyphony, post_process_txt_labels
from desed.logger import create_logger

import config_data as cfg


def parse_args():
    absolute_dir_path = osp.abspath(osp.dirname(__file__))
    default_soundbank_path = osp.join(absolute_dir_path, "..", "..", "synthetic", "audio", "train", "soundbank")
    out_default_path = osp.join(absolute_dir_path, "..", "..", "dataset", "audio", "train", "synthetic20")
    default_json_path = osp.join(absolute_dir_path, "..", "..", "dataset", "metadata",
                                 "event_occurences", "event_occurences_train.json")

    parser = argparse.ArgumentParser()
    audio_path_eval = osp.join("..", "synthetic", "audio", "eval")
    parser.add_argument('-o', '--out_folder', type=str, required=True)
    parser.add_argument('-sf', '--out_tsv_folder', type=str, required=True)
    parser.add_argument('-n', '--number', type=int, default=1000)
    parser.add_argument('-fg', '--fg_folder', type=str,
                        default=osp.join(audio_path_eval, "soundbank", "foreground_on_off"),
                        help="The foreground folder to be used to create the soundscapes. "
                             "By default we keep only the foreground with an onset and an offset, "
                             "you should consider doing the same.")
    parser.add_argument('-bg', '--bg_folder', type=str, default=osp.join(audio_path_eval, "soundbank", "background"))
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000,
                        help="The sample rate (in Hz) of the generated soundscapes "
                             "(be careful the soundbank you're using is >= to this sample rate)")
    parser.add_argument('-j', '--nproc', type=int, default=8,
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

    # General output folder, in args
    base_out_folder = args.out_folder
    create_folder(base_out_folder)
    out_tsv_folder = args.out_tsv_folder
    create_folder(out_tsv_folder)

    n_soundscapes = args.number
    fg_folder = args.fg_folder
    bg_folder = args.bg_folder
    nproc = args.nproc
    sample_rate = args.sample_rate
    random_state = args.random_seed

    # ################
    # Varying onset of a single event
    # ###########
    # SCAPER SETTINGS
    clip_duration = cfg.clip_duration
    ref_db = cfg.ref_db

    source_time_dist = 'const'
    source_time = 0.0

    event_duration_dist = 'uniform'
    event_duration_min = 0.25
    event_duration_max = 10.0

    snr_dist = 'uniform'
    snr_min = 6
    snr_max = 30

    pitch_dist = 'uniform'
    pitch_min = -3.0
    pitch_max = 3.0

    time_stretch_dist = 'uniform'
    time_stretch_min = 1
    time_stretch_max = 1

    event_time_dist = 'truncnorm'
    event_time_mean = 0.5
    event_time_std = 0.25
    event_time_min = 0.25
    event_time_max = 0.750

    out_folder_500 = osp.join(base_out_folder, "500ms")
    create_folder(out_folder_500)

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
            sg.generate(number, out_folder_500,
                        min_events=1, max_events=1, labels=('choose', []),
                        source_files=('choose', []),
                        sources_time=(source_time_dist, source_time),
                        events_start=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max),
                        events_duration=(event_duration_dist, event_duration_min, event_duration_max),
                        snrs=(snr_dist, snr_min, snr_max),
                        pitch_shifts=(pitch_dist, pitch_min, pitch_max),
                        time_stretches=(time_stretch_dist, time_stretch_min, time_stretch_max),
                        txt_file=True,
                        start_from=start_from)


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
        sg.generate(n_soundscapes, out_folder_500,
                    min_events=1, max_events=1, labels=('choose', []),
                    source_files=('choose', []),
                    sources_time=(source_time_dist, source_time),
                    events_start=(event_time_dist, event_time_mean, event_time_std, event_time_min, event_time_max),
                    events_duration=(event_duration_dist, event_duration_min, event_duration_max),
                    snrs=(snr_dist, snr_min, snr_max),
                    pitch_shifts=(pitch_dist, pitch_min, pitch_max),
                    time_stretches=(time_stretch_dist, time_stretch_min, time_stretch_max),
                    txt_file=True)

    rm_high_polyphony(out_folder_500, 2)
    out_tsv = osp.join(out_tsv_folder, "500ms.tsv")
    post_process_txt_labels(out_folder_500, output_folder=out_folder_500,
                            output_tsv=out_tsv)

    # Generate 2 variants of this dataset
    jams_to_modify = glob.glob(osp.join(out_folder_500, "*.jams"))
    # Be careful, if changing the values of the added onset value,
    # you maybe want to rerun the post_processing_annotations to be sure there is no inconsistency

    # 5.5s onset files
    out_folder_5500 = osp.join(base_out_folder, "5500ms")
    add_onset = 5.0
    modif_onset_5s = functools.partial(modify_fg_onset, slice_seconds=add_onset)
    list_jams5500 = modify_jams(jams_to_modify, modif_onset_5s, out_folder_5500)
    generate_files_from_jams(list_jams5500, out_folder_5500, out_folder_5500)
    # we also need to generate a new DataFrame with the right values
    df = pd.read_csv(out_tsv, sep="\t")
    df["onset"] += add_onset
    df["offset"] = df["offset"].apply(lambda x: min(10, x + add_onset))
    df.to_csv(osp.join(out_tsv_folder, "5500ms.tsv"),
              sep="\t", float_format="%.3f", index=False)

    # 9.5s onset files
    out_folder_9500 = osp.join(base_out_folder, "9500ms")
    add_onset = 9.0
    modif_onset_5s = functools.partial(modify_fg_onset, slice_seconds=add_onset)
    list_jams9500 = modify_jams(jams_to_modify, modif_onset_5s, out_folder_9500)
    generate_files_from_jams(list_jams9500, out_folder_9500, out_folder_9500)
    df = pd.read_csv(out_tsv, sep="\t")
    df["onset"] += add_onset
    df["offset"] = df["offset"].apply(lambda x: min(10, x + add_onset))
    df.to_csv(osp.join(out_tsv_folder, "9500ms.tsv"),
              sep="\t", float_format="%.3f", index=False)
