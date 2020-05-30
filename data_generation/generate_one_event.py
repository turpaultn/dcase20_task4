import argparse
import logging
import multiprocessing
from pprint import pformat

import numpy as np
import os.path as osp

from desed.utils import create_folder
from desed.post_process import post_process_txt_labels
from desed.generate_synthetic import SoundscapesGenerator


if __name__ == '__main__':
    dataset_path = osp.join("..", "..", "dataset")
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, required=True)
    parser.add_argument('-sb', '--soundbank_dir', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str,
                        default=osp.join(dataset_path, "audio", "train", "one_event_generated"))
    parser.add_argument('-s', '--outtsv', type=str,
                        default=osp.join(dataset_path, "metadata", "train", "one_event_generated.tsv"))
    parser.add_argument('-sr', '--samplerate', type=int, default=16000)
    parser.add_argument('-b', "--background_labels", type=str, default=None,
                        help="list of background labels to use, each separated by a comma")
    parser.add_argument('-j', "--nproc", type=int, default=1)
    args = parser.parse_args()
    pformat(vars(args))

    soundbank_path = args.soundbank_dir
    fg_folder = osp.join(soundbank_path, "foreground")
    bg_folder = osp.join(soundbank_path, "background")

    outfolder = args.out_dir
    out_tsv = args.outtsv
    # generate_one_event(args.number, fg_folder, bg_folder, outfolder,
    #                    bg_labels=background_labels,
    #                    nproc=args.nproc)

    create_folder(outfolder)

    n_soundscapes = args.number
    bg_labels = args.background_labels
    if bg_labels is not None:
        bg_labels = [s.strip() for s in args.background_labels.split(",")]

    sample_rate = args.samplerate
    nproc = args.nproc

    duration = 10
    ref_db = -50
    random_state = 2020
    list_labels = ['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog', 'Electric_shaver_toothbrush',
                   'Frying', 'Running_water', 'Speech', 'Vacuum_cleaner']

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
            sg = SoundscapesGenerator(duration=duration,
                                      fg_folder=fg_folder,
                                      bg_folder=bg_folder,
                                      ref_db=ref_db,
                                      samplerate=sample_rate,
                                      random_state=rand_state,
                                      logger=logger)
            sg.generate_balance(number=number,
                                out_folder=outfolder,
                                min_events=1,
                                max_events=1,
                                start_from=start_from,
                                pitch_shift=('uniform', -3, 3),
                                bg_labels=bg_labels)


        random_states = [random_state + i for i in range(nproc)]
        print(random_states)
        with multiprocessing.Pool(nproc) as p:
            p.starmap(generate_multiproc, zip(list_start, numbers, random_states))
    # Single process
    else:
        sg = SoundscapesGenerator(duration=duration,
                                  fg_folder=fg_folder,
                                  bg_folder=bg_folder,
                                  ref_db=ref_db,
                                  samplerate=sample_rate,
                                  random_state=random_state)
        sg.generate_balance(number=n_soundscapes,
                            out_folder=outfolder,
                            min_events=1,
                            max_events=1,
                            pitch_shift=('uniform', -3, 3),
                            bg_labels=bg_labels)

    # should give 0 corrections, and create the tsv file
    post_process_txt_labels(outfolder, output_tsv=out_tsv)
