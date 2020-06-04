"""Apply RIR on the generated data (comes from source separation folder from Google code)"""
import argparse
import glob
import inspect
import multiprocessing
import os

import numpy as np
import os.path as osp
import pandas as pd
import logging
import re
import sys
import time
from pprint import pformat

from desed.logger import create_logger
from desed.post_process import get_data, _post_process_labels_file, save_tsv
from desed.utils import create_folder

import config_data as cfg


absolute_dir_path = osp.abspath(osp.dirname(__file__))
relative_path_ss_repo = osp.join(absolute_dir_path, "..")
base_dir_repo = osp.abspath(relative_path_ss_repo)
sys.path.append(osp.join(base_dir_repo, "sound-separation", "datasets", "fuss"))

from reverberate_and_mix import reverberate_and_mix, make_rir_dict_from_folder, make_mix_info_subsources, \
    read_mix_info, write_item_dict, write_mix_info
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


# Need to be adapted because reverb added lines in the label file.
def post_process_txt_labels(txtdir, wavdir=None, output_folder=None, output_tsv=None, min_dur_event=0.250,
                            min_dur_inter=0.150, background_label=False, rm_nOn_nOff=True):
    """ clean the .txt files of each file. It is the same processing as the real data
    - overlapping events of the same class are mixed
    - if silence < 150ms between two conscutive events of the same class, they are mixed
    - if event < 250ms, the event lasts 250ms

    Args:
        txtdir: str, directory path where the XXX.txt files are.
        wavdir: str, directory path where the associated XXX.wav audio files are (associated with .txt files)
        output_folder: str, optional, folder in which to put the checked files
        output_tsv: str, optional, tsv with all the annotations concatenated
        min_dur_event: float, optional in sec, minimum duration of an event
        min_dur_inter: float, optional in sec, minimum duration between 2 events
        background_label: bool, whether to include the background label in the annotations.
        rm_nOn_nOff: bool, whether to delete the additional _nOn _nOff at the end of labels.

    Returns:
        None
    """
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name)
    if wavdir is None:
        wavdir = txtdir
    fix_count = 0
    logger.info("Correcting annotations ... \n"
                "* annotations with negative duration will be removed\n" +
                "* annotations with duration <250ms will be extended on the offset side)")

    if output_folder is not None:
        create_folder(output_folder)

    df_single = pd.DataFrame()  # only useful if output_csv defined

    if background_label:
        list_files = glob.glob(osp.join(txtdir, "*.jams"))
    else:
        list_files = glob.glob(osp.join(txtdir, "*.txt"))
        if len(list_files) == 0:
            list_files = glob.glob(osp.join(txtdir, '*.jams'))
    out_extension = '.txt'
    for fn in list_files:
        logger.debug(fn)
        df, length_sec = get_data(fn, osp.join(wavdir, osp.splitext(osp.basename(fn))[0] + '.wav'),
                                  background_label=background_label)
        df = df.dropna()  # trick: will remove the 'rir' lines because they do not have \t
        df, fc = _post_process_labels_file(df, length_sec, min_dur_event, min_dur_inter, rm_nOn_nOff)
        fix_count += fc

        if output_folder is not None:
            filepath = os.path.splitext(os.path.basename(fn))[0] + out_extension
            df[['onset', 'offset', 'event_label']].to_csv(osp.join(output_folder, filepath),
                                                          header=False, index=False, sep="\t")
        if output_tsv is not None:
            df['filename'] = osp.join(osp.splitext(osp.basename(fn))[0] + '.wav')
            df_single = df_single.append(df[['filename', 'onset', 'offset', 'event_label']], ignore_index=True)

    if output_tsv:
        save_tsv(df_single, output_tsv)

    logger.info(f"{fix_count} problems Fixed")


if __name__ == '__main__':
    LOG = create_logger(__name__, terminal_level=logging.INFO)
    LOG.info(__file__)
    t = time.time()
    parser = argparse.ArgumentParser()
    # Apply reverb
    parser.add_argument('-r', '--rir_folder', help="the Room impulse responses folder base path",
                        type=str, default="rir_data", required=True)
    parser.add_argument('-i', '--input_folder', help="the input folder containing a subfolder called 'soundscapes'"
                                                  "conatining the data to reverberate",
                        type=str, default=None, required=True)
    parser.add_argument('-o', '--reverb_out_folder', help="the output folder in which to save the reverberated generated"
                                                         "examples",
                        type=str, default=None)
    parser.add_argument('--reverb_out_tsv', help="output metadata file, if not defined the folder will be taken "
                                                 "from --reverb_out_folder and audio replaced to metadata in the path",
                        type=str, default=None)
    parser.add_argument('--rir_subset', help="Choice between train, validation and eval", type=str, default="train")
    parser.add_argument('--clip_duation', help="Duration of the clips to reverberate, just for checking purposes",
                        type=float, default=10.)

    # Useful to reproduce a reverberated dataset
    parser.add_argument('--mix_info_file', help="if defined load the mix_info or create one if not exists and save it",
                        type=str, default="")
    parser.add_argument('--src_list_file', help="if defined load the src_list or create one if not exists and save it",
                        type=str, default="")
    parser.add_argument('--rir_list_file', help="if defined load the mix_info or create one if not exists and save it",
                        type=str, default="")
    parser.add_argument('-sr', '--sample_rate', type=int, default=16000,
                        help="The samplerate of the generated soundscapes "
                             "(be careful the soundbank you're using is >= to this samplerate)")
    parser.add_argument('--nproc', type=int, default=4, help="The number of parallel processes to use")
    parser.add_argument('-rs', '--random_seed', help='Random seed', type=int, default=2020)
    args = parser.parse_args()
    pformat(vars(args))

    # Parameters
    subset = "soundscapes"  # need to be the same in generate_synth_dcase20.py (last folder before sounds)
    nproc = args.nproc
    sample_rate = args.sample_rate
    clip_duration = cfg.clip_duration

    rir_folder = args.rir_folder
    input_folder = args.input_folder
    rir_subset = args.rir_subset
    reverb_folder = args.reverb_out_folder
    if reverb_folder is None:
        reverb_folder = input_folder + "_reverb"
    if args.reverb_out_tsv is None:
        out_tsv = osp.join(reverb_folder.replace("audio", "metadata"), subset + ".tsv")
    else:
        out_tsv = args.reverb_out_tsv

    create_folder(reverb_folder)
    create_folder(osp.dirname(out_tsv))
    # ########
    # Make lists of examples, rir and mix_info needed to reverberate
    # (see reverberate_and_mix.py from Google folder for more info)
    # ########
    src_list_file = args.src_list_file
    rir_list_file = args.rir_list_file
    mix_info_file = args.mix_info_file
    np.random.seed(args.random_seed)
    mix_info = None
    source_dict = None
    rir_dict = None
    if mix_info_file != "":
        if osp.exists(mix_info_file):
            mix_info = read_mix_info(mix_info_file)

    if mix_info is None:
        # Preparing lists to reverberate
        # Create a dictionnary of the isolated events used to create a soundscape
        source_dict = make_example_dict_from_folder(input_folder, subset=subset,
                                                    ss_regex=re.compile(r".*_events"), pattern="_events",
                                                    subfolder_events=None)
        if src_list_file != "":
            write_item_dict(source_dict, src_list_file)

        # Create the Room ipulse responses dictionnary
        rir_dict = make_rir_dict_from_folder(rir_folder)
        if rir_list_file != "":
            write_item_dict(rir_dict, rir_list_file)

        # Creat the mix we want to apply
        mix_info = make_mix_info_subsources({}, source_dict[subset], rir_dict[rir_subset],
                                            assign_rir_based_on_class=True)
        if mix_info_file != "":
            write_mix_info(mix_info, mix_info_file)

    # ########
    # Reverberate
    # ########
    if nproc > 1:
        def reverb_mix_partial(part):
            reverberate_and_mix(reverb_folder, input_folder, rir_folder,
                                mix_info, scale_rirs=10.0, part=part, nparts=nproc,
                                chat=True)


        with multiprocessing.Pool(nproc) as p:
            p.map(reverb_mix_partial, range(nproc))
    else:
        # Easy case without multiprocessing
        reverberate_and_mix(reverb_folder, input_folder, rir_folder,
                            mix_info, part=0, nparts=1,
                            chat=True)

    # Check the data (fix sox bug)
    examples_list = make_example_list(reverb_folder)
    for example in examples_list:
        check_and_correct_example(example, reverb_folder,
                                  check_length=True, fix_length=True, check_mix=True, fix_mix=True,
                                  sample_rate=sample_rate, duration=clip_duration)

    rev_subset_folder = osp.join(reverb_folder, subset)
    post_process_txt_labels(rev_subset_folder,
                            output_folder=rev_subset_folder,
                            output_tsv=out_tsv, rm_nOn_nOff=True)
    print(time.time() - t)
