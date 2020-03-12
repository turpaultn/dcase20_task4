# -*- coding: utf-8 -*-
import time
import argparse
import os.path as osp
import glob
from pprint import pformat
import logging

from desed.generate_synthetic import generate_files_from_jams, generate_tsv_from_jams
from desed.logger import create_logger


if __name__ == '__main__':
    LOG = create_logger("DESED", terminal_level=logging.INFO)
    LOG.info(__file__)
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--jams_folder', type=str, required=True)
    parser.add_argument('--soundbank', type=str, required=True)
    parser.add_argument('--out_audio_dir', type=str)
    parser.add_argument('--out_tsv', type=str, default=None)
    parser.add_argument('--save_jams', action="store_true", default=False)
    parser.add_argument('--save_isolated', action="store_true", default=False)
    args = parser.parse_args()
    pformat(vars(args))

    # ########
    # Parameters
    # ########
    jams_folder = args.jams_folder
    list_jams = glob.glob(osp.join(jams_folder, "*.jams"))
    if len(list_jams) == 0:
        jams_folder = osp.join(jams_folder, "soundscapes")
        list_jams = glob.glob(osp.join(jams_folder, "*.jams"))
    if len(list_jams) == 0:
        raise IndexError(f"empty list jams. "
                         f"You need to provide a jams_folder with .jams files in it. your path: {jams_folder}")

    soundbank_dir = args.soundbank
    fg_path_train = osp.join(soundbank_dir, "foreground")
    bg_path_train = osp.join(soundbank_dir, "background")

    out_audio_dir = args.out_audio_dir
    if out_audio_dir is None:
        out_audio_dir = jams_folder

    if args.save_jams:
        out_folder_jams = out_audio_dir
    else:
        out_folder_jams = None

    out_tsv = args.out_tsv
    if out_tsv is None:
        out_tsv = out_audio_dir.replace("audio", "metadata") + ".tsv"

    save_isolated = args.save_isolated

    # ########
    # Generate
    # #######
    generate_files_from_jams(list_jams, out_audio_dir, out_folder_jams=out_folder_jams,
                             fg_path=fg_path_train, bg_path=bg_path_train, save_isolated_events=save_isolated)
    if out_tsv:
        generate_tsv_from_jams(list_jams, out_tsv)
