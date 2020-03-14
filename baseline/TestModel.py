# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
import argparse
import os
import warnings

import torch
from torch.utils.data import DataLoader
import pandas as pd

from DataLoad import DataLoadDf, DataLoadDfSS
from Desed import DESED
from evaluation_measures import compute_sed_eval_metrics, get_predictions, \
    psds_results
from utilities.utils import ManyHotEncoder, to_cuda_if_available, get_transforms, generate_tsv_from_isolated_events, \
    generate_tsv_wav_durations, meta_path_to_audio_dir
from utilities.Logger import create_logger
from utilities.Scaler import Scaler, ScalerPerAudio
from models.CRNN import CRNN
import config as cfg

from desed.post_process import post_process_df_labels


logger = create_logger(__name__)


def _load_crnn(state):
    crnn_args = state["model"]["args"]
    crnn_kwargs = state["model"]["kwargs"]
    crnn = CRNN(*crnn_args, **crnn_kwargs)
    crnn.load(parameters=state["model"]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    return crnn


def _load_scaler(state):
    scaler_state = state["scaler"]
    type_sc = scaler_state["type"]
    if type_sc == "ScalerPerAudio":
        scaler = ScalerPerAudio(*scaler_state["args"])
    elif type_sc == "Scaler":
        scaler = Scaler()
    else:
        raise NotImplementedError("Not the right type of Scaler has been saved in state")
    scaler.load_state_dict(state["scaler"]["state_dict"])
    return scaler


def _get_predictions(state, strong_dataloader_ind, median_window=None, save_preds_path=None):
    pooling_time_ratio = state["pooling_time_ratio"]
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    if median_window is None:
        median_window = state["median_window"]
    crnn = _load_crnn(state)

    predictions = get_predictions(crnn, strong_dataloader_ind, many_hot_encoder.decode_strong, pooling_time_ratio,
                                  median_window=median_window, save_predictions=save_preds_path)
    return predictions


def test_model(state, gtruth_df, save_preds_path=None, median_window=None, add_axis_conv=True):
    if save_preds_path is not None and os.path.exists(save_preds_path):
        warnings.warn(f"Predictions are not computing since {save_preds_path} already exists")
        predictions = pd.read_csv(save_preds_path, sep="\t")
    else:
        pred_df = gtruth_df.copy()
        ### Define dataloader
        many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
        scaler = _load_scaler(state)
        transforms_valid = get_transforms(cfg.max_frames, scaler=scaler, add_axis_conv=add_axis_conv)

        strong_dataload = DataLoadDf(pred_df, many_hot_encoder.encode_strong_df,
                                     transform=transforms_valid, return_indexes=True)
        strong_dataloader_ind = DataLoader(strong_dataload, batch_size=cfg.batch_size, drop_last=False)

        predictions = _get_predictions(state, strong_dataloader_ind, median_window=median_window,
                                       save_preds_path=save_preds_path)

    compute_sed_eval_metrics(predictions, gtruth_df)

    # weak_dataload = DataLoadDf(df, dataset.get_feature_file, many_hot_encoder.encode_weak,
    #                            transform=transforms_valid)
    # weak_metric = get_f_measure_by_class(crnn, len(cfg.classes), DataLoader(weak_dataload, batch_size=cfg.batch_size))
    # logger.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    # logger.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Just an example of how to get the weak predictions from dataframes.
    # print(audio_tagging_results(df, predictions))
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", '--model_path', type=str, required=True,
                        help="Path of the ss_model to be evaluated")
    parser.add_argument("-g", '--groundtruth_tsv', type=str, required=True,
                        help="Path of the groundtruth tsv file")

    # Not required after that, but recommended to defined
    parser.add_argument("-mw", "--median_window", type=int, default=None,
                        help="Nb of frames for the median window, "
                             "if None the one defined for testing after training is used")

    # Next groundtruth variable could be ommited if same organization than DESED dataset
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of feat_filenames and durations")
    parser.add_argument("-t", '--groundtruth_audio_dir', type=str, default=None,
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")
    parser.add_argument("-s", '--save_predictions_path', type=str, default=None,
                        help="Path for the predictions to be saved (if needed)")

    # Source separation
    parser.add_argument("-a", '--base_dir_ss', type=str, default=None,
                        help="Base directory where to search subdirectories in which there are isolated events")

    # Dev
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")
    f_args = parser.parse_args()

    model_path = f_args.model_path
    gt_fname, ext = os.path.splitext(f_args.groundtruth_tsv)
    median_window = f_args.median_window
    meta_gt = f_args.meta_gt
    gt_audio_dir = f_args.groundtruth_audio_dir

    if meta_gt is None:
        meta_gt = gt_fname + "_durations" + ext

    if gt_audio_dir is None:
        gt_audio_dir = meta_path_to_audio_dir(gt_fname)
        # Useful because of the data format
        if "validation" in gt_audio_dir:
            gt_audio_dir = os.path.dirname(gt_audio_dir)

    if os.path.exists(meta_gt):
        meta_df = pd.read_csv(meta_gt, sep='\t')
        if len(meta_df) == 0:
            meta_df = generate_tsv_wav_durations(gt_audio_dir, meta_gt)
    else:
        meta_df = generate_tsv_wav_durations(gt_audio_dir, meta_gt)

    # Model
    expe_state = torch.load(model_path, map_location="cpu")
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False)
    if f_args.base_dir_ss is not None:
        groundtruth_df = dataset.initialize_and_get_df(f_args.groundtruth_tsv, gt_audio_dir, f_args.base_dir_ss,
                                                       pattern_ss="_events", nb_files=f_args.nb_files)

        logger.info("\n #### Separated sources ####")
        pred_ss = test_model(expe_state, groundtruth_df, save_preds_path=f_args.save_predictions_path,
                             median_window=median_window, add_axis_conv=False)
        psds_results(
            predictions=pred_ss,
            gtruth_df=groundtruth_df.drop("feature_filename", axis=1),
            gtruth_durations=meta_df
        )
        logger.info("\n #### Combination of original and separated sources ####")

    else:
        groundtruth_df = dataset.initialize_and_get_df(f_args.groundtruth_tsv, gt_audio_dir, nb_files=f_args.nb_files)
        logger.info("\n #### Original files ####")
        pred_mixt = test_model(expe_state, groundtruth_df, save_preds_path=f_args.save_predictions_path,
                               median_window=median_window)

        psds_results(
            predictions=pred_mixt,
            gtruth_df=groundtruth_df.drop("feature_filename", axis=1),
            gtruth_durations=meta_df
        )
