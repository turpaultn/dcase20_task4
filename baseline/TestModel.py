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
import glob
import os.path as osp
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from psds_eval import PSDSEval

from DataLoad import DataLoadDf
from Desed import DESED
from evaluation_measures import audio_tagging_results, get_f_measure_by_class, compute_strong_metrics, get_predictions
from utils.utils import ManyHotEncoder, to_cuda_if_available, get_transforms, generate_tsv_from_isolated_events, \
    generate_tsv_wav_durations
from utils.Logger import create_logger
from utils.Scaler import Scaler, ScalerPerAudio
from models.CRNN import CRNN
import config as cfg

from desed.utils import post_processing_df_annotations


logger = create_logger(__name__)


def _load_crnn(state):
    crnn_kwargs = state["model"]["kwargs"]
    crnn = CRNN(**crnn_kwargs)
    crnn.load(parameters=state["model"]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    return crnn


def _load_scaler(state):
    scaler = Scaler()
    scaler.load_state_dict(state["scaler"])
    return scaler


def _get_predictions(state, pred_df, save_preds_path=None, audio_source=None, nb_files=None):
    if save_preds_path is not None and os.path.exists(save_preds_path):
        warnings.warn(f"Predictions are not computing since {save_preds_path} already exists")
        predictions = pd.read_csv(save_preds_path, sep="\t")
    else:
        if audio_source is None:
            raise NameError(f"if {save_preds_path} is not already computed audio_source should be defined")
        dataset = DESED(os.path.join(cfg.workspace),
                        base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                        compute_log=False)
        # Keep only the number we want to test on
        if nb_files is not None:
            pred_df = DESED.get_subpart_data(pred_df, nb_files)
        pred_df = dataset.extract_features_from_df(pred_df, audio_source)

        pooling_time_ratio = state["pooling_time_ratio"]
        many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
        scaler = _load_scaler(state)
        crnn = _load_crnn(state)

        transforms_valid = get_transforms(cfg.max_frames, scaler=scaler)
        strong_dataload = DataLoadDf(pred_df, dataset.get_feature_file, many_hot_encoder.encode_strong_df,
                                     transform=transforms_valid, return_indexes=True)
        strong_dataloader_ind = DataLoader(strong_dataload, batch_size=cfg.batch_size, drop_last=False)
        predictions = get_predictions(crnn, strong_dataloader_ind, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      save_predictions=save_preds_path)
    return predictions


def test_model(state, gtruth_df, save_preds_path=None, audio_source=None, nb_files=None,):
    predictions = _get_predictions(state, gtruth_df, save_preds_path=save_preds_path,
                                   audio_source=audio_source, nb_files=nb_files)
    compute_strong_metrics(predictions, gtruth_df)

    # weak_dataload = DataLoadDf(df, dataset.get_feature_file, many_hot_encoder.encode_weak,
    #                            transform=transforms_valid)
    # weak_metric = get_f_measure_by_class(crnn, len(cfg.classes), DataLoader(weak_dataload, batch_size=cfg.batch_size))
    # logger.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
    # logger.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

    # Just an example of how to get the weak predictions from dataframes.
    # print(audio_tagging_results(df, predictions))
    return predictions


def test_model_ss(state, gtruth_df, folder_sources=None, pattern_isolated_events="_events",
                  nb_files=None, save_preds_path=None):
    isolated_ev_df = generate_tsv_from_isolated_events(folder_sources)

    predictions = _get_predictions(state, isolated_ev_df, save_preds_path, audio_source=folder_sources,
                                   nb_files=nb_files)
    print(predictions.head())
    predictions["filename"] = predictions.filename.apply(
        lambda x: x.split(os.sep)[0].split(pattern_isolated_events)[0] + ".wav")
    predictions = post_processing_df_annotations(predictions, 10)

    if nb_files is not None:
        gtruth_df = gtruth_df[gtruth_df.filename.isin(predictions.filename.tolist())]
    compute_strong_metrics(predictions, gtruth_df)
    return predictions


def psds_results(gtruth_df, meta_csv, predictions, data_dir=None):
    dtc_threshold = 0.5
    gtc_threshold = 0.5
    cttc_threshold = 0.3
    # If meta_table does not exist
    if not osp.exists(meta_csv):
        if data_dir is None:
            raise FileNotFoundError(f"psds_results, meta_csv={meta_csv} does not exist, "
                                    f"and no data_dir has been computed")
        meta_df = generate_tsv_wav_durations(data_dir, meta_csv)
    else:
        meta_df = pd.read_csv(meta_csv, sep='\t')
    # Instantiate PSDSEval
    psds = PSDSEval(dtc_threshold, gtc_threshold, cttc_threshold,
                    ground_truth=gtruth_df, metadata=meta_df)

    psds.add_operating_point(predictions)
    psds_score = psds.psds(alpha_ct=0, alpha_st=0, max_efpr=100)
    print(f"\nPSD-Score (0, 0, 100): {psds_score.value:.5f}")
    psds_score = psds.psds(alpha_ct=1, alpha_st=0, max_efpr=100)
    print(f"\nPSD-Score (1, 0, 100): {psds_score.value:.5f}")
    psds_score = psds.psds(alpha_ct=0, alpha_st=1, max_efpr=100)
    print(f"\nPSD-Score (0, 1, 100): {psds_score.value:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-m", '--model_path', type=str,
                        help="Path of the model to be evaluated")
    parser.add_argument("-s", '--save_predictions_path', type=str, default=None,
                        help="Path for the predictions to be saved (if needed)")
    parser.add_argument("-g", '--groundtruth_tsv', type=str,
                        help="Path of the groundtruth tsv file")
    # Next groundtruth variable could be ommited if same organization than DESED dataset
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of filenames and durations")
    parser.add_argument("-t", '--groundtruth_audio_dir', type=str, default=None,
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")

    parser.add_argument("-a", '--base_dir_ss', type=str, default=None,
                        help="Base directory where to search subdirectories in which there are isolated events")
    parser.add_argument("-z", '--save_predictions_path_ss', type=str, default=None,
                        help="Path for the predictions of source separation to be saved, if not set, they are not saved")

    f_args = parser.parse_args()

    gt_fname, ext = os.path.splitext(f_args.groundtruth_tsv)
    if f_args.meta_gt is None:
        meta_gt = gt_fname + "_meta" + ext
    else:
        meta_gt = f_args.meta_gt

    if f_args.groundtruth_audio_dir is None:
        gt_audio_dir = gt_fname.replace("audio", "metadata")
    else:
        gt_audio_dir = f_args.groundtruth_audio_dir

    model_path = f_args.model_path
    expe_state = torch.load(model_path, map_location="cpu")

    groundtruth_df = pd.read_csv(f_args.groundtruth_tsv, sep="\t")
    logger.info("\n #### Original files ####")
    pred_mixt = test_model(expe_state, groundtruth_df, save_preds_path=f_args.save_predictions_path,
                           audio_source=gt_audio_dir,
                           nb_files=f_args.nb_files)
    psds_results(
        gtruth_df=groundtruth_df,
        meta_csv=meta_gt,
        predictions=pred_mixt,
        data_dir=gt_audio_dir,
    )

    if f_args.save_predictions_path_ss is not None:
        logger.info("\n #### Separated sources ####")
        pred_ss = test_model_ss(expe_state, groundtruth_df, f_args.base_dir_ss, nb_files=f_args.nb_files,
                                save_preds_path=f_args.save_predictions_path_ss)
        psds_results(
            gtruth_df=groundtruth_df,
            meta_csv=meta_gt,
            predictions=pred_ss,
            data_dir=gt_audio_dir,
        )
        logger.info("\n #### Combination of original and separated sources ####")

        pred_ss = pd.read_csv(f_args.save_predictions_path_ss, sep="\t")
        predict_comb = pd.concat([pred_ss, pred_mixt]).reset_index(drop=True)
        logger.debug(predict_comb.head())
        logger.debug(predict_comb.tail())
        predictions = post_processing_df_annotations(predict_comb, 10)

        # # Event based
        event_metric, segment_metric = compute_strong_metrics(predictions, groundtruth_df)
        # Load metadata and ground truth tables
        psds_results(
            gtruth_df=groundtruth_df,
            meta_csv=meta_gt,
            predictions=predictions,
            data_dir=gt_audio_dir,
        )
