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

from DataLoad import DataLoadDf
from Desed import DESED
from evaluation_measures import compute_sed_eval_metrics, get_predictions, \
    psds_results
from utilities.utils import ManyHotEncoder, to_cuda_if_available, get_transforms, generate_tsv_from_isolated_events, \
    generate_tsv_wav_durations
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
    scaler.load_state_dict(state["scaler"])
    return scaler


def _get_predictions(state, pred_df, save_preds_path=None, audio_source=None, nb_files=None):
    if save_preds_path is not None and os.path.exists(save_preds_path):
        warnings.warn(f"Predictions are not computing since {save_preds_path} already exists")
        predictions = pd.read_csv(save_preds_path, sep="\t")
    else:
        if audio_source is None:
            raise NameError(f"if {save_preds_path} is not already computed audio_source should be defined")
        dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
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
    compute_sed_eval_metrics(predictions, gtruth_df)

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
    fnames_to_match = isolated_ev_df.filename.apply(lambda x: (x.split(pattern_isolated_events)[0] + ".wav"))
    to_pred_df = gtruth_df[gtruth_df.filename.isin(fnames_to_match)]

    predictions = _get_predictions(state, to_pred_df, save_preds_path, audio_source=folder_sources,
                                   nb_files=nb_files)
    print(predictions.head())
    predictions["filename"] = predictions.filename.apply(
        lambda x: x.split(os.sep)[0].split(pattern_isolated_events)[0] + ".wav")
    predictions = post_process_df_labels(predictions, 10)

    if nb_files is not None:
        gtruth_df = gtruth_df[gtruth_df.filename.isin(predictions.filename.tolist())]
    compute_sed_eval_metrics(predictions, gtruth_df)
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-m", '--model_path', type=str,
                        help="Path of the ss_model to be evaluated")
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

    if not os.path.exists(meta_gt):
        meta_df = generate_tsv_wav_durations(gt_audio_dir, meta_gt)
    else:
        meta_df = pd.read_csv(meta_gt, sep='\t')

    psds_results(
        predictions=pred_mixt,
        gtruth_df=groundtruth_df,
        gtruth_durations=meta_df
    )
    # Todo remove the concatenation of ss and sed, since it will be merged in the model
    if f_args.save_predictions_path_ss is not None:
        logger.info("\n #### Separated sources ####")
        pred_ss = test_model_ss(expe_state, groundtruth_df, f_args.base_dir_ss, nb_files=f_args.nb_files,
                                save_preds_path=f_args.save_predictions_path_ss)
        psds_results(
            predictions=pred_ss,
            gtruth_df=groundtruth_df,
            gtruth_durations=meta_df
        )
        logger.info("\n #### Combination of original and separated sources ####")

        pred_ss = pd.read_csv(f_args.save_predictions_path_ss, sep="\t")
        predict_comb = pd.concat([pred_ss, pred_mixt]).reset_index(drop=True)
        logger.debug(predict_comb.head())
        logger.debug(predict_comb.tail())
        predictions = post_process_df_labels(predict_comb, 10)

        # # Event based
        event_metric = compute_sed_eval_metrics(predictions, groundtruth_df)
        # Load metadata and ground truth tables
        psds_results(
            predictions=predictions,
            gtruth_df=groundtruth_df,
            gtruth_durations=meta_df
        )
