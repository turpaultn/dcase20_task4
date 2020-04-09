# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp
import scipy

import torch
from dcase_util.data import ProbabilityEncoder
import pandas as pd
import numpy as np

from data_utils.DataLoad import DataLoadDf
from data_utils.Desed import DESED
from TestModel import _load_scaler, _load_crnn
from evaluation_measures import psds_score, compute_psds_from_operating_points, compute_metrics
from utilities.utils import to_cuda_if_available, generate_tsv_wav_durations, meta_path_to_audio_dir
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms
from utilities.Logger import create_logger
import config as cfg

logger = create_logger(__name__)


def norm_alpha(x, alpha_val):
    return ((1 / x.shape[0]) * (x ** alpha_val).sum(0)) ** (1 / alpha_val)


def get_predictions_ss_late_integration(model, valid_dataload, decoder, pooling_time_ratio=1, thresholds=[0.5],
                                        median_window=1, save_predictions=None, alpha=1):
    """ Get the predictions of a trained model on a specific set
    Args:
        model: torch.Module, a trained pytorch model (you usually want it to be in .eval() mode).
        valid_dataload: DataLoadDf, giving ((input_data, label), index) but label is not used here, the multiple
            data are the multiple sources (the mixture should always be the first one to appear, and then the sources)
            example: if the input data is (3, 1, timesteps, freq) there is the mixture and 2 sources.
        decoder: function, takes a numpy.array of shape (time_steps, n_labels) as input and return a list of lists
            of ("event_label", "onset", "offset") for each label predicted.
        pooling_time_ratio: the division to make between timesteps as input and timesteps as output
        median_window: int, the median window (in number of time steps) to be applied
        save_predictions: str or list, the path of the base_filename to save the predictions or a list of names
            corresponding for each thresholds
        thresholds: list, list of threshold to be applied
        alpha: float, the value of the norm to combine the predictions

    Returns:
        dict of the different predictions with associated threshold
    """

    # Init a dataframe per threshold
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    # Get predictions
    for i, ((input_data, _), index) in enumerate(valid_dataload):
        input_data = to_cuda_if_available(input_data)
        with torch.no_grad():
            pred_strong, _ = model(input_data)
        pred_strong = pred_strong.cpu()
        pred_strong = pred_strong.detach().numpy()
        if i == 0:
            logger.debug(pred_strong)

        pred_strong_sources = pred_strong[1:]
        pred_strong_sources = norm_alpha(pred_strong_sources, alpha)
        pred_strong_comb = norm_alpha(np.stack((pred_strong[0], pred_strong_sources), 0), alpha)

        # Get different post processing per threshold
        for threshold in thresholds:
            pred_strong_bin = ProbabilityEncoder().binarization(pred_strong_comb,
                                                                binarization_type="global_threshold",
                                                                threshold=threshold)
            pred_strong_m = scipy.ndimage.filters.median_filter(pred_strong_bin, (median_window, 1))
            pred = decoder(pred_strong_m)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            # Put them in seconds
            pred.loc[:, ["onset", "offset"]] *= pooling_time_ratio / (cfg.sample_rate / cfg.hop_size)
            pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(0, cfg.max_len_seconds)

            pred["filename"] = valid_dataload.filenames.iloc[index]
            prediction_dfs[threshold] = prediction_dfs[threshold].append(pred, ignore_index=True)

            if i == 0:
                logger.debug("predictions: \n{}".format(pred))
                logger.debug("predictions strong: \n{}".format(pred_strong_comb))

    # Save predictions
    if save_predictions is not None:
        if isinstance(save_predictions, str):
            if len(thresholds) == 1:
                save_predictions = [save_predictions]
            else:
                base, ext = osp.splitext(save_predictions)
                save_predictions = [osp.join(base, f"{threshold:.3f}{ext}") for threshold in thresholds]
        else:
            assert len(save_predictions) == len(thresholds), \
                f"There should be a prediction file per threshold: len predictions: {len(save_predictions)}\n" \
                f"len thresholds: {len(thresholds)}"
            save_predictions = save_predictions

        for ind, threshold in enumerate(thresholds):
            dir_to_create = osp.dirname(save_predictions[ind])
            if dir_to_create != "":
                os.makedirs(dir_to_create, exist_ok=True)
                if ind % 10 == 0:
                    logger.info(f"Saving predictions at: {save_predictions[ind]}. {ind + 1} / {len(thresholds)}")
                prediction_dfs[threshold].to_csv(save_predictions[ind], index=False, sep="\t", float_format="%.3f")

    list_predictions = []
    for key in prediction_dfs:
        list_predictions.append(prediction_dfs[key])

    if len(list_predictions) == 1:
        list_predictions = list_predictions[0]

    return list_predictions


def _load_state_vars(state, gtruth_df, median_win=None):
    pred_df = gtruth_df.copy()
    # Define dataloader
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    scaler = _load_scaler(state)
    crnn = _load_crnn(state)
    # Note, need to unsqueeze axis 1
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler, add_axis=1)

    # Note, no dataloader here
    strong_dataload = DataLoadDf(pred_df, many_hot_encoder.encode_strong_df, transforms_valid, return_indexes=True)

    pooling_time_ratio = state["pooling_time_ratio"]
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    if median_win is None:
        median_win = state["median_window"]
    return {
        "model": crnn,
        "dataload": strong_dataload,
        "pooling_time_ratio": pooling_time_ratio,
        "many_hot_encoder": many_hot_encoder,
        "median_window": median_win
    }


def get_variables(args):
    model_pth = args.model_path
    gt_fname, ext = osp.splitext(args.groundtruth_tsv)
    median_win = args.median_window
    meta_gt = args.meta_gt
    gt_audio_pth = args.groundtruth_audio_dir

    if meta_gt is None:
        meta_gt = gt_fname + "_durations" + ext

    if gt_audio_pth is None:
        gt_audio_pth = meta_path_to_audio_dir(gt_fname)
        # Useful because of the data format
        if "validation" in gt_audio_pth:
            gt_audio_pth = osp.dirname(gt_audio_pth)

    if osp.exists(meta_gt):
        meta_dur_df = pd.read_csv(meta_gt, sep='\t')
        if len(meta_dur_df) == 0:
            meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)
    else:
        meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)
    keep_sources = args.keep_sources
    if keep_sources is not None:
        keep_sources = keep_sources.split(",")

    return model_pth, median_win, gt_audio_pth, meta_dur_df, keep_sources


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", '--model_path', type=str, required=True,
                        help="Path of the model to be evaluated")
    parser.add_argument("-g", '--groundtruth_tsv', type=str, required=True,
                        help="Path of the groundtruth tsv file")
    # Source separation
    parser.add_argument("-a", '--base_dir_ss', type=str, required=True,
                        help="Base directory of source separation. "
                             "Path where to search subdirectories in which there are isolated events")
    parser.add_argument("-k", "--keep_sources", type=str, default=None,
                        help="The sources to be kept from the sound_separation (each source separated by a comma)."
                             "Example: '1,2' keeps the 2nd and 3rd sources (begins at 0).")

    # Not required after that, but recommended to defined
    parser.add_argument("-mw", "--median_window", type=int, default=None,
                        help="Nb of frames for the median window, "
                             "if None the one defined for testing after training is used")
    # Next groundtruth variable could be ommited if same organization than DESED dataset
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of feat_filenames and durations")
    parser.add_argument("-ga", '--groundtruth_audio_dir', type=str, default=None,
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")
    parser.add_argument("-s", '--save_predictions_path', type=str, default=None,
                        help="Path for the predictions to be saved (if needed)")

    # Dev only
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")
    f_args = parser.parse_args()

    # Get variables from f_args
    model_path, median_window, gt_audio_dir, durations, keep_sources = get_variables(f_args)

    expe_state = torch.load(model_path, map_location="cpu")
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"), compute_log=False)
    groundtruth = pd.read_csv(f_args.groundtruth_tsv, sep="\t")

    gt_df_feat_ss = dataset.initialize_and_get_df(f_args.groundtruth_tsv, gt_audio_dir, f_args.base_dir_ss,
                                                  pattern_ss="_events", nb_files=f_args.nb_files,
                                                  keep_sources=keep_sources)
    params = _load_state_vars(expe_state, gt_df_feat_ss, median_window)
    alpha_norm = 1
    # Preds with only one value (note that in comparison of TestModel, here we do not use a dataloader)
    single_predictions = get_predictions_ss_late_integration(params["model"], params["dataload"],
                                                             params["many_hot_encoder"].decode_strong,
                                                             params["pooling_time_ratio"],
                                                             median_window=params["median_window"],
                                                             save_predictions=f_args.save_predictions_path,
                                                             alpha=alpha_norm)
    compute_metrics(single_predictions, groundtruth, durations)

    # ##########
    # Optional but recommended
    # ##########
    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    pred_ss_thresh = get_predictions_ss_late_integration(params["model"], params["dataload"],
                                                         params["many_hot_encoder"].decode_strong,
                                                         params["pooling_time_ratio"],
                                                         thresholds=thresholds,
                                                         median_window=params["median_window"],
                                                         save_predictions=f_args.save_predictions_path)
    psds = compute_psds_from_operating_points(pred_ss_thresh, groundtruth, durations)
    psds_score(psds, filename_roc_curves=osp.splitext(f_args.save_predictions_path)[0] + "_roc.png")
