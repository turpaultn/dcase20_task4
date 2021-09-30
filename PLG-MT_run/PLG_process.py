# -*- coding: utf-8 -*-
import argparse
import os.path as osp
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import time
from data_utils.DataLoad import DataLoadDf
from data_utils.Desed import DESED
from evaluation_measures import psds_score, get_predictions, \
    compute_psds_from_operating_points, compute_metrics
from utilities.utils import to_cuda_if_available, generate_tsv_wav_durations, meta_path_to_audio_dir
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms
from utilities.Logger import create_logger
from utilities.Scaler import Scaler, ScalerPerAudio
from models.CRNN import CRNN
import config as cfg

logger = create_logger(__name__)
torch.manual_seed(2020)


def _load_crnn(state, model_name="model"):
    crnn_args = state[model_name]["args"]
    crnn_kwargs = state[model_name]["kwargs"]
    crnn = CRNN(*crnn_args, **crnn_kwargs)
    crnn.load_state_dict(state[model_name]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    logger.info(crnn)
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


def _load_state_vars(state, gtruth_df, median_win=None):
    pred_df = gtruth_df.copy()
    # Define dataloader
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    scaler = _load_scaler(state)
    crnn = _load_crnn(state)
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler, add_axis=0)

    strong_dataload = DataLoadDf(pred_df, many_hot_encoder.encode_strong_df, transforms_valid, return_indexes=True)
    strong_dataloader_ind = DataLoader(strong_dataload, batch_size=cfg.batch_size, drop_last=False)

    pooling_time_ratio = state["pooling_time_ratio"]
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    if median_win is None:
        median_win = state["median_window"]
    return {
        "model": crnn,
        "dataloader": strong_dataloader_ind,
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

    groundtruth = pd.read_csv(args.groundtruth_tsv, sep="\t")
    if osp.exists(meta_gt):
        meta_dur_df = pd.read_csv(meta_gt, sep='\t')
        if len(meta_dur_df) == 0:
            meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)
    else:
        meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)

    return model_pth, median_win, gt_audio_pth, groundtruth, meta_dur_df


def swap_columns(input_csv, output_csv):
    read_data = pd.read_csv(input_csv,sep='\t')
    read_data = read_data.loc[:,['filename', 'onset', 'offset', 'event_label']]
    read_data.to_csv(output_csv, index=False, sep='\t')


def extract_week(input_csv, output_csv):
    read_data = pd.read_csv(input_csv,sep='\t')
    read_data = read_data.loc[:,['filename', 'event_label']]
    read_data.to_csv(output_csv, index=False, sep='\t')


if __name__ == '__main__':

    # Model
    model_path = "stored_data/MeanTeacher_with_synthetic/model/baseline_best"
    median_window = None
    durations = None
    expe_state = torch.load(model_path, map_location="cpu")
    dataset = DESED(base_feature_dir=osp.join(cfg.workspace, "dataset", "features"), compute_log=False)

    # Pseudo unlabeled clip
    gt_audio_dir = "../dataset/audio/train/unlabel_in_domain"
    groundtruth_tsv = "../dataset/metadata/train/unlabel_in_domain.tsv"
    save_predictions_path = "stored_data/PLG_results/unlabel_in_domain_predictions.tsv"
    gt_df_feat = dataset.initialize_and_get_df(groundtruth_tsv, gt_audio_dir, nb_files=None)
    params = _load_state_vars(expe_state, gt_df_feat, median_window)
    single_predictions = get_predictions(params["model"], params["dataloader"],
                                         params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                         median_window=params["median_window"],
                                         save_predictions=save_predictions_path)

    # Pseudo week clip
    gt_audio_dir = "../dataset/audio/train/weak"
    groundtruth_tsv = "../dataset/metadata/train/weak.tsv"
    save_predictions_path = "stored_data/PLG_results/weak_predictions.tsv"
    gt_df_feat = dataset.initialize_and_get_df(groundtruth_tsv, gt_audio_dir, nb_files=None)
    params = _load_state_vars(expe_state, gt_df_feat, median_window)
    single_predictions = get_predictions(params["model"], params["dataloader"],
                                         params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                         median_window=params["median_window"],
                                         save_predictions=save_predictions_path)



    # step1: Generate UPS
    input_csv = 'stored_data/PLG_results/unlabel_in_domain_predictions.tsv'
    output_csv = 'stored_data/PLG_results/UPS.tsv'
    swap_columns(input_csv, output_csv)   
    
    # step2: Generate UPW
    input_csv = 'stored_data/PLG_results/unlabel_in_domain_predictions.tsv'
    output_csv = 'stored_data/PLG_results/UPW.tsv'
    extract_week(input_csv, output_csv)

    # step3: Generate WPS
    input_csv = 'stored_data/PLG_results/weak_predictions.tsv'
    output_csv = 'stored_data/PLG_results/WPS.tsv'
    swap_columns(input_csv, output_csv)
    
    # step4: True_before_Pseudo
    os.popen("cp -rf ../dataset/metadata/train ../dataset/metadata/train_TbS")
    os.popen("cd ../dataset/metadata/train_TbS")
    os.popen("sed '1d' stored_data/PLG_results/UPW.tsv >> ../dataset/metadata/train_TbS/weak.tsv") 
    os.popen("sed '1d' stored_data/PLG_results/UPS.tsv >> ../dataset/metadata/train_TbS/synthetic20/soundscapes.tsv")
    os.popen("sed '1d' stored_data/PLG_results/WPS.tsv >> ../dataset/metadata/train_TbS/synthetic20/soundscapes.tsv")
    os.popen("cp -rf ../dataset/audio/train ../dataset/audio/train_TbS")
    time.sleep(5*60)
    os.popen("cp -rf ../dataset/audio/train_TbS/unlabel_in_domain/* ../dataset/audio/train_TbS/weak/")
    os.popen("cp -rf ../dataset/audio/train_TbS/unlabel_in_domain/* ../dataset/audio/train_TbS/synthetic20/soundscapes/")
    os.popen("cp -rf ../dataset/audio/train_TbS/weak/* ../dataset/audio/train_TbS/synthetic20/soundscapes/")




 
