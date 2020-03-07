# -*- coding: utf-8 -*-
#########################################################################
# This file is derived from Curious AI/mean-teacher, under the Creative Commons Attribution-NonCommercial
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import argparse
import functools
import glob
import os
import time
import warnings
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from desed.utils import create_folder
from torch.utils.data import DataLoader
from torch import nn

from Desed import DESED
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler, logger
from TestModel import test_model
from evaluation_measures import get_f_measure_by_class, get_predictions, compute_sed_eval_metrics, psds_results
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio
from utilities.utils import ManyHotEncoder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, AverageMeterSet, generate_tsv_wav_durations, meta_path_to_audio_dir, audio_dir_to_meta_path

logger = create_logger(__name__)


def adjust_learning_rate(optimizer, rampup_value, rampdown_value):
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (beta1, beta2)
        param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, optimizer, epoch, ema_model=None, weak_mask=None, strong_mask=None):
    """ One epoch of a Mean Teacher ss_model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, ss_model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the ss_model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student ss_model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    meters = AverageMeterSet()

    logger.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    rampup_length = len(train_loader) * cfg.n_epoch // 2
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        if global_step < rampup_length:
            rampup_value = ramps.sigmoid_rampup(global_step, rampup_length)
        else:
            rampup_value = 1.0

        # Todo check if this improves the performance
        # adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        batch_input, ema_batch_input, target = to_cuda_if_available(batch_input, ema_batch_input, target)
        logger.debug(batch_input.mean())
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        strong_pred, weak_pred = model(batch_input)
        loss = None
        # Weak BCE Loss
        # Take the max in the time axis
        target_weak = target.max(-2)[0]
        if weak_mask is not None:
            weak_class_loss = class_criterion(weak_pred[weak_mask], target_weak[weak_mask])
            ema_class_loss = class_criterion(weak_pred_ema[weak_mask], target_weak[weak_mask])

            if i == 0:
                logger.debug("target: {}".format(target.mean(-2)))
                logger.debug("Target_weak: {}".format(target_weak))
                logger.debug("Target_weak mask: {}".format(target_weak[weak_mask]))
                logger.debug(weak_class_loss)
                logger.debug("rampup_value: {}".format(rampup_value))
            meters.update('weak_class_loss', weak_class_loss.item())

            meters.update('Weak EMA loss', ema_class_loss.item())

            loss = weak_class_loss

        # Strong BCE loss
        if strong_mask is not None:
            strong_class_loss = class_criterion(strong_pred[strong_mask], target[strong_mask])
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[strong_mask], target[strong_mask])
            meters.update('Strong EMA loss', strong_ema_class_loss.item())
            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:

            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(strong_pred,
                                                                               strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong

            meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost * consistency_criterion(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start

    logger.info(
        'Epoch: {}\t'
        'Time {:.2f}\t'
        '{meters}'.format(
            epoch, epoch_time, meters=meters))


# Todo try to test if this would work
def get_feature_file_ss(filename, ss_df, feature_dir, ss_pattern="_events"):
    fnames_no_pattern = ss_df.filename.apply(lambda x: x.split(ss_pattern)[0])
    fname_to_match = os.path.splitext(filename)[0]
    filenames_to_load = ss_df.filename[fnames_no_pattern == fname_to_match].tolist()
    filenames_to_load.append(filename)
    loaded_data = []
    for fname in filenames_to_load:
        fname = os.path.join(feature_dir, os.path.splitext(fname)[0] + ".npy")
        data = np.load(fname)
        loaded_data.append(data)
    arr_loaded_data = np.array(loaded_data)
    print(arr_loaded_data.shape)
    return arr_loaded_data

def add_ss_df(data_df, data_ss_df):
    df_data = data_df.copy()
    df_data_ss = data_ss_df.copy()
    df_data.loc[:, "id"] = df_data.filename.apply(lambda x: os.path.splitext(x)[0])
    # df_data_ss.loc[:, "id"] = df_data_ss.filename.apply(lambda x: )


def get_dfs(desed_dataset, reduced_nb_data):
    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, nb_files=reduced_nb_data)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, nb_files=reduced_nb_data)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, nb_files=reduced_nb_data, download=False)
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        nb_files=reduced_nb_data)

    # Divide weak in train and valid
    train_weak_df = weak_df.sample(frac=0.8, random_state=26)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    logger.debug(valid_weak_df.event_labels.value_counts())

    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    logger.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {"weak": weak_df,
                "train_weak": train_weak_df,
                "valid_weak": valid_weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df
                }

    return data_dfs


# Todo Generate separate_wavs on all the data (weak, unlabel in domain...)
# Function create df from ss_folder (write the ..._events folder in the df) (so maybe take a pattern as entry)
# Extract features from df from the dataset, will work since we use the .._events folder in df
# Create a Transform "ConcatMixturesSeparated" that replaces the unsqueeze_axis of ToTensor
# (how do we do for the folder ??, do we do a match ?)
# Adapt the CNN to take more than 1 channel at the beginning
def create_tsv_from_ss_folder(ss_folder, out_tsv=None):
    not_only_dir = False
    if out_tsv is None:
        out_tsv = audio_dir_to_meta_path(ss_folder)
    create_folder(os.path.dirname(out_tsv))

    if not os.path.exists(out_tsv):
        logger.info(f"generating: {out_tsv}, from {ss_folder}")
        list_events = []
        list_dirs = os.listdir(ss_folder)
        for ss_dir in list_dirs:
            dir_path = os.path.join(ss_folder, ss_dir)
            if os.path.isdir(dir_path):
                files_to_add = [os.path.join(ss_dir, fname) for fname in os.listdir(dir_path)]
                if len(files_to_add) == 0:
                    warnings.warn(f"Empty separated source folder {dir_path}")
                list_events.extend(files_to_add)
            else:
                not_only_dir = True
        if not_only_dir:
            warnings.warn(f"Be careful, not only directories of separated events in the folder specified: {ss_folder}")
        df_events = pd.DataFrame(list_events, columns=["filename"])
        df_events.to_csv(out_tsv, sep="\t", index=False)
    return out_tsv


# Todo reput as normal
def get_dfs_ss(desed_dataset, reduced_nb_data):
    weak_csv = create_tsv_from_ss_folder(cfg.weak_ss)
    unlabel_csv = create_tsv_from_ss_folder(cfg.unlabel_ss)
    synthetic_csv = create_tsv_from_ss_folder(cfg.synthetic_ss)
    validation_csv = create_tsv_from_ss_folder(cfg.validation_ss)

    # Extract features, audio_dir="" since we put the full path in the dataframe
    weak_df = desed_dataset.initialize_and_get_df(weak_csv, nb_files=reduced_nb_data)
    unlabel_df = desed_dataset.initialize_and_get_df(unlabel_csv, nb_files=reduced_nb_data)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(synthetic_csv, nb_files=reduced_nb_data, download=False)
    validation_df = desed_dataset.initialize_and_get_df(validation_csv, audio_dir=cfg.audio_validation_dir,
                                                        nb_files=reduced_nb_data)

    data_dfs = {
        "weak": weak_df,
        "unlabel": unlabel_df,
        "synthetic": synthetic_df,
        "validation": validation_df
    }
    return data_dfs


if __name__ == '__main__':
    logger.info("MEAN TEACHER")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                        help="Number of files to be used. Useful when testing on small number of files.")

    parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                        help="Not using synthetic labels during training")
    parser.add_argument("-ss", "--use_separated_sources", action="store_true", default=False,
                        help="If using this option, make sure you config.py points to the right folders")
    f_args = parser.parse_args()
    pprint(vars(f_args))

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    use_separated_sources = f_args.use_separated_sources

    if no_synthetic:
        add_dir_model_name = "_no_synthetic"
    else:
        add_dir_model_name = "_with_synthetic"

    store_dir = os.path.join("stored_data", "MeanTeacher" + add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "ss_model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(saved_pred_dir, exist_ok=True)

    pooling_time_ratio = cfg.pooling_time_ratio  # --> Be careful, it depends of the ss_model time axis pooling
    # ##############
    # DATA
    # ##############
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False)

    if use_separated_sources:
        dfs_ss = get_dfs_ss(dataset, reduced_number_of_data)
        add_axis_conv = False
        get_feature_file_weak = functools.partial(get_feature_file_ss, ss_df=dfs_ss["weak"],
                                                  feature_dir=dataset.feature_dir)
        get_feature_file_unlabel = functools.partial(get_feature_file_ss, ss_df=dfs_ss["unlabel"],
                                                     feature_dir=dataset.feature_dir)
        get_feature_file_synth = functools.partial(get_feature_file_ss, ss_df=dfs_ss["synthetic"],
                                                   feature_dir=dataset.feature_dir)
        get_feature_file_validation = functools.partial(get_feature_file_ss, ss_df=dfs_ss["validation"],
                                                        feature_dir=dataset.feature_dir)

    else:
        add_axis_conv = True
        get_feature_file_weak = dataset.get_feature_file
        get_feature_file_unlabel = dataset.get_feature_file
        get_feature_file_synth = dataset.get_feature_file
        get_feature_file_validation = dataset.get_feature_file

    dfs = get_dfs(dataset, reduced_number_of_data)

    # Meta path for psds
    path, ext = os.path.splitext(cfg.synthetic)
    path_durations_synth = path + "_durations" + ext
    if not os.path.exists(path_durations_synth):
        durations_synth = generate_tsv_wav_durations(meta_path_to_audio_dir(cfg.synthetic), path_durations_synth)
    else:
        durations_synth = pd.read_csv(path_durations_synth, sep="\t")

    classes = cfg.classes
    many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)

    scaler = ScalerPerAudio(cfg.normalization_on, cfg.normalization_type)
    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv=add_axis_conv, augment_type="noise")

    train_weak_data = DataLoadDf(dfs["train_weak"],
                                 get_feature_file_weak, many_hot_encoder.encode_strong_df,
                                 transform=transforms)
    unlabel_data = DataLoadDf(dfs["unlabel"],
                              get_feature_file_unlabel, many_hot_encoder.encode_strong_df,
                              transform=transforms)

    train_synth_data = DataLoadDf(dfs["train_synthetic"],
                                  get_feature_file_synth, many_hot_encoder.encode_strong_df,
                                  transform=transforms)

    if not no_synthetic:
        list_dataset = [train_weak_data, unlabel_data, train_synth_data]
        batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
        strong_mask = slice(cfg.batch_size//4 + cfg.batch_size//2, cfg.batch_size)
    else:
        list_dataset = [train_weak_data, unlabel_data]
        batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
        strong_mask = None
    # Assume weak data is always the first one
    weak_mask = slice(batch_sizes[0])

    # scaler = Scaler()
    # scaler.calculate_scaler(ConcatDataset(list_dataset))

    # logger.debug(scaler.mean_)

    # transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv=add_axis_conv, augment_type="noise")
    # for i in range(len(list_dataset)):
    #     list_dataset[i].set_transform(transforms)

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes)

    training_data = DataLoader(concat_dataset, batch_sampler=sampler)

    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler, add_axis_conv=add_axis_conv)
    valid_synth_data = DataLoadDf(dfs["valid_synthetic"],
                                  get_feature_file_synth, many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid, return_indexes=True)
    valid_synth_dataloader = DataLoader(valid_synth_data, batch_size=cfg.batch_size)
    valid_weak_data = DataLoadDf(dfs["valid_weak"],
                                 get_feature_file_weak, many_hot_encoder.encode_weak,
                                 transform=transforms_valid)

    # Eval 2018
    eval_2018_df = dataset.initialize_and_get_df(cfg.eval2018, audio_dir=cfg.audio_validation_dir,
                                                 nb_files=reduced_number_of_data)
    eval_2018 = DataLoadDf(eval_2018_df, get_feature_file_validation, many_hot_encoder.encode_strong_df,
                           transform=transforms_valid)

    # ##############
    # Model
    # ##############
    crnn_kwargs = cfg.crnn_kwargs
    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn_ema.apply(weights_init)
    logger.info(crnn)

    for param in crnn_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    state = {
        'ss_model': {"name": crnn.__class__.__name__,
                  'args': '',
                  "kwargs": crnn_kwargs,
                  'state_dict': crnn.state_dict()},
        'model_ema': {"name": crnn_ema.__class__.__name__,
                      'args': '',
                      "kwargs": crnn_kwargs,
                      'state_dict': crnn_ema.state_dict()},
        'optimizer': {"name": optimizer.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optimizer.state_dict()},
        "pooling_time_ratio": pooling_time_ratio,
        "scaler": scaler.state_dict(),
        "many_hot_encoder": many_hot_encoder.state_dict()
    }

    save_best_cb = SaveBest("sup")

    # ##############
    # Train
    # ##############
    for epoch in range(cfg.n_epoch):
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        crnn, crnn_ema = to_cuda_if_available(crnn, crnn_ema)

        train(training_data, crnn, optimizer, epoch, ema_model=crnn_ema, weak_mask=weak_mask, strong_mask=strong_mask)

        crnn = crnn.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_dataloader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      save_predictions=None)
        # Validation with synthetic data
        valid_synth = dfs["valid_synthetic"]
        valid_events_metric = compute_sed_eval_metrics(predictions, valid_synth)
        psds_results(predictions, valid_synth, durations_synth)

        logger.info("\n ### Valid weak metric ### \n")
        weak_metric = get_f_measure_by_class(crnn, len(classes),
                                             DataLoader(valid_weak_data, batch_size=cfg.batch_size))

        logger.info("Weak F1-score per class: \n {}".format(pd.DataFrame(weak_metric * 100, many_hot_encoder.labels)))
        logger.info("Weak F1-score macro averaged: {}".format(np.mean(weak_metric)))

        state['ss_model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_events_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if not no_synthetic:
                global_valid = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
                global_valid = global_valid + np.mean(weak_metric)
            else:
                global_valid = np.mean(weak_metric)
            if save_best_cb.apply(global_valid):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        logger.info("testing ss_model: {}".format(model_fname))
    else:
        logger.info("testing ss_model of last epoch: {}".format(cfg.n_epoch))

    # ##############
    # Validation
    # ##############
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")
    test_model(state, dfs["validation"], reduced_number_of_data, predicitons_fname)

    # ##############
    # Evaluation
    # ##############
    predicitons_eval2019_fname = os.path.join(saved_pred_dir, "baseline_eval2019.tsv")
    evaluation_df = pd.read_csv(cfg.eval_desed, sep="\t")
    test_model(state, eval, reduced_number_of_data, predicitons_eval2019_fname)
