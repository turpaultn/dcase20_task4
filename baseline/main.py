# -*- coding: utf-8 -*-
#########################################################################
# This file is derived from Curious AI/mean-teacher, under the Creative Commons Attribution-NonCommercial
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

import argparse
import functools
import glob
import inspect
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
from DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_crnn
from evaluation_measures import get_f_measure_by_class, get_predictions, compute_sed_eval_metrics, psds_results
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import ManyHotEncoder, SaveBest, to_cuda_if_available, weights_init, \
    get_transforms, AverageMeterSet, generate_tsv_wav_durations, meta_path_to_audio_dir, audio_dir_to_meta_path, \
    generate_tsv_from_isolated_events


def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that increases linearly

    Returns:

    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, optimizer, epoch, ema_model=None, weak_mask=None, strong_mask=None, adjust_lr=False):
    """ One epoch of a Mean Teacher model
    :param train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
    Should return 3 values: teacher input, student input, labels
    :param model: torch.Module, model to be trained, should return a weak and strong prediction
    :param optimizer: torch.Module, optimizer used to train the model
    :param epoch: int, the current epoch of training
    :param ema_model: torch.Module, student model, should return a weak and strong prediction
    :param weak_mask: mask the batch to get only the weak labeled data (used to calculate the loss)
    :param strong_mask: mask the batch to get only the strong labeled data (used to calcultate the loss)
    """
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    meters = AverageMeterSet()

    logger.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()

    # rampup_value = ramps.exp_rampup(epoch, cfg.n_epoch_rampup)
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        global_step = epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(train_loader))
        # Todo check if this improves the performance
        if adjust_lr:
            adjust_learning_rate(optimizer, rampup_value)
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
    return loss


def get_dfs(desed_dataset, reduced_nb_data, separated_sources=False):
    if separated_sources:
        audio_weak_ss = cfg.weak_ss
        audio_unlabel_ss = cfg.unlabel_ss
        audio_validation_ss = cfg.validation_ss
        audio_synthetic_ss = cfg.synthetic_ss
    else:
        audio_weak_ss = None
        audio_unlabel_ss = None
        audio_validation_ss = None
        audio_synthetic_ss = None

    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)

    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss,
                                                  nb_files=reduced_nb_data)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss,
                                                     nb_files=reduced_nb_data)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, audio_dir_ss=audio_synthetic_ss,
                                                       nb_files=reduced_nb_data, download=False)
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        audio_dir_ss=audio_validation_ss, nb_files=reduced_nb_data)

    # Divide weak in train and valid
    # train_weak_df = weak_df.sample(frac=0.8, random_state=26)
    # valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    # train_weak_df = train_weak_df.reset_index(drop=True)
    # logger.debug(valid_weak_df.event_labels.value_counts())

    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_length // pooling_time_ratio
    logger.debug(valid_synth_df.event_label.value_counts())

    # Eval 2018 to report results
    eval2018 = pd.read_csv(cfg.eval2018, sep="\t")
    eval_2018_df = validation_df[validation_df.filename.isin(eval2018.filename)]
    logger.info(f"eval2018 len: {len(eval_2018_df)}")

    data_dfs = {"weak": weak_df,
                # "train_weak": train_weak_df,
                # "valid_weak": valid_weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df,
                "eval2018": eval_2018_df
                }

    return data_dfs


# Todo Generate separate_wavs on all the data (weak, unlabel in domain...)
# Function create df from ss_folder (write the ..._events folder in the df) (so maybe take a pattern as entry)
# Extract features from df from the dataset, will work since we use the .._events folder in df
# Create a Transform "ConcatMixturesSeparated" that replaces the unsqueeze_axis of ToTensor
# (how do we do for the folder ??, do we do a match ?)
# Adapt the CNN to take more than 1 channel at the beginning


if __name__ == '__main__':
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
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

    if use_separated_sources:
        add_dir_model_name += "_ss"

    store_dir = os.path.join("stored_data", "MeanTeacher" + add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(saved_pred_dir, exist_ok=True)

    # Model and transform parameters different for source separated
    if use_separated_sources:
        n_channel = 5
        add_axis_conv = False
    else:
        n_channel = 1
        add_axis_conv = True

    n_layers = 5
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.classes), "attention": True, "n_RNN_cell": 128,
                   "n_layers_RNN": 2,
                   "activation": "glu",
                   "dropout": 0.5,
                   "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                   "nb_filters": [16,  32,  64,  128,  128],
                   "pooling": [[2, 2], [2, 2], [1, 2], [1, 4], [1, 4]]}
    pooling_time_ratio = 4  # 2 * 2
    out_nb_frames_1s = cfg.sample_rate / cfg.hop_length / pooling_time_ratio
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    logger.debug(f"median_window: {median_window}")
    # ##############
    # DATA
    # ##############
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False)

    dfs = get_dfs(dataset, reduced_number_of_data, separated_sources=use_separated_sources)

    # Meta path for psds
    path, ext = os.path.splitext(cfg.synthetic)
    path_durations_synth = path + "_durations" + ext
    if not os.path.exists(path_durations_synth):
        durations_synth = generate_tsv_wav_durations(meta_path_to_audio_dir(cfg.synthetic), path_durations_synth)
    else:
        durations_synth = pd.read_csv(path_durations_synth, sep="\t")

    classes = cfg.classes
    many_hot_encoder = ManyHotEncoder(classes, n_frames=cfg.max_frames // pooling_time_ratio)

    if cfg.scaler_type == "dataset":
        transforms = get_transforms(cfg.max_frames)
        train_weak_data = DataLoadDf(dfs["weak"], many_hot_encoder.encode_strong_df,
                                     transform=transforms)
        unlabel_data = DataLoadDf(dfs["unlabel"],
                                  many_hot_encoder.encode_strong_df,
                                  transform=transforms)
        train_synth_data = DataLoadDf(dfs["train_synthetic"],
                                      many_hot_encoder.encode_strong_df,
                                      transform=transforms)
        scaler_args = []
        scaler = Scaler()
        # # Only on real data since that's our final goal and test data are real
        scaler.calculate_scaler(ConcatDataset([train_weak_data, unlabel_data, train_synth_data]))
    else:
        scaler_args = [cfg.scale_peraudio_on, cfg.scale_peraudio_type]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv=add_axis_conv, augment_type="noise")
    transforms_valid = get_transforms(cfg.max_frames, scaler=scaler, add_axis_conv=add_axis_conv)

    train_weak_data = DataLoadDf(dfs["weak"], many_hot_encoder.encode_strong_df, transform=transforms,
                                 in_memory=cfg.in_memory)
    unlabel_data = DataLoadDf(dfs["unlabel"], many_hot_encoder.encode_strong_df, transform=transforms,
                              in_memory=cfg.in_memory)

    train_synth_data = DataLoadDf(dfs["train_synthetic"], many_hot_encoder.encode_strong_df, transform=transforms,
                                  in_memory=cfg.in_memory)

    valid_synth_data = DataLoadDf(dfs["valid_synthetic"], many_hot_encoder.encode_strong_df,
                                  transform=transforms_valid, return_indexes=True, in_memory=cfg.in_memory)

    logger.debug(f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, "
                 f"len weak: {len(train_weak_data)}")
    if not no_synthetic:
        list_dataset = [train_weak_data, unlabel_data, train_synth_data]
        batch_sizes = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
        strong_mask = slice((3*cfg.batch_size)//4, cfg.batch_size)
    else:
        list_dataset = [train_weak_data, unlabel_data]
        batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
        strong_mask = None
    # Assume weak data is always the first one
    weak_mask = slice(batch_sizes[0])

    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset,
                                      batch_sizes=batch_sizes)

    training_data = DataLoader(concat_dataset, batch_sampler=sampler)
    valid_synth_dataloader = DataLoader(valid_synth_data, batch_size=cfg.batch_size)

    # ##############
    # Model
    # ##############

    crnn_kwargs = crnn_kwargs
    crnn = CRNN(**crnn_kwargs)
    crnn_ema = CRNN(**crnn_kwargs)

    crnn.apply(weights_init)
    crnn_ema.apply(weights_init)
    logger.info(crnn)

    for param in crnn_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
    bce_loss = nn.BCELoss()

    state = {
        'model': {"name": crnn.__class__.__name__,
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
        "scaler": {
            "type": type(scaler).__name__,
            "args": scaler_args,
            "state_dict": scaler.state_dict()},
        "many_hot_encoder": many_hot_encoder.state_dict(),
        "median_window": median_window,
        "desed": dataset.state_dict()
    }

    save_best_cb = SaveBest("sup")

    # ##############
    # Train
    # ##############
    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
    for epoch in range(cfg.n_epoch):
        crnn = crnn.train()
        crnn_ema = crnn_ema.train()

        crnn, crnn_ema = to_cuda_if_available(crnn, crnn_ema)

        loss = train(training_data, crnn, optimizer, epoch,
                     ema_model=crnn_ema, weak_mask=weak_mask, strong_mask=strong_mask, adjust_lr=cfg.adjust_lr)

        crnn = crnn.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_dataloader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      median_window=median_window,
                                      save_predictions=None)
        # Validation with synthetic data (dropping feature_filename for psds)
        valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        valid_events_metric = compute_sed_eval_metrics(predictions, valid_synth)
        try:
            psds_results(predictions, valid_synth, durations_synth)
        except Exception as e:
            logger.error("psds did not work ....")

        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optimizer.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_events_metric.results()
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        valid_synth_f1 = valid_events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
        if cfg.save_best:
            if save_best_cb.apply(valid_synth_f1):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)

            results.loc[epoch, "global_valid"] = valid_synth_f1
        results.loc[epoch, "loss"] = loss.item()
        results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

    if cfg.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        crnn = _load_crnn(state)
        logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
    else:
        logger.info("testing model of last epoch: {}".format(cfg.n_epoch))
    results_df = pd.DataFrame(results).to_csv(os.path.join(saved_pred_dir, "results.tsv"),
                                              sep="\t", index=False, float_format="%.4f")
    # ##############
    # Validation
    # ##############
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv=add_axis_conv)
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")
    predicitons_fname18 = os.path.join(saved_pred_dir, "baseline_eval18.tsv")

    validation_data = DataLoadDf(dfs["validation"], many_hot_encoder.encode_strong_df,
                                 transform=transforms_valid, return_indexes=True, in_memory=cfg.in_memory)
    eval18_data = DataLoadDf(dfs["eval2018"], many_hot_encoder.encode_strong_df,
                             transform=transforms_valid, return_indexes=True, in_memory=cfg.in_memory)

    validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    valid_predictions = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                        pooling_time_ratio, median_window=median_window,
                                        save_predictions=predicitons_fname)
    compute_sed_eval_metrics(valid_predictions, dfs["validation"])

    # Eval 2018
    eval18_dataloader = DataLoader(eval18_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    eval18_predictions = get_predictions(crnn, eval18_dataloader, many_hot_encoder.decode_strong,
                                         pooling_time_ratio, median_window=median_window,
                                         save_predictions=predicitons_fname18)
    compute_sed_eval_metrics(eval18_predictions, dfs["eval2018"])
