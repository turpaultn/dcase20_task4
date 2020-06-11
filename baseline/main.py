# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from data_utils.Desed import DESED
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_crnn
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics, \
    audio_tagging_results, get_f_measure_by_class, get_f1_sed_score, bootstrap, get_psds_ct, get_f1_psds
from models.CRNN import CRNN
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms


def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
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
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def train(train_loader, model, optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None, rampup=None):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        rampup: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i
        if rampup is not None:
            rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(train_loader))
            if rampup in ["lr", "all"]:
                adjust_learning_rate(optimizer, rampup_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        batch_input, ema_batch_input, target = to_cuda_if_available(batch_input, ema_batch_input, target)
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()
        strong_pred, weak_pred = model(batch_input)

        loss = None
        # Weak BCE Loss
        target_weak = target.max(-2)[0]  # Take the max in the time axis
        if mask_weak is not None:
            weak_class_loss = class_criterion(weak_pred[mask_weak], target_weak[mask_weak])
            ema_class_loss = class_criterion(weak_pred_ema[mask_weak], target_weak[mask_weak])
            loss = weak_class_loss

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                          f"Target weak mask: {target_weak[mask_weak]} \n "
                          f"weak loss: {weak_class_loss} \t"
                          f"tensor mean: {batch_input.mean()}")
                if rampup:
                    log.debug(f"rampup_value: {rampup_value}")
            meters.update('weak_class_loss', weak_class_loss.item())
            meters.update('Weak EMA loss', ema_class_loss.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss = class_criterion(strong_pred[mask_strong], target[mask_strong])
            strong_ema_class_loss = class_criterion(strong_pred_ema[mask_strong], target[mask_strong])
            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

            if i == 0:
                log.debug(f"Target strong mask: {target[mask_strong].sum(-2)}\n")
            meters.update('Strong loss', strong_class_loss.item())
            meters.update('Strong EMA loss', strong_ema_class_loss.item())

        # Teacher-student consistency cost
        if ema_model is not None:
            if rampup in ["all", "consistency"]:
                consistency_weight = cfg.max_consistency_cost * rampup_value
            else:
                consistency_weight = cfg.max_consistency_cost
            meters.update('Consistency weight', consistency_weight)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_weight * consistency_criterion(strong_pred, strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong
            meters.update('Consistency weight', consistency_weight)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_weight * consistency_criterion(weak_pred, weak_pred_ema)
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
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss


def get_dfs(desed_dataset, subsets, nb_files=None, separated_sources=False, no_ps=None, reverb=None):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_unlabel_ss = cfg.unlabel_ss if separated_sources else None
    audio_validation_ss = cfg.validation_ss if separated_sources else None
    audio_synthetic_ss = cfg.synthetic_ss if separated_sources else None

    # Always take weak since we need the validation part
    audio_weak_ss = cfg.weak_ss if separated_sources else None
    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss, nb_files=nb_files)
    train_weak_df = weak_df.sample(frac=0.9)
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)

    # Event if synthetic not used for training, used on validation purpose
    train_synth_pth = cfg.train_synth
    valid_synth_pth = cfg.valid_synth
    if no_ps is not None:
        if no_ps in ["valid", "all"]:
            if reverb == "valid":
                valid_synth_pth = cfg.valid_synth_no_ps_reverb
            else:
                valid_synth_pth = cfg.valid_synth_no_ps
            if no_ps == "all":
                train_synth_pth = cfg.train_synth_no_ps
        else:
            raise NotImplementedError("no_ps in get_dfs() can be only in {None, 'valid', 'all'}")
    elif reverb is not None:
        if reverb in ["valid", "all"]:
            valid_synth_pth = cfg.valid_synth_reverb
            if reverb == "all":
                train_synth_pth = cfg.train_synth_reverb
        else:
            raise NotImplementedError("reverb in get_dfs() can be only in {None, 'valid', 'all'}")

    train_synth_df = desed_dataset.initialize_and_get_df(train_synth_pth, audio_dir_ss=audio_synthetic_ss,
                                                         nb_files=nb_files)
    valid_synth_df = desed_dataset.initialize_and_get_df(valid_synth_pth, audio_dir_ss=audio_synthetic_ss,
                                                         nb_files=nb_files)

    log.debug(f"synthetic: {train_synth_df.head()}")
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation,
                                                        audio_dir_ss=audio_validation_ss, nb_files=nb_files)
    # Todo find a better way to avoid that
    # Put train_synth in frames so many_hot_encoder can work.
    # Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {
        "train_weak": train_weak_df,
        "valid_weak": valid_weak_df,
        "train_synthetic": train_synth_df,
        "valid_synthetic": valid_synth_df,
        "validation": validation_df,
    }

    # Unlabel is only for training
    if "unlabel" in subsets:
        unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss, nb_files=nb_files)
        data_dfs["unlabel"] = unlabel_df

    return data_dfs


def set_model_name(subsets):
    add_model_name = ""
    if "synthetic" in subsets:
        add_model_name += "_s"
    if "unlabel" in subsets:
        add_model_name += "_u"
    if "weak" in subsets:
        add_model_name += "_w"
    return add_model_name


def set_train_dataset(subsets, dfs_dict, encoding_func=None, batch_nbs=None):
    # By default batch sizes
    if batch_nbs is None:
        subsets.sort()
        if subsets == ["synthetic", "unlabel", "weak"]:
            batch_nbs = [cfg.batch_size//4, cfg.batch_size//2, cfg.batch_size//4]
        elif subsets == ["unlabel", "weak"]:
            batch_nbs = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
        elif subsets == ["synthetic", "unlabel"]:
            batch_nbs = [3 * cfg.batch_size // 4, cfg.batch_size // 4]
        elif subsets == ["synthetic", "weak"]:
            batch_nbs = [cfg.batch_size // 2, cfg.batch_size // 2]
        else:
            assert len(subsets) == 1, "subsets is in the wrong format"
            batch_nbs = [cfg.batch_size]

    assert len(subsets) == len(batch_nbs), "The number of subsets should match the number of batch_prob"

    batch_szs = []
    list_data = []
    if "synthetic" in subsets:
        bs_synth = batch_nbs[subsets.index("synthetic")]
        batch_szs.append(bs_synth)
        strong_slice = slice(bs_synth)
        synth_data = DataLoadDf(dfs_dict["train_synthetic"], encoding_func, in_memory=cfg.in_memory)
        list_data.append(synth_data)
        logger.debug(f"len synthetic: {len(synth_data)}")
    else:
        strong_slice = None

    if "unlabel" in subsets:
        batch_szs.append(batch_nbs[subsets.index("unlabel")])
        unlabel_data = DataLoadDf(dfs_dict["unlabel"], encoding_func, in_memory=cfg.in_memory)
        list_data.append(unlabel_data)
        logger.debug(f"len synthetic: {len(unlabel_data)}")

    if "weak" in subsets:
        bs_weak = batch_nbs[subsets.index("weak")]
        batch_szs.append(bs_weak)
        weak_slice = slice(sum(batch_nbs) - bs_weak, sum(batch_nbs))
        weak_data = DataLoadDf(dfs_dict["train_weak"], encoding_func, in_memory=cfg.in_memory)
        list_data.append(weak_data)
        logger.debug(f"len synthetic: {len(weak_data)}")
    else:
        weak_slice = None

    logger.debug(f"len data: {[len(dt) for dt in list_data]}\n"
                 f"batch_szs: {batch_szs}\n"
                 f"strong mask {strong_slice}\n"
                 f"weak mask: {weak_slice}")

    return list_data, batch_szs, strong_slice, weak_slice


if __name__ == '__main__':
    torch.manual_seed(2020)
    np.random.seed(2020)
    logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-sb", '--subsets', nargs='+', default=["weak", "unlabel", "synthetic"],
                        help="choose a subset of the data, if multiple separated per space. Example:"
                             "'-sb weak unlabel' subset possibilities: {'weak', 'unlabel', 'synthetic'}")
    parser.add_argument("-bs", "--batch_sizes", type=int, nargs='+', default=None,
                        help="The number of each subset per batch. "
                             "Number of batch_sizes should be equal to the number of subsets.")
    parser.add_argument("-snr", "--noise_snr", type=float, default=None)
    parser.add_argument("-s", '--subpart_data', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")
    parser.add_argument("-r", "--rampup", type=str, default="all",
                        help="Rampup applied or not, possible values: {'lr', 'consistency', 'all'}")
    parser.add_argument("-np", "--no_ps", type=str, default=None,
                        help="If pitch shifting not wanted, values possible: ['valid', all']")
    parser.add_argument("--reverb", type=str, default=None,
                        help="If reverb wants to be applied, values possible: ['valid', all']")
    f_args = parser.parse_args()
    pprint(vars(f_args))

    reduced_number_of_data = f_args.subpart_data
    subset_list = f_args.subsets
    batch_sizes = f_args.batch_sizes
    noise_snr = f_args.noise_snr
    assert f_args.rampup in [None, "lr", "consistency", "all"], \
        "Rampup need to be in [None, 'lr', 'consistency', 'all']"

    if subset_list is None:
        subset_list = ["synthetic", "unlabel", "weak"]
    else:
        for val in subset_list:
            if val not in ["synthetic", "unlabel", "weak"]:
                raise NotImplementedError("Available subsets are: 'synthetic', 'unlabel' and 'weak'")

    # ##########
    # General params
    # #########
    n_channel = 1
    add_axis_conv = 0
    # Model taken from 2nd of dcase19 challenge: see Delphin-Poulat2019 in the results.
    n_layers = 7
    crnn_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.classes), "attention": True, "n_RNN_cell": 128,
                   "n_layers_RNN": 2,
                   "activation": "glu",
                   "dropout": 0.5,
                   "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                   "nb_filters": [16,  32,  64,  128,  128, 128, 128],
                   "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}
    pooling_time_ratio = 4  # 2 * 2

    out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio
    median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)
    logger.debug(f"median_window: {median_window}")

    # Path to save model and preds
    add_dir_model_name = set_model_name(subset_list)
    store_dir = os.path.join("stored_data", "MeanTeacher" + add_dir_model_name)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")
    os.makedirs(store_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(saved_pred_dir, exist_ok=True)

    # Meta path for psds
    durations_valid_synth = get_durations_df(cfg.valid_synth)
    many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
    encod_func = many_hot_encoder.encode_strong_df

    # ##############
    # DATA
    # ##############
    dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                    compute_log=False, use_multiprocessing=False)
    dfs = get_dfs(dataset, subset_list, reduced_number_of_data, no_ps=f_args.no_ps, reverb=f_args.reverb)

    list_dataset, batch_sizes, strong_mask, weak_mask = set_train_dataset(subset_list, dfs, encod_func, batch_sizes)

    # Normalisation per audio or on the full dataset
    if cfg.scaler_type == "dataset":
        transforms = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
        for dset in list_dataset:
            dset.set_transform(transforms)
        scaler_args = []
        scaler = Scaler()
        # # Only on real data since that's our final goal and test data are real
        scaler.calculate_scaler(ConcatDataset(list_dataset))
        logger.debug(f"scaler mean: {scaler.mean_}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    if noise_snr is None:
        noise_dict_params = None
    else:
        noise_dict_params = {"mean": 0., "snr": noise_snr}
    transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                noise_dict_params=noise_dict_params, add_teacher=True)
    for dset in list_dataset:
        dset.set_transform(transforms)
    concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)
    training_loader = DataLoader(concat_dataset, batch_sampler=sampler, num_workers=cfg.num_workers)

    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)
    valid_synth_data = DataLoadDf(dfs["valid_synthetic"], encod_func, transforms_valid,
                                  return_indexes=True, in_memory=cfg.in_memory)
    valid_synth_loader = DataLoader(valid_synth_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    valid_weak_data = DataLoadDf(dfs["valid_weak"], encod_func, transforms_valid,
                                 return_indexes=True, in_memory=cfg.in_memory)
    valid_weak_loader = DataLoader(valid_weak_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    # ##############
    # Model
    # ##############
    # Model
    crnn = CRNN(**crnn_kwargs)
    pytorch_total_params = sum(p.numel() for p in crnn.parameters() if p.requires_grad)
    logger.info(crnn)
    logger.info("number of parameters in the model: {}".format(pytorch_total_params))
    crnn.apply(weights_init)

    crnn_ema = CRNN(**crnn_kwargs)
    crnn_ema.apply(weights_init)
    for param in crnn_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
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
        'optimizer': {"name": optim.__class__.__name__,
                      'args': '',
                      "kwargs": optim_kwargs,
                      'state_dict': optim.state_dict()},
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
    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

    # ##############
    # Train
    # ##############
    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
    for epoch in range(2):#cfg.n_epoch):
        crnn.train()
        crnn_ema.train()
        crnn, crnn_ema = to_cuda_if_available(crnn, crnn_ema)

        loss_value = train(training_loader, crnn, optim, epoch,
                           ema_model=crnn_ema, mask_weak=weak_mask, mask_strong=strong_mask,
                           rampup=f_args.rampup)

        # Validation
        crnn = crnn.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(crnn, valid_synth_loader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                      median_window=median_window, save_predictions=None)
        # Validation with synthetic data (dropping feature_filename for psds)
        valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        valid_synth_f1, lvf1, hvf1 = bootstrap(predictions, valid_synth, get_f1_sed_score)
        psds_f1_valid, lvps, hvps = bootstrap(predictions, valid_synth, get_f1_psds, meta_df=durations_valid_synth)

        logger.info(f"F1 event_based: {valid_synth_f1}, +- {max(valid_synth_f1-lvf1, hvf1 - valid_synth_f1)},\n"
                    f"Psds ct: {psds_f1_valid}, +- {max(psds_f1_valid - lvps, hvps - psds_f1_valid)}")

        valid_weak_f1_pc = get_f_measure_by_class(crnn, len(many_hot_encoder.labels), valid_weak_loader)
        valid_weak_f1 = np.mean(valid_weak_f1_pc)
        logger.info(f"\n ### Valid weak metric \n F1 per class: {valid_weak_f1_pc} \n Macro average: {valid_weak_f1}")
        # Update state
        state['model']['state_dict'] = crnn.state_dict()
        state['model_ema']['state_dict'] = crnn_ema.state_dict()
        state['optimizer']['state_dict'] = optim.state_dict()
        state['epoch'] = epoch
        state['valid_metric'] = valid_synth_f1
        state['psds_f1_valid'] = psds_f1_valid
        state['valid_weak_f1'] = valid_weak_f1

        global_valid = valid_weak_f1 + valid_synth_f1
        # Callbacks
        if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if cfg.save_best:
            if save_best_cb.apply(valid_synth_f1):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)
            results.loc[epoch, "global_valid"] = global_valid
        results.loc[epoch, "loss"] = loss_value.item()
        results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

        if cfg.early_stopping:
            if early_stopping_call.apply(global_valid):
                logger.warn("EARLY STOPPING")
                break

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
    crnn.eval()
    transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")

    validation_data = DataLoadDf(dfs["validation"], encod_func, transform=transforms_valid, return_indexes=True)
    validation_dataloader = DataLoader(validation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                                       num_workers=cfg.num_workers)
    validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
    durations_validation = get_durations_df(cfg.validation)
    # Preds with only one value
    valid_predictions = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                        pooling_time_ratio, median_window=median_window,
                                        save_predictions=predicitons_fname)
    get_f1_sed_score(valid_predictions, validation_labels_df, verbose=True)
    f1, low_f1, high_f1 = bootstrap(valid_predictions, validation_labels_df, get_f1_sed_score)
    logger.info(f"F1 event_based: {f1}, +- {max(f1 - low_f1, high_f1 - f1)}")

    # ##########
    # Optional but recommended
    # ##########
    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    pred_thresh = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
                                  pooling_time_ratio, thresholds=list_thresholds, median_window=median_window,
                                  save_predictions=predicitons_fname)
    get_psds_ct(pred_thresh, validation_labels_df, durations_validation, verbose=True)
    psds_ct, low_p, high_p = bootstrap(pred_thresh, validation_labels_df, get_psds_ct,
                                       meta_df=durations_validation)
    logger.info(f"Psds ct: {psds_ct}, +- {max(psds_ct - low_p, high_p - psds_ct)}")
    df_res = pd.DataFrame([[f"{f1*100:.1f}~$\pm$~{max(f1 - low_f1, high_f1 - f1)*100:.1f}",
                            f"{psds_ct:.3f}~$\pm$~{max(psds_ct - low_p, high_p - psds_ct):.3f}"]],
                          columns=["f1", "psds_ct"])
    logger.info(df_res)
    df_res.to_csv(os.path.join("stored_data", "results.tsv"), index=False, sep="\t")
    # # ##########
    # # Optional but recommended
    # # ##########
    # # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    # n_thresholds = 50
    # # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    # list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    # pred_thresh = get_predictions(crnn, validation_dataloader, many_hot_encoder.decode_strong,
    #                               pooling_time_ratio, thresholds=list_thresholds, median_window=median_window,
    #                               save_predictions=predicitons_fname)
    # psds = compute_psds_from_operating_points(pred_thresh, validation_labels_df, durations_validation)
    # psds_score(psds, filename_roc_curves=os.path.join(saved_pred_dir, "figures/psds_roc.png"))
