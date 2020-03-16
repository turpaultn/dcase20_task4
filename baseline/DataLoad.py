# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################
import bisect
import glob
import itertools
import os

import numpy as np
import pandas as pd
import torch
import random
import warnings
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from utilities.Logger import create_logger

torch.manual_seed(0)
random.seed(0)
logger = create_logger(__name__)


class DataLoadDf(Dataset):
    """ Class derived from pytorch DESED
    Prepare the data to be use in a batch mode

    Args:
        df: pandas.DataFrame, the dataframe containing the set infromation (feat_filenames, labels),
            it should contain these columns :
            "feature_filename"
            "feature_filename", "event_labels"
            "feature_filename", "onset", "offset", "event_label"
        get_feature_file_func: function(), function which take a filename as input and return a feature file
        encode_function: function(), function which encode labels
        transform: function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, (Default value = False) whether or not to return indexes when use __getitem__

    Attributes:
        df: pandas.DataFrame, the dataframe containing the set infromation (feat_filenames, labels, ...)
        get_feature_file_func: function(), function which take a filename as input and return a feature file
        encode_function: function(), function which encode labels
        transform : function(), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, whether or not to return indexes when use __getitem__
    """
    def __init__(self, df, encode_function, transform=None,
                 return_indexes=False, in_memory=False):

        self.df = df
        self.encode_function = encode_function
        self.transform = transform
        self.return_indexes = return_indexes
        self.feat_filenames = df.feature_filename.drop_duplicates()
        self.filenames = df.filename.drop_duplicates()
        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def set_return_indexes(self, val):
        """ Set the value of self.return_indexes

        Args:
            val : bool, whether or not to return indexes when use __getitem__
        """
        self.return_indexes = val

    def get_feature_file_func(self, filename):
        """Get a feature file from a filename
        Args:
            filename:  str, name of the file to get the feature

        Returns:
            numpy.array
            containing the features computed previously
        """
        if not self.in_memory:
            data = np.load(filename).astype(np.float32)
        else:
            if self.features.get(filename) is None:
                data = np.load(filename).astype(np.float32)
                self.features[filename] = data
            else:
                data = self.features[filename]
        return data

    def __len__(self):
        """
        Returns:
            int
                Length of the object
        """
        length = len(self.feat_filenames)
        return length

    def get_sample(self, index):
        """From an index, get the features and the labels to create a sample

        Args:
            index: int, Index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array)

        """
        features = self.get_feature_file_func(self.feat_filenames.iloc[index])

        # event_labels means weak labels, event_label means strong labels
        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            if "event_labels" in self.df.columns:
                label = self.df.iloc[index]["event_labels"]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            else:
                cols = ["onset", "offset", "event_label"]
                label = self.df[self.df.filename == self.filenames.iloc[index]][cols]
                if label.empty:
                    label = []
        else:
            label = "empty"  # trick to have -1 for unlabeled data and concat them with labeled
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns))
        if index == 0:
            logger.debug("label to encode: {}".format(label))
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            y = self.encode_function(label)
        else:
            y = label
        sample = features, y
        return sample

    def __getitem__(self, index):
        """ Get a sample and transform it to be used in a ss_model, use the transformations

        Args:
            index : int, index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array) or
            Tuple containing the features, the labels and the index (numpy.array, numpy.array, int)

        """
        sample = self.get_sample(index)

        if self.transform:
            sample = self.transform(sample)

        if self.return_indexes:
            sample = (sample, index)

        return sample

    def set_transform(self, transform):
        """Set the transformations used on a sample

        Args:
            transform: function(), the new transformations
        """
        self.transform = transform

    def add_transform(self, transform):
        if type(self.transform) is not Compose:
            raise TypeError("To add transform, the transform should already be a compose of transforms")
        transforms = self.transform.add_transform(transform)
        return DataLoadDf(self.df, self.encode_function, transforms, self.return_indexes, self.in_memory)


class DataLoadDfSS(DataLoadDf):
    def __init__(self, df, encode_function, transform=None, return_indexes=False, in_memory=False,
                 ss_pattern="_events"):
        super(DataLoadDfSS, self).__init__(df, encode_function, transform, return_indexes, in_memory)
        self.ss_pattern = ss_pattern

    def get_feature_file_func(self, filename):
        if not self.in_memory or self.features.get(filename) is None:
            # Get the mixture
            arr_loaded_data = np.expand_dims(np.load(filename), 0)
            # Get feat_filenames of separated sources
            path_to_search = os.path.join(os.path.splitext(filename)[0] + self.ss_pattern, "*.npy")
            filenames_to_load = glob.glob(path_to_search)
            if len(filenames_to_load) == 0:
                warnings.warn(f"Unexpected empty list for filenames_to_load, check {path_to_search}")
            for fname in filenames_to_load:
                arr_loaded_data = np.concatenate((arr_loaded_data, np.expand_dims(np.load(fname), 0)))
            if self.in_memory:
                self.features[filename] = arr_loaded_data
        else:
            arr_loaded_data = self.features[filename]

        return arr_loaded_data


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms: list of ``Transform`` objects, list of transforms to compose.
        Example of transform: ToTensor()
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transform):
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class ConcatDataset(Dataset):
    """
    DESED to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets : sequence, list of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @property
    def cluster_indices(self):
        cluster_ind = []
        prec = 0
        for size in self.cumulative_sizes:
            cluster_ind.append(range(prec, size))
            prec = size
        return cluster_ind

    def __init__(self, datasets):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @property
    def df(self):
        df = self.datasets[0].df
        for dataset in self.datasets[1:]:
            df = pd.concat([df, dataset.df], axis=0, ignore_index=True, sort=False)
        return df


class Subset(DataLoadDf):
    """
    Subset of a dataset to be used when separating in multiple subsets

    Args:
        dataload_df: DataLoadDf or similar, dataset to be split
        indices: sequence, list of indices to keep in this subset
    """
    def __init__(self, dataload_df, indices):
        self.indices = indices
        self.df = dataload_df.df.loc[indices].reset_index(inplace=False, drop=True)

        super(Subset, self).__init__(self.df, dataload_df.encode_function,
                                     dataload_df.transform, dataload_df.return_indexes)

    def __getitem__(self, idx):
        return super(Subset, self).__getitem__(idx)


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Args:
        dataset: DESED, dataset to be split
        lengths: sequence, lengths of splits to be produced
    """
    # if ratio > 1:
    # 	raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.permutation(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(itertools.accumulate(lengths), lengths)]


def train_valid_split(dataset, validation_amount):
    valid_length = int(validation_amount * len(dataset))
    train_length = len(dataset) - valid_length

    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])
    return train_dataset, valid_dataset


class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size=None, shuffle=True):
        super(ClusterRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, "do not declare batch size in sampler " \
                                                         "if data source already got one"
            self.batch_sizes = [batch_size for _ in self.data_source.cluster_indices]
        else:
            self.batch_sizes = self.data_source.batch_sizes
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for j, cluster_indices in enumerate(self.data_source.cluster_indices):
            batches = [
                cluster_indices[i:i + self.batch_sizes[j]] for i in range(0, len(cluster_indices), self.batch_sizes[j])
            ]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

            # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class MultiStreamBatchSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_sizes, shuffle=True):
        super(MultiStreamBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_sizes = batch_sizes
        l_bs = len(batch_sizes)
        nb_dataset = len(self.data_source.cluster_indices)
        assert l_bs == nb_dataset, "batch_sizes must be the same length as the number of datasets in " \
                                   "the source {} != {}".format(l_bs, nb_dataset)
        self.shuffle = shuffle

    def __iter__(self):
        indices = self.data_source.cluster_indices
        if self.shuffle:
            for i in range(len(self.batch_sizes)):
                indices[i] = np.random.permutation(indices[i])
        iterators = []
        for i in range(len(self.batch_sizes)):
            iterators.append(grouper(indices[i], self.batch_sizes[i]))

        return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))

    def __len__(self):
        val = np.inf
        for i in range(len(self.batch_sizes)):
            val = min(val, len(self.data_source.cluster_indices[i]) // self.batch_sizes[i])
        return val


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n

    return zip(*args)
