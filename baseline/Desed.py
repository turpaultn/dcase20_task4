# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2020, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

from __future__ import print_function
import numpy as np
import os
import os.path as osp
import librosa
import time
import pandas as pd

import config as cfg
from desed.download_real import download
from utilities.Logger import create_logger
from utilities.utils import read_audio, meta_path_to_audio_dir, generate_tsv_from_isolated_events

logger = create_logger(__name__)


class DESED:
    """DCASE 2020 task 4 dataset, uses DESED dataset
    Data are organized in `audio/` and corresponding `metadata/` folders.
    audio folder contains wav files, and metadata folder contains .tsv files.

    The organisation should always be the same in the audio and metadata folders. (See example)
    If there are multiple metadata files for a single audio files, add the name in the list of `merged_folders_name`.
    (See validation folder example). Be careful, it works only for one level of folder.

    tab separated value metadata files (.tsv) contains columns:
        - filename                                  (unlabeled data)
        - filename  event_labels                    (weakly labeled data)
        - filename  onset   offset  event_label     (strongly labeled data)

    Example:
    - dataset
        - metadata
            - train
                - weak.tsv              (will search filename in audio/train/weak)
            - validation
                - validation.tsv        (will search filename in audio/validation because `merged_folders_name=["validation"]`)
                - test_dcase2018.tsv    (will search filename in audio/validation because `merged_folders_name=["validation"]`)
                - eval_dcase2018.tsv    (will search filename in audio/validation because `merged_folders_name=["validation"]`)
            -eval
                - public.tsv            (will search filename in audio/eval/public)
        - audio
            - train
                - weak
            - validation
            - eval
                - public
                    - aaa.wav
                    - aaa_sources
                        - foreground
                            - s1.wav
                            - s2.wav
                        - background
                            - s3.wav
                    - bbb.wav

    Args:
        base_feature_dir: str, optional, base directory to store the features
        recompute_features: bool, optional, wether or not to recompute features
        compute_log: bool, optional, whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)

    Attributes:
        base_feature_dir: str, base directory to store the features
        recompute_features: bool, wether or not to recompute features
        compute_log: bool, whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)
        feature_dir : str, directory to store the features

    """
    def __init__(self, base_feature_dir="features", recompute_features=False, compute_log=True):

        self.recompute_features = recompute_features
        self.compute_log = compute_log

        feature_dir = osp.join(base_feature_dir, "sr" + str(cfg.sample_rate) + "_win" + str(cfg.n_window)
                                   + "_hop" + str(cfg.hop_length) + "_mels" + str(cfg.n_mels))
        if not self.compute_log:
            feature_dir += "_nolog"

        self.feature_dir = osp.join(feature_dir, "features")
        self.meta_feat_dir = osp.join(feature_dir, "metadata")
        # create folder if not exist
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.meta_feat_dir, exist_ok=True)

    def initialize_and_get_df(self, tsv_path, audio_dir=None, nb_files=None, download=True):
        """ Initialize the dataset, extract the features dataframes
        Args:
            tsv_path: str, tsv path in the initial dataset
            audio_dir: str, the path where to search the filename of the df
            nb_files: int, optional, the number of file to take in the dataframe if taking a small part of the dataset.
            download: bool, optional, whether or not to download the data from the internet (youtube).

        Returns:
            pd.DataFrame
            The dataframe containing the right features and labels
        """
        df_meta = self.get_df_from_meta(tsv_path, nb_files)
        logger.info("{} Total file number: {}".format(tsv_path, len(df_meta.filename.unique())))
        if audio_dir is None:
            audio_dir = meta_path_to_audio_dir(tsv_path)
        if download:
            # Get only one filename once
            filenames = df_meta.filename.drop_duplicates()
            if nb_files is not None:
                filenames = filenames.sample(nb_files)
            self.download(filenames, audio_dir)
        df_features = self.extract_features_from_df(df_meta, audio_dir)
        df_features.to_csv(osp.join(self.meta_feat_dir, osp.basename(tsv_path)), sep="\t", index=False)
        return df_features

    def initialize_ss(self, audio_dir, tsv_path, nb_files=None, pattern_ss="_events"):
        df = generate_tsv_from_isolated_events(audio_dir, tsv_path)
        df = self.get_subpart_data(df, nb_files, pattern_ss)
        df_features = self.extract_features_from_df(df, audio_dir)
        # Dirname because we assume that's the changing part
        df_features.to_csv(osp.join(self.meta_feat_dir, osp.basename(osp.dirname(tsv_path))), sep="\t", index=False)
        return df_features

    def get_feature_file(self, filename):
        """
        Get a feature file from a filename
        Args:
            filename:  str, name of the file to get the feature

        Returns:
            numpy.array
            containing the features computed previously
        """
        fname = osp.join(self.feature_dir, osp.splitext(filename)[0] + ".npy")
        data = np.load(fname)
        return data

    def calculate_mel_spec(self, audio):
        """
        Calculate a mal spectrogram from raw audio waveform
        Note: The parameters of the spectrograms are in the config.py file.
        Args:
            audio : numpy.array, raw waveform to compute the spectrogram

        Returns:
            numpy.array
            containing the mel spectrogram
        """
        # Compute spectrogram
        ham_win = np.hamming(cfg.n_window)

        spec = librosa.stft(
            audio,
            n_fft=cfg.n_window,
            hop_length=cfg.hop_length,
            window=ham_win,
            center=True,
            pad_mode='reflect'
        )

        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
            sr=cfg.sample_rate,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min, fmax=cfg.f_max,
            htk=False, norm=None)

        if self.compute_log:
            mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
        mel_spec = mel_spec.T
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def extract_features_from_df(self, df_meta, audio_dir):
        """Extract log mel spectrogram features.

        Args:
            df_meta : pd.DataFrame, containing at least column "filename" with name of the wav to compute features
            audio_dir: str, the path where to find the wav files specified by the dataframe
        """
        t1 = time.time()
        df_features = pd.DataFrame()
        for ind, wav_name in enumerate(df_meta.filename.unique()):
            if ind % 500 == 0:
                logger.debug(ind)
            wav_path = osp.join(audio_dir, wav_name)

            out_filename = osp.splitext(wav_name)[0] + ".npy"
            out_path = osp.join(self.feature_dir, out_filename)

            if not osp.exists(out_path):
                if not osp.isfile(wav_path):
                    logger.error("File %s is in the tsv file but the feature is not extracted because "
                                 "file do not exist!" % wav_path)
                    df_meta = df_meta.drop(df_meta[df_meta.filename == wav_name].index)
                else:
                    (audio, _) = read_audio(wav_path, cfg.sample_rate)
                    if audio.shape[0] == 0:
                        print("File %s is corrupted!" % wav_path)
                    else:
                        mel_spec = self.calculate_mel_spec(audio)
                        os.makedirs(osp.dirname(out_path), exist_ok=True)
                        np.save(out_path, mel_spec)

                    logger.debug("compute features time: %s" % (time.time() - t1))
            row_features = df_meta[df_meta.filename == wav_name]
            row_features.loc[:, "raw_filename"] = row_features["filename"]
            row_features.loc[:, "filename"] = out_path
            df_features = df_features.append(row_features, ignore_index=True)

        return df_features.reset_index(drop=True)

    @staticmethod
    def get_classes(list_dfs):
        """ Get the different classes of the dataset
        Returns:
            A list containing the classes
        """
        classes = []
        for df in list_dfs:
            if "event_label" in df.columns:
                classes.extend(df["event_label"].dropna().unique())  # dropna avoid the issue between string and float
            elif "event_labels" in df.columns:
                classes.extend(df.event_labels.str.split(',', expand=True).unstack().dropna().unique())
        return list(set(classes))

    @staticmethod
    def get_subpart_data(df, nb_files, pattern_ss=None):
        """Get a subpart of a dataframe (only the number of files specified), if ss, corresponds to the number of folder
        Args:
            meta_name : str, path of the tsv file to extract the df
            nb_files: int, the number of file to take in the dataframe if taking a small part of the dataset.
            pattern_ss: str, if nb_files is not None, the pattern is needed to get same ss than soundscapes
        Returns:
            dataframe
        """
        column = "filename"
        if not nb_files > len(df[column].unique()):
            if pattern_ss:
                filenames = df[column].apply(lambda x: x.split(pattern_ss)[0])
                filenames = filenames.drop_duplicates()
                # sort_values and random_state are used to have the same filenames each time (also for normal and ss)
                filenames_kept = filenames.sort_values().sample(nb_files, random_state=10)
                df_kept = df[df[column].apply(lambda x: x.split(pattern_ss)[0]).isin(filenames_kept)].reset_index(drop=True)

            else:
                filenames = df[column].drop_duplicates()
                # sort_values and random_state are used to have the same filenames each time (also for normal and ss)
                filenames_kept = filenames.sort_values().sample(nb_files, random_state=10)
                df_kept = df[df[column].isin(filenames_kept)].reset_index(drop=True)

            logger.debug("Taking subpart of the data, len : {}, df_len: {}".format(nb_files, len(df)))
        else:
            df_kept = df
        return df_kept

    @staticmethod
    def get_df_from_meta(meta_name, nb_files=None, pattern_ss=None):
        """
        Extract a pandas dataframe from a tsv file

        Args:
            meta_name : str, path of the tsv file to extract the df
            nb_files: int, the number of file to take in the dataframe if taking a small part of the dataset.
            pattern_ss: str, if nb_files is not None, the pattern is needed to get same ss than soundscapes
        Returns:
            dataframe
        """
        df = pd.read_csv(meta_name, header=0, sep="\t")
        if nb_files is not None:
            df = DESED.get_subpart_data(df, nb_files, pattern_ss=pattern_ss)
        return df

    @staticmethod
    def download(filenames, audio_dir, n_jobs=3, chunk_size=10):
        """
        Download files contained in a list of filenames

        Args:
            filenames: list or pd.Series, filenames of files to be downloaded ()
            audio_dir: str, the directory where the wav file should be downloaded (if not exist)
            nb_files: int, the number of files to use, if a subpart of the dataframe wanted.
            chunk_size: int, (Default value = 10) number of files to download in a chunk
            n_jobs : int, (Default value = 3) number of parallel jobs
        """
        download(filenames, audio_dir, n_jobs=n_jobs, chunk_size=chunk_size)
