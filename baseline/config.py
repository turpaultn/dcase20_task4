import logging
import math
import os
import pandas as pd

workspace = ".."
# DESED Paths
weak = os.path.join(workspace, 'dataset/metadata/train/weak.tsv')
unlabel = os.path.join(workspace, 'dataset/metadata/train/unlabel_in_domain.tsv')
synthetic = os.path.join(workspace, 'dataset/metadata/train/synthetic20_pitch/soundscapes.tsv')
validation = os.path.join(workspace, 'dataset/metadata/validation/validation.tsv')
test2018 = os.path.join(workspace, 'dataset/metadata/validation/test_dcase2018.tsv')
eval2018 = os.path.join(workspace, 'dataset/metadata/validation/eval_dcase2018.tsv')
eval_desed = os.path.join(workspace, "dataset/metadata/eval/public.tsv")

# Useful because not just metadata replaced by audio, there are subsets (eval2018, test2018) so we specify the audio
audio_validation_dir = os.path.join(workspace, 'dataset/audio/validation')

## Separated data
weak_ss = os.path.join(workspace, 'dataset/audio/train/weak_ss/separated_sources')
unlabel_ss = os.path.join(workspace, 'dataset/audio/train/unlabel_in_domain_ss/separated_sources')
synthetic_ss = os.path.join(workspace, 'dataset/audio/train/synthetic20_pitch/separated_sources')
validation_ss = os.path.join(workspace, 'dataset/audio/validation_ss/separated_sources')
eval_desed_ss = os.path.join(workspace, "dataset/audio/eval/public_ss/separated_sources")

scaler_type = "dataset"
scale_peraudio_on = "global"
scale_peraudio_type = "min-max"

# Data preparation
ref_db = -55
sample_rate = 16000
max_len_seconds = 10.
# features
n_window = 2048
hop_size = 255
n_mels = 128
max_frames = math.ceil(max_len_seconds * sample_rate / hop_size)

mel_f_min = 0.
mel_f_max = 8000.

# Model
max_consistency_cost = 2

# Training
in_memory = True
num_workers = 12
batch_size = 24
# Todo, reput as normal
import torch
if torch.cuda.is_available():
    n_epoch = 200
    n_epoch_rampup = 50
else:
    n_epoch = 2
    n_epoch_rampup = 1

checkpoint_epochs = 1
save_best = True
early_stopping = 10
es_init_wait = 50

adjust_lr = False
max_learning_rate = 0.001

default_learning_rate = 0.001

# Post processing
median_window_s = 0.45

# Classes
file_path = os.path.abspath(os.path.dirname(__file__))
classes = pd.read_csv(os.path.join(file_path, validation), sep="\t").event_label.dropna().sort_values().unique()


# Logger
terminal_level = logging.DEBUG
