import logging
import math
import os
import pandas as pd

workspace = ".."
# DESED Paths
weak = os.path.join(workspace, 'dataset/metadata/train/weak.tsv')
unlabel = os.path.join(workspace, 'dataset/metadata/train/unlabel_in_domain.tsv')
synthetic = os.path.join(workspace, 'dataset/metadata/train/synthetic20_reverb/soundscapes.tsv')
validation = os.path.join(workspace, 'dataset/metadata/validation/validation.tsv')
test2018 = os.path.join(workspace, 'dataset/metadata/validation/test_dcase2018.tsv')
eval2018 = os.path.join(workspace, 'dataset/metadata/validation/eval_dcase2018.tsv')
eval_desed = os.path.join(workspace, "dataset/metadata/eval/public.tsv")

# Useful because not just metadata replaced by audio, there are subsets (eval2018, test2018) so we specify the audio
audio_validation_dir = os.path.join(workspace, 'dataset/audio/validation')

## Separated data
weak_ss = os.path.join(workspace, 'dataset/audio/train/weak_ss/ss_computed')
unlabel_ss = os.path.join(workspace, 'dataset/audio/train/unlabel_in_domain_ss/ss_computed')
synthetic_ss = os.path.join(workspace, 'dataset/audio/train/synthetic20_reverb/ss_computed')
validation_ss = os.path.join(workspace, 'dataset/audio/validation_ss/ss_computed')
# test2018_ss = os.path.join(workspace, 'dataset/audio/validation/test_dcase2018_ss/ss_computed')
# eval2018_ss = os.path.join(workspace, 'dataset/audio/validation/eval_dcase2018_ss/ss_computed')
eval_desed_ss = os.path.join(workspace, "dataset/audio/eval/public_ss/ss_computed")


normalization_on = "per_band"
normalization_type = "min-max"

ref_db = -55
# config
# prepare_data
sample_rate = 16000
n_window = 2048
hop_length = 511
n_mels = 64
max_len_seconds = 10.
max_frames = math.ceil(max_len_seconds * sample_rate / hop_length)

f_min = 0.
f_max = 8000.

lr = 0.0001
initial_lr = 0.
beta1_before_rampdown = 0.9
beta1_after_rampdown = 0.5
beta2_during_rampdup = 0.99
beta2_after_rampup = 0.999
weight_decay_during_rampup = 0.99
weight_decay_after_rampup = 0.999

max_consistency_cost = 2
max_learning_rate = 0.001

median_window = 5

# Main
num_workers = 12
batch_size = 24
n_epoch = 100

checkpoint_epochs = 1

save_best = True

file_path = os.path.abspath(os.path.dirname(__file__))
classes = pd.read_csv(os.path.join(file_path, validation), sep="\t").event_label.dropna().sort_values().unique()


# Logger
terminal_level = logging.DEBUG
