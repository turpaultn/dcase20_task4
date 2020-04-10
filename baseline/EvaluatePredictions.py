import glob
import os.path as osp
import pandas as pd

from evaluation_measures import psds_score, compute_psds_from_operating_points, compute_metrics
from utilities.utils import generate_tsv_wav_durations

if __name__ == '__main__':
    groundtruth_path = "../dataset/metadata/validation/validation.tsv"
    durations_path = "../dataset/metadata/validation/validation_durations.tsv"
    # If durations do not exists, audio dir is needed
    groundtruth_audio_path = "../dataset/audio/validation"
    base_prediction_path = "stored_data/MeanTeacher_with_synthetic/predictions/baseline_validation"

    groundtruth = pd.read_csv(groundtruth_path, sep="\t")
    if osp.exists(durations_path):
        meta_dur_df = pd.read_csv(durations_path, sep='\t')
    else:
        meta_dur_df = generate_tsv_wav_durations(groundtruth_audio_path, durations_path)
    
    # Evaluate a single prediction
    single_predictions = pd.read_csv(base_prediction_path + ".tsv", sep="\t")
    compute_metrics(single_predictions, groundtruth, meta_dur_df)

    # Evaluate predictions with multiple thresholds (better). Need a list of predictions.
    prediction_list_path = glob.glob(osp.join(base_prediction_path, "*.tsv"))
    list_predictions = []
    for fname in prediction_list_path:
        pred_df = pd.read_csv(fname, sep="\t")
        list_predictions.append(pred_df)
    psds = compute_psds_from_operating_points(list_predictions, groundtruth, meta_dur_df)
    psds_score(psds, filename_roc_curves=osp.join(base_prediction_path, "figures/psds_roc.png"))
