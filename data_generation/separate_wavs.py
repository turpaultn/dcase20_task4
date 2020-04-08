"""Use source separation baseline (in tensorflow) to separate audio files"""
import numpy as np
import soundfile as sf
import librosa
import os
import os.path as osp
import sys
import argparse


absolute_dir_path = os.path.abspath(os.path.dirname(__file__))
relative_path_ss_repo = osp.join(absolute_dir_path, "..")
base_dir_repo = osp.abspath(relative_path_ss_repo)
sys.path.append(osp.join(base_dir_repo, "sound-separation", "models", "dcase2020_fuss_baseline"))


import inference


def read_audio(path, target_fs=None):
    """ Read a wav file
    Args:
        path: str, path of the audio file
        target_fs: int, (Default value = None) sampling rate of the returned audio file, if not specified, the sampling
            rate of the audio file is taken

    Returns:
        tuple
        (numpy.array, sampling rate), array containing the audio at the sampling rate given

    """
    (audio, fs) = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def main(wav_files, model, out_folder, pattern_isolated_events="_events"):
    os.makedirs(out_folder, exist_ok=True)

    for cnt, file in enumerate(wav_files):
        if cnt % 500 == 0:
            print(file)
            print(f"{cnt}/{len(wav_files)}")
        waveform, sr = read_audio(file, 16000)
        separated_waveforms = model.separate(waveform)
        for cnt, sep_wav in enumerate(separated_waveforms):
            out_sep_fodler = osp.join(out_folder, osp.splitext(osp.basename(file))[0] + pattern_isolated_events)
            os.makedirs(out_sep_fodler, exist_ok=True)
            out_file = osp.join(out_sep_fodler, f"{cnt}.wav")
            sf.write(out_file, sep_wav, samplerate=16000)


if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-a", '--audio_path', type=str, required=True)
    parser.add_argument("-o", '--output_folder', type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-i", "--inference", type=str, required=True)
    f_args = parser.parse_args()

    wav_list = glob.glob(osp.join(f_args.audio_path, "*.wav"))
    if len(wav_list) == 0:
        wav_list = glob.glob(osp.join(f_args.audio_path, "soundscapes", "*.wav"))
    if len(wav_list) == 0:
        raise IndexError(f"Empty wav_list, you need to give a valid audio_path. Not valid: {f_args.audio_path}")
    # model_dir = f_args.model_dir
    # checkpoint_path = osp.join(model_dir, 'baseline_model')
    # metagraph_path = osp.join(model_dir, 'baseline_inference.meta')
    checkpoint_path = f_args.checkpoint
    metagraph_path = f_args.inference
    ss_model = inference.SeparationModel(checkpoint_path, metagraph_path)

    main(wav_list, ss_model, f_args.output_folder)
