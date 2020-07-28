import argparse
import os

import librosa
import numpy as np
import os.path as osp
import soundfile as sf


def read_audio(path, target_fs=None):
    """ Read a wav file, and resample it if target_fs defined
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


if __name__ == '__main__':
    import glob
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-a", '--audio_path', type=str, required=True)
    parser.add_argument("-o", '--output_folder', type=str, required=True)
    f_args = parser.parse_args()

    wav_list = glob.glob(osp.join(f_args.audio_path, "*.wav"))
    if len(wav_list) == 0:
        wav_list = glob.glob(osp.join(f_args.audio_path, "soundscapes", "*.wav"))
    if len(wav_list) == 0:
        raise IndexError(f"Empty wav_list, you need to give a valid audio_path. Not valid: {f_args.audio_path}")

    os.makedirs(f_args.output_folder, exist_ok=True)

    for audio_path in wav_list:
        waveform, sr = read_audio(audio_path, 16000)
        out_sep_file = osp.join(f_args.output_folder, osp.basename(audio_path))
        sf.write(out_sep_file, waveform, samplerate=16000)
