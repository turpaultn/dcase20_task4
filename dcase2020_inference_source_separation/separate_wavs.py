"""Run inference for DCASE 2020 source separation model on random input."""

import numpy as np
import inference
import soundfile as sf
import librosa
import os
import os.path as osp
import argparse


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

def new_version():
    checkpoint_path = 'model2_model.ckpt-1901521'
    metagraph_path = 'model2_inference.meta'
    model = inference.SeparationModel(checkpoint_path,
                                      metagraph_path)

    num_samples = 1000
    num_sources = 4
    source_waveforms = 0.2 * np.random.rand(num_sources, num_samples)
    mixture_waveform = np.sum(source_waveforms, axis=0)

    # Separate with a trained model.
    separated_waveforms = model.separate(mixture_waveform)
    assert separated_waveforms.shape == (num_sources, num_samples)

    # Separate with an oracle binary mask.
    model_obm = inference.OracleBinaryMasking()
    separated_waveforms_obm = model_obm.separate(mixture_waveform,
                                                 source_waveforms)
    assert separated_waveforms_obm.shape == (num_sources, num_samples)

    print('Inference test completed successfully.')

def main(wav_files, out_folder, pattern_isolated_events="_events"):
    checkpoint_path = 'model_ckpt/model2_model.ckpt-1901521'
    metagraph_path = 'model_ckpt/model2_inference.meta'
    model = inference.SeparationModel(checkpoint_path, metagraph_path)
    os.makedirs(out_folder, exist_ok=True)

    for file in wav_files:
        print(file)
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
    parser.add_argument("-a", '--audio_path', type=str)
    parser.add_argument("-o", '--output_folder', type=str)
    f_args = parser.parse_args()
    # audio_path = "/Users/nturpaul/Documents/Seafile/DCASE/Desed/dataset/audio/eval/public"
    # out_folder = "/Users/nturpaul/Documents/code/dcase2020/dcase20_task4/dataset/audio/eval/generated/public"

    # audio_path = "/Users/nturpaul/Documents/Seafile/DCASE/Desed/dcase2019/dataset/audio/eval/fbsnr_30dB"
    # out_folder = "./eval/fbsnr_30dB"
    wav_files = glob.glob(osp.join(f_args.audio_path, "*.wav"))
    main(wav_files, f_args.output_folder)
