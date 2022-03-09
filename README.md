
## Couple Learning for SED

- This repository provides the data and source code for sound event detection (SED) task.
- The improvement of the Couple Learning method is verified on the basis of the dcase20-task4 baseline.
- Information about Couple Learning please visit [paper: Couple Learning for semi-supervised sound event detection].


### Couple Learning model

More info in the [PLG-MT_run] folder.

### Reproducing the results
See [PLG-MT_run] folder.

## Dependencies

Python >= 3.6, pytorch >= 1.0, cudatoolkit>=9.0, pandas >= 0.24.1, scipy >= 1.2.1, pysoundfile >= 0.10.2,
scaper >= 1.3.5, librosa >= 0.6.3, youtube-dl >= 2019.4.30, tqdm >= 4.31.1, ffmpeg >= 4.1, 
dcase_util >= 0.2.5, sed-eval >= 0.2.1, psds-eval >= 0.1.0, desed >= 1.3.0

A simplified installation procedure example is provided below for python 3.6 based Anconda distribution 
for Linux based system:
1. [install Ananconda][anaconda_download]
2. launch `conda_create_environment.sh` (recommended line by line)

## Dataset

All the scripts to get the data (soundbank, generated, separated) are in the `scripts` folder 
and they use python files from `data_generation` folder.

### Scripts to generate the dataset

In the [`scripts/`][scripts] folder, you can find the different steps to:
- Download recorded data and synthetic material.
- Generate synthetic soundscapes
- Reverberate synthetic data (Not used in the baseline)
- Separate sources of recorded and synthetic mixtures 


**It is likely that you'll have download issues with the real recordings.
At the end of the download, please send a mail with the TSV files
created in the `missing_files` directory.** 

However, if none of the audio files have been downloaded, it is probably due to an internet, proxy problem.
See [Desed repo][desed] or [Desed_website][desed_website] for more info.


#### Base dataset
The dataset for sound event detection of DCASE2020 task 4 is composed of:
- Train:
	- *weak *(DESED, recorded, 1 578 files)*
	- *unlabel_in_domain *(DESED, recorded, 14 412 files)*
	- synthetic soundbank *(DESED, synthetic, 2060 background (SINS only) + 1006 foreground files)*
- *Validation (DESED, recorded, 1 168 files):
	- test2018 (288 files)
	- eval2018 (880 files)


#### Baselines dataset
##### SED baseline
- Train:
	- weak
	- unlabel_in_domain
	- synthetic20/soundscapes (separated in train/valid-80%/20%)
- Validation:
	- validation
	
-----

[github]: https://github.com/turpaultn/dcase20_task4
[PLG-MT_run]: ./PLG-MT_run
[anaconda_download]: https://www.anaconda.com/download/
[Audioset]: https://research.google.com/audioset/index.html
[dcase2019-task4]: http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments
[dcase18_task4]: http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection
[dcase-discussions]: https://groups.google.com/forum/#!forum/dcase-discussions
[dcase_website]: http://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments
[desed]: https://github.com/turpaultn/DESED
[desed_website]: https://project.inria.fr/desed/dcase-challenge/dcase-2020-task-4/
[evaluation_dataset]: https://doi.org/10.5281/zenodo.3571049
[eval_zenodo_repo]: https://doi.org/10.5281/zenodo.3866363
[FSD]: https://datasets.freesound.org/fsd/
[fuss]: https://doi.org/10.5281/zenodo.3694383
[fuss_repo]: https://github.com/google-research/sound-separation
[fuss-repo-model]: https://github.com/google-research/sound-separation/tree/master/models/dcase2020_fuss_baseline
[fuss_scripts]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[paper2019-description]: https://hal.inria.fr/hal-02160855
[paper2019-eval]: https://hal.inria.fr/hal-02355573
[paper-psds]: https://arxiv.org/pdf/1910.08440.pdf
[Scaper]: https://github.com/justinsalamon/scaper
[sed_paper]: https://hal.inria.fr/hal-02891665
[ss_sed_paper]: https://hal.inria.fr/hal-02891700
[synthetic_dataset]: https://doi.org/10.5281/zenodo.3550598
[submission_page]: http://dcase.community/challenge2020/submission
[website]: http://dcase.community/challenge2020/
[paper: Couple Learning for semi-supervised sound event detection]:https://arxiv.org/abs/2110.05809

[sed_roc_0_0_100]: figures/sed_baseline/psds_roc_0_0_100.png
[sed_roc_1_0_100]: figures/sed_baseline/psds_roc_1_0_100.png
[sed_roc_0_1_100]: figures/sed_baseline/psds_roc_0_1_100.png
[sed_ss_roc_0_0_100]: figures/sed_ss_baseline/psds_roc_0_0_100.png
[sed_ss_roc_1_0_100]: figures/sed_ss_baseline/psds_roc_1_0_100.png
[sed_ss_roc_0_1_100]: figures/sed_ss_baseline/psds_roc_0_1_100.png

[scripts]: ./scripts
[sound-separation]: ./sound-separation
[baseline]: ./baseline
