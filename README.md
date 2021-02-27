# DCASE2020 task4: Sound event detection in domestic environments using source separation

- Information about the DCASE 2020 challenge please visit the challenge [website].
- You can find discussion about the dcase challenge here: [dcase-discussions]. 
- This task follows [dcase2019-task4]. More info about 2019: 
[Turpault et al.][paper2019-description], [Serizel et al.][paper2019-eval] 
- Papers associated: [Sound separation + SED paper][ss_sed_paper] and [SED paper][sed_paper]
- More information about sound separation process can be found in Google repo: [FUSS][fuss-repo-model]

## Updates
- **`Important update`** 8th September 2020: Code associated to papers [ss_sed_paper] and [sed_paper] 
available in the branch `papers_code`
- 9th March 2020: update `scripts` to get the recorded data in the download.
- 18th March 2020: update the `DESED_synth_dcase20_train_jams.tar`on [DESED_synthetic][synthetic_dataset] 
and comment reverb since we do not use it for the baseline.
- 24th March 2020: release baseline without sound-separation
- 8th April 2020: update baseline without sound-separation. (common baseline, late integration).
Rearrange the scripts download. Final metrics done. Test model on multiple operating points.
Test model for sound-separation (late integration).

## Dependencies

Python >= 3.6, pytorch >= 1.0, cudatoolkit>=9.0, pandas >= 0.24.1, scipy >= 1.2.1, pysoundfile >= 0.10.2,
scaper >= 1.3.5, librosa >= 0.6.3, youtube-dl >= 2019.4.30, tqdm >= 4.31.1, ffmpeg >= 4.1, 
dcase_util >= 0.2.5, sed-eval >= 0.2.1, psds-eval >= 0.1.0, desed >= 1.3.0

A simplified installation procedure example is provided below for python 3.6 based Anconda distribution 
for Linux based system:
1. [install Ananconda][anaconda_download]
2. launch `conda_create_environment.sh` (recommended line by line)

## Papers code

If you are on this repo and searching for the code assoctiated to:
- [*Training sound event detection on a heterogeneous dataset*][sed_paper]
- [*Improving sound event detection in domestic environments using sound separation*][ss_sed_paper]

Please go to the branch `papers_code`.

## Submission

Please check the [submission_page].

The evaluation data is on this [eval_zenodo_repo].

Before doing your submission, please check your submission folder for task 4 with the dedicated scripts:
- `python validate_submissions.py -i <path to task 4 submission folder>`

## Baseline

This year, a **sound separation** model is used: see [sound-separation] folder which is the [fuss_repo] integrated as a 
git subtree.

### Sound separation model

More info in [Original FUSS model repo][fuss-repo-model].

### SED model

More info in the [baseline] folder.

### Combination of sound separation (SS) and sound event detection (SED)

The baseline to combine SS and SED is a late integration.

The sound separation baseline has been trained using 3 sources, so it returns: 
- DESED background
- DESED foreground
- FUSS mixture

In our case, we use only the ouput of the second source.

To get the predictions of the combination of SED and SS we do as follow:
- Get the output (not binarized with threshold) of the validation soundscapes (usual SED)
- Get the output (not binarized with threshold) of the DESED foreground source from SS model.
- Take the average of both outputs.
- Apply thresholds (different for F-scores and psds)
- Apply median filtering (0.45s)

### Results

System performance are reported in term of event-based F-scores [[1]] 
with a 200ms collar on onsets and a 200ms / 20% of the events length collar on offsets.

Additionally, the PSDS [[2]] performance are reported. 

*F-scores are computed using a single operating point (threshold=0.5) 
while other PSDS values are computed using 50 operating points (linear from 0.01 to 0.99).*

- Sound event detection baseline:

|         | Macro F-score Event-based | PSDS macro F-score | PSDS | PSDS cross-trigger | PSDS macro
----------|--------------------------:|-------------------:|-----:|-------------------:|----------:
Validation| **34.8 %**                | **60.0%**          | 0.610| 0.524              | 0.433


- Sound event detection + sound separation baseline

|         | Macro F-score Event-based | PSDS macro F-score | PSDS | PSDS cross-trigger | PSDS macro
----------|--------------------------:|-------------------:|-----:|-------------------:|----------:
Validation| **35.6 %**                | **60.5%**          | 0.626| 0.546              | 0.449
 

**Validation roc curves**

|                       | SED baseline         | SED + SS baseline | 
:-----------------------|:--------------------:|:------------------------:
psds roc curve          |![sed_roc_0_0_100]    | ![sed_ss_roc_0_0_100]    
psds cross-trigger curve|![sed_roc_1_0_100]    | ![sed_ss_roc_1_0_100]  
psds macro curve        | ![sed_roc_0_1_100]   |![sed_ss_roc_0_1_100]

Please refer to the PSDS paper [[2]] for more information about it.
The parameters used for psds performances are:
- Detection Tolerance parameter (dtc): 0.5
- Ground Truth intersection parameter (gtc): 0.5
- Cross-Trigger Tolerance parameter (cttc): 0.3
- maximum False Positive rate (e_max): 100

The difference between the 3 performances reported:

|                       | alpha_ct  | alpha_st  |
|-----------------------|----------:|----------:|
| PSDS                  | 0         | 0         |
| PSDS cross-trigger    | 1         | 0         |
| PSDS macro            | 0         | 1         |

alpha_ct is the cost of cross-trigger, alpha_st is the cost of instability across classes.

### Reproducing the results
See [baseline] folder.

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
created in the `missing_files` directory.** ([to Nicolas Turpault and Romain Serizel](#contact)).

However, if none of the audio files have been downloaded, it is probably due to an internet, proxy problem.
See [Desed repo][desed] or [Desed_website][desed_website] for more info.

### Description
- The **sound event detection** dataset is using [desed] dataset.
- To compute the separated sources, we use [fuss_repo] (included as `sound-separation/` here (using subtree))
 	- Specifically, we use [fuss] baseline model and 
	`sound-separation/models/dcase2020_fuss_baseline/inference.py`

#### Base dataset
The dataset for sound event detection of DCASE2020 task 4 is composed of:
- Train:
	- *weak *(DESED, recorded, 1 578 files)*
	- *unlabel_in_domain *(DESED, recorded, 14 412 files)*
	- synthetic soundbank *(DESED, synthetic, 2060 background (SINS only) + 1006 foreground files)*
- *Validation (DESED, recorded, 1 168 files):
	- test2018 (288 files)
	- eval2018 (880 files)

#### Pre-computed data used to train baseline
- Train:
	- synthetic20/soundscapes [2584 files] (DESED)
	- synthetic20/separated_sources [2584 files] (DESED)
	- weak_ss/separated_sources [1578 folders] (uses [fuss] baseline_model and [fuss_scripts])
	- unlabel_in_domain_ss/separated_sources [14 412 folders] (uses [fuss] baseline_model and [fuss_scripts])
- Validation
	- validation_ss/separated_sources [1168 files] (uses [fuss] baseline_model and [fuss_scripts])

*Note: the reverberated data (see [scripts](#scripts-to-generate-the-dataset)) are not computed for the baseline*

#### Baselines dataset
##### SED baseline
- Train:
	- weak
	- unlabel_in_domain
	- synthetic20/soundscapes (separated in train/valid-80%/20%)
- Validation:
	- validation

##### SED + SS baseline
- No new training involved
- Validation:
	- validation + validation_ss/separated_sources

### Annotation format

#### Weak annotations
The weak annotations have been verified manually for a small subset of the training set. 
The weak annotations are provided in a tab separated csv file (.tsv) under the following format:

```
[filename (string)][tab][event_labels (strings)]
```
For example:
```
Y-BJNMHMZDcU_50.000_60.000.wav	Alarm_bell_ringing,Dog
```

#### Strong annotations
Synthetic subset and validation set have strong annotations.

The minimum length for an event is 250ms. The minimum duration of the pause between two events from the same class 
is 150ms. 
When the silence between two consecutive events from the same class was less than 150ms the events have been merged 
to a single event.
The strong annotations are provided in a tab separated csv file (.tsv) under the following format:

```
[filename (string)][tab][event onset time in seconds (float)][tab][event offset time in seconds (float)][tab][event_label (strings)]
```
For example:

```
YOTsn73eqbfc_10.000_20.000.wav	0.163	0.665	Alarm_bell_ringing
```

## A word on sound separation dataset 
#### [Free Universal Sound Separation (FUSS) Dataset][fuss]
The free universal sound separation (FUSS) dataset [3] contains mixtures of arbitrary sources of different types 
for use in training sound separation models. Each 10 second mixture contains between 1 and 4 sounds. 

The source clips for the mixtures are from a prerelease of FSD50k [4], [5], which is composed of Freesound content 
annotated with labels from the AudioSet Ontology. Using the FSD50k labels, the sound source files have been screened 
such that they likely only contain a single type of sound. Labels are not provided for these sound source files, 
and are not considered part of the challenge, although they will become available when FSD50k is released.


Train:
- 20000 mixtures

Validation:
- 1000 mixtures

## Authors

|Author                 | Affiliation               |
|-----------------------|---------------            |
|Nicolas Turpault       | INRIA                     |
|Romain Serizel         | University of Lorraine    |
|Scott Wisdom           | Google Research           |
|John R. Hershey        | Google Research           |
|Hakan Erdogan          | Google Research           |
|Justin Salamon         | Adobe Research            |
|Dan Ellis              | Google Research           |
|Prem Seetharaman       | Northwestern University   |

## Contact
If you have any problem feel free to contact [Nicolas](mailto:nicolas.turpault@inria.fr) 
(and [Romain](mailto:romain.serizel@loria.fr) )

## References

- [[1]] A. Mesaros, T. Heittola, & T. Virtanen, "Metrics for polyphonic sound event detection", 
Applied Sciences, 6(6):162, 2016
- [[2]] C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta, S. Krstulovic, 
A Framework for the Robust Evaluation of Sound Event Detection.
- [[3]] Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, Romain Serizel, Nicolas Turpault, Eduardo Fonseca, Justin Salamon, Prem Seetharaman, and John R. Hershey. 
What's all the fuss about free universal sound separation data? In preparation. 2020.
- [[4]] E. Fonseca, J. Pons, X. Favory, F. Font, D. Bogdanov, A. Ferraro, S. Oramas, A. Porter, and X. Serra.
Freesound datasets: a platform for the creation of open audio datasets.
In Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR 2017), 486–493.
Suzhou, China, 2017.
- [[5]] F. Font, G. Roma, and X. Serra. Freesound technical demo.

In Proceedings of the 21st ACM international conference on Multimedia, 411–412. ACM, 2013.
[1]: http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Delphin_15.pdf
[2]: https://arxiv.org/pdf/1910.08440.pdf
[3]: ./
[4]: https://repositori.upf.edu/bitstream/handle/10230/33299/fonseca_ismir17_freesound.pdf
[5]: mtg.upf.edu/system/files/publications/Font-Roma-Serra-ACMM-2013.pdf

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

[sed_roc_0_0_100]: figures/sed_baseline/psds_roc_0_0_100.png
[sed_roc_1_0_100]: figures/sed_baseline/psds_roc_1_0_100.png
[sed_roc_0_1_100]: figures/sed_baseline/psds_roc_0_1_100.png
[sed_ss_roc_0_0_100]: figures/sed_ss_baseline/psds_roc_0_0_100.png
[sed_ss_roc_1_0_100]: figures/sed_ss_baseline/psds_roc_1_0_100.png
[sed_ss_roc_0_1_100]: figures/sed_ss_baseline/psds_roc_0_1_100.png

[scripts]: ./scripts
[sound-separation]: ./sound-separation
[baseline]: ./baseline
