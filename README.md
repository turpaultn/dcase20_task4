# DCASE2020 task4: Sound event detection in domestic environments using source separation

- Information about the DCASE 2020 challenge please visit the challenge [website].
- You can find discussion about the dcase challenge here: [dcase-discussions]. 
- This task follows [dcase2019-task4]. More info about 2019: 
[Turpault et al.][paper2019-description], [Serizel et al.][paper2019-eval] 

## Updates
- 9th March 2020: update `scripts` to get the recorded data in the download.
- 18th March 2020: update the `DESED_synth_dcase20_train_jams.tar`on [DESED_synthetic][synthetic_dataset] 
and comment reverb since we do not use it for the baseline.
- 24th March 2020: release baseline without sound-separation

## Dependencies

Python >= 3.6, pytorch >= 1.0, cudatoolkit=9.0, pandas >= 0.24.1, scipy >= 1.2.1, pysoundfile >= 0.10.2,
scaper >= 1.3.5, librosa >= 0.6.3, youtube-dl >= 2019.4.30, tqdm >= 4.31.1, ffmpeg >= 4.1, 
dcase_util >= 0.2.5, sed-eval >= 0.2.1, psds-eval >= 0.0.1, desed >= 1.1.7

A simplified installation procedure example is provided below for python 3.6 based Anconda distribution for Linux based system:
1. [install Ananconda][anaconda_download]
2. launch `conda_create_environment.sh` (recommended line by line)

## Baseline

This year, a **sound separation** model is used: see [sound-separation] folder which is the [fuss_repo] integrated as a 
git subtree.

### Source separation model

More info in [Original FUSS model repo][fuss-repo-model].



### SED model

More info in the [baseline] folder.

### Results

System performance are reported in term of event-based F-scores [[1]] 
with a 200ms collar on onsets and a 200ms / 20% of the events length collar on offsets. 

Additionally, the PSDS [[2]] performance are reported. 

<table class="table table-striped">
 <thead>
 <tr>
 <td></td>
 <td colspan="1">Baseline without sound separation</td>
  <td colspan="1">Baseline with sound separation</td>
 </tr>
 </thead>
 <tbody>
 <tr>
 <td></td>
 <td> Validation </td>
 </tr>
 <tr>
 <td><strong>Event-based</strong></td>
 <td><strong> 33.05 %</strong></td>
 </tr>
 <tr>
 <td>PSDS </td>
 <td> 0.403 </td>
 </tr>
 <tr>
 <td>PSDS cross-trigger</td>
 <td> 0.234 </td>
 </tr>
 <tr>
 <td>PSDS macro</td>
 <td> 0.199 </td>
 </tr>
 </tbody>
 </table>

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

## Dataset

### Description
- The **sound event detection** dataset is using [desed] dataset.
- To compute the reverberated data and the separated sources, we use [fuss_repo] 
(included as `sound-separation/` here (using subtree))
	- To compute the reverberated sounds, we use [fuss] rir_data and 
	`sound-separation/datasets/fuss/reverberate_and_mix.py`
	- To compute **sound separation**, we use [fuss] baseline model and 
	`sound-separation/models/dcase2020_fuss_baseline/inference.py`

### Base dataset
The dataset for sound event detection of DCASE2020 task 4 is composed of:
- Train:
	- *weak *(DESED, recorded)*
	- *unlabel_in_domain *(DESED, recorded)*
	- synthetic soundbank *(DESED, synthetic)*
- *Validation (DESED, recorded):
	- test2018
	- eval2018


### Pre-computed synthetic data used to train baseline
- Train:
	- synthetic20/soundscapes [2584 files] (DESED) --> base files, not used to train baseline
	- *synthetic20/separated_sources [2584 files] (DESED) --> base files, not used to train baseline
	- *weak_ss/separated_sources [1578 folders] (uses [fuss] baseline_model and [fuss_scripts])
	- *unlabel_in_domain_ss/separated_sources [14 412 folders] (uses [fuss] baseline_model and [fuss_scripts])
- Validation
	- *validation_ss/separated_sources [1168 files] (uses [fuss] baseline_model and [fuss_scripts])

* Only used in baseline with sound separation

*Note: the reverberated data are not computed for the baseline*

## Scripts

In the [`scripts/`](scripts) folder, you can find the different steps to generate:
- Synthetic soundscapes
- Reverberated synthetic data (Not used in the baseline)
- Separated sources of recorded and synthetic mixtures 

	
### DESED Dataset
**It is likely that you'll have download issues with the real recordings.
At the end of the download, please send a mail with the TSV files
created in the `missing_files` directory.** (in priority to Nicolas Turpault and Romain Serizel)

However, if none of the audio files have been downloaded, it is probably due to an internet, proxy problem.

See [Desed repo][desed] or [Desed_website][desed_website] for more info.

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

The minimum length for an event is 250ms. The minimum duration of the pause between two events from the same class is 150ms. When the silence between two consecutive events from the same class was less than 150ms the events have been merged to a single event.
The strong annotations are provided in a tab separated csv file (.tsv) under the following format:

```
[filename (string)][tab][event onset time in seconds (float)][tab][event offset time in seconds (float)][tab][event_label (strings)]
```
For example:

```
YOTsn73eqbfc_10.000_20.000.wav	0.163	0.665	Alarm_bell_ringing
```

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
If you have any contact feel free to contact [Nicolas](mailto:nicolas.turpault@inria.fr) 
(and [Romain](mailto:romain.serizel@loria.fr) )

## References

- [[1]] A. Mesaros, T. Heittola, & T. Virtanen, "Metrics for polyphonic sound event detection", 
Applied Sciences, 6(6):162, 2016
- [[2]] C. Bilen, G. Ferroni, F. Tuveri, J. Azcarreta, S. Krstulovic, 
A Framework for the Robust Evaluation of Sound Event Detection.

[1]: http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Delphin_15.pdf
[2]: https://arxiv.org/pdf/1910.08440.pdf

[anaconda_download]: https://www.anaconda.com/download/
[Audioset]: https://research.google.com/audioset/index.html
[dcase2019-task4]: http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments
[dcase18_task4]: http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection
[dcase-discussions]: https://groups.google.com/forum/#!forum/dcase-discussions
[dcase_website]: http://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments
[desed]: https://github.com/turpaultn/DESED
[desed_website]: https://project.inria.fr/desed/dcase-challenge/dcase-2020-task-4/
[evaluation_dataset]: https://zenodo.org/record/3588172
[FSD]: https://datasets.freesound.org/fsd/
[fuss]: https://zenodo.org/record/3694384/
[fuss_repo]: https://github.com/google-research/sound-separation
[fuss-repo-model]: https://github.com/google-research/sound-separation/tree/master/models/dcase2020_fuss_baseline
[fuss_scripts]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[paper2019-description]: https://hal.inria.fr/hal-02160855
[paper2019-eval]: https://hal.inria.fr/hal-02355573
[paper-psds]: https://arxiv.org/pdf/1910.08440.pdf
[Scaper]: https://github.com/justinsalamon/scaper
[synthetic_dataset]: https://zenodo.org/record/3702397
[website]: http://dcase.community/challenge2020/

[sound-separation]: ./sound-separation
[baseline]: ./baseline