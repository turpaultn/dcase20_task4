# DCASE2020 task4: Sound event detection in domestic environments using source separation

- Information about the DCASE 2020 challenge please visit the challenge [website].
- You can find discussion about the dcase challenge here: [dcase-discussions]. 
- This task follows [dcase2019-task4]. More info about 2019: 
[Turpault et al.][paper2019-description], [Serizel et al.][paper2019-eval] 

## Updates
- 7th March 2020: update the `DESED_synth_dcase20_train_jams.tar`on [DESED_synthetic][synthetic_dataset]

## Dependencies

Python >= 3.6, pytorch >= 1.0, cudatoolkit=9.0, pandas >= 0.24.1, scipy >= 1.2.1, pysoundfile >= 0.10.2,
scaper >= 1.3.5, librosa >= 0.6.3, youtube-dl >= 2019.4.30, tqdm >= 4.31.1, ffmpeg >= 4.1, 
dcase_util >= 0.2.5, sed-eval >= 0.2.1

A simplified installation procedure example is provide below for python 3.6 based Anconda distribution for Linux based system:
1. [install Ananconda][anaconda_download]
2. launch `conda_create_environment.sh` (recommended line by line)

## Dataset
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
	- synthetic20/soundscapes [2595 files] (DESED) --> base files, not used to train baseline
	- *synthetic20_reverb/soundscapes [2595 files] (uses [fuss] rir_data and [fuss_scripts])
	- **synthetic20_reverb/ss_computed [2595 folders] (uses [fuss] baseline_model and [fuss_scripts])
	- **weak_ss/ss_computed [1578 folders] (uses [fuss] baseline_model and [fuss_scripts])
	- **unlabel_in_domain_ss/ss_computed [14 412 folders] (uses [fuss] baseline_model and [fuss_scripts])
- **Validation
	- validation_ss/ss_computed [1168 files] (uses [fuss] baseline_model and [fuss_scripts])

*Used in baseline without source separation
**Used in baseline with source separation

## Scripts

In the [`scripts/`](scripts) folder, you can find the different steps to generate:
- Synthetic data
- Reverberated synthetic data
- Separated sources of recorded and synthetic mixtures 

	
### DESED Dataset
See [Desed repo][desed] or [Desed_website][desed_website] for more info.

**It is likely that you'll have download issues with the real recordings.
At the end of the download, please send a mail with the TSV files
created in the `missing_files` directory.** (in priority to Nicolas Turpault and Romain Serizel)

#### Synthetic data (1.8Gb)
Freesound dataset [1,2]: A subset of [FSD] is used as foreground sound events for the synthetic subset of the DESED dataset. 
FSD is a large-scale, general-purpose audio dataset composed of Freesound content annotated with labels from the AudioSet Ontology [3].

SINS dataset [4]: The derivative of the SINS dataset used for DCASE2018 task 5 is used as background for the synthetic subset of the dataset for DCASE 2019 task 4.
The SINS dataset contains a continuous recording of one person living in a vacation home over a period of one week.
It was collected using a network of 13 microphone arrays distributed over the entire home.
The microphone array consists of 4 linearly arranged microphones.

The synthetic set is composed of 10 sec audio clips generated with [Scaper] [5].
The foreground events are obtained from FSD.
Each event audio clip was verified manually to ensure that the sound quality and the event-to-background ratio were sufficient to be used an isolated event. 
We also verified that the event was actually dominant in the clip and we controlled if the event onset and offset are present in the clip. Each selected clip was then segmented when needed to remove silences before and after the event and between events when the file contained multiple occurrences of the event class.


##### License
All sounds comming from FSD are released under Creative Commons licences.
**Synthetic sounds can only be used for competition purposes until the full CC license list is made available at the end of the competition.**


#### Real recordings (23.4Gb):
Subset of [Audioset] [3].
Audioset: Real recordings are extracted from Audioset. It consists of an expanding ontology of 632 sound event classes and a collection of 2 million human-labeled 10-second sound clips (less than 21% are shorter than 10-seconds) drawn from 2 million Youtube videos. The ontology is specified as a hierarchical graph of event categories, covering a wide range of human and animal sounds, musical instruments and genres, and common everyday environmental sounds.

The download/extraction process can take approximately 4 hours.
If you experience problems during the download of this subset please contact the task organizers.

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

- [1] F. Font, G. Roma & X. Serra. Freesound technical demo. In Proceedings of the 21st ACM international conference on Multimedia. ACM, 2013.
- [2] E. Fonseca, J. Pons, X. Favory, F. Font, D. Bogdanov, A. Ferraro, S. Oramas, A. Porter & X. Serra. Freesound Datasets: A Platform for the Creation of Open Audio Datasets.
In Proceedings of the 18th International Society for Music Information Retrieval Conference, Suzhou, China, 2017.

- [3] Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter.
Audio Set: An ontology and human-labeled dataset for audio events.
In Proceedings IEEE ICASSP 2017, New Orleans, LA, 2017.

- [4] Gert Dekkers, Steven Lauwereins, Bart Thoen, Mulu Weldegebreal Adhana, Henk Brouckxon, Toon van Waterschoot, Bart Vanrumste, Marian Verhelst, and Peter Karsmakers.
The SINS database for detection of daily activities in a home environment using an acoustic sensor network.
In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2017 Workshop (DCASE2017), 32–36. November 2017.

- [5] J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. Scaper: A library for soundscape synthesis and augmentation
In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017.

- [6] Romain Serizel, Nicolas Turpault. 
Sound Event Detection from Partially Annotated Data: Trends and Challenges. 
IcETRAN conference, Srebrno Jezero, Serbia, June 2019.

[anaconda_download]: https://www.anaconda.com/download/
[Audioset]: https://research.google.com/audioset/index.html
[dcase2019-task4]: http://dcase.community/challenge2019/task-sound-event-detection-in-domestic-environments
[dcase18_task4]: http://dcase.community/challenge2018/task-large-scale-weakly-labeled-semi-supervised-sound-event-detection
[dcase-discussions]: https://groups.google.com/forum/#!forum/dcase-discussions
[dcase_website]: http://dcase.community/challenge2020/task-sound-event-detection-and-separation-in-domestic-environments
[desed]: https://github.com/turpaultn/DESED
[evaluation_dataset]: https://zenodo.org/record/3588172
[FSD]: https://datasets.freesound.org/fsd/
[fuss]: https://zenodo.org/record/3694384/
[fuss_scripts]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[paper2019-description]: https://hal.inria.fr/hal-02160855
[paper2019-eval]: https://hal.inria.fr/hal-02355573
[Scaper]: https://github.com/justinsalamon/scaper
[synthetic_dataset]: https://zenodo.org/record/3700195
[website]: http://dcase.community/challenge2020/