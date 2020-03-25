## Explanation

The scripts defines the different steps to download the data.

- 1- It downloads all the necessary equipment:
	- Download recorded data corresponding to DESED dataset. (see contacts in [repo README][readme_repo] for missing_files)
	- Soundbank training from DESED, see [DESED_synth][desed_synth]
	- Download background training to be included in soundbank
	- Download RIR from [FUSS]
	- Download baseline model from Fuss

- 2- You have 2 scenarios to generate the base synthetic data:
	- You download the Jams from 2020 on the DESED dataset
	- You recreate the files thanks by launching the same parameters (random_seed, n_jobs)

- 3- (NOT USED FOR THE BASELINE) Reverberate the synthetic data generated in 2) thanks to [FUSS] RIRs.
Reverberated data are used to train the baseline not using Source separation.

- 4- Apply the FUSS baseline model on the synthetic soundscapes + recorded soundscapes from DESED.
These data are used to train the baseline using Source separation.


[desed_synth]: https://zenodo.org/record/3702397
[desed_repo]: https://github.com/turpaultn/DESED
[desed_website]: https://project.inria.fr/desed/dcase-challenge/dcase-2020-task-4/
[FUSS]: https://zenodo.org/record/3694384/

[readme_repo]: ../