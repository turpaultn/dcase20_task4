## Explanation

The scripts defines the different steps to download the data.

- 1- It downloads the data (real data and synthetic soundbank):
	- Download recorded data corresponding to DESED dataset. (see contacts in [repo README][readme_repo] for missing_files)
	- Soundbank training from DESED, see [DESED_synth][desed_synth]
	- Download background training to be included in soundbank

- 2- You have 2 scenarios to generate the base synthetic data:
	- You download the Jams from 2020 on the DESED dataset
	- You recreate the files thanks by launching the same parameters (random_seed, n_jobs)

- 3- (NOT USED FOR THE BASELINE) Reverberate synthetic data:
	- Download RIR from [FUSS]
	- Reverberate the synthetic data generated in 2)
	- Reverberated data are used to train the baseline not using Source separation.

- 4- Separate sounds using FUSS:
	- Download baseline model from Fuss
	- Apply the FUSS baseline model on the synthetic soundscapes + recorded soundscapes from DESED.
These data are used to train the baseline using Source separation.


[desed_synth]: https://doi.org/10.5281/zenodo.3550598
[FUSS]: https://doi.org/10.5281/zenodo.3694383

[readme_repo]: ../