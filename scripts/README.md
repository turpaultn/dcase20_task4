## Explanation

NOTE: Before using step 4 you need to be sure you downloaded the recorded soundscapes from DESED.
See [DESED repo][desed_repo] or [DESED website][desed_website] for more information.

The scripts defines the different steps to download the data.

- 1- It downloads all the necessary equipment:
	- Soundbank training for [DESED][desed_synth]
	- Download background training to be included in soundbank
	- Download RIR from [FUSS]
	- Download baseline model from Fuss

- 2- You have 2 scenarios to generate the base synthetic data:
	- You download the Jams from 2020 on the DESED dataset
	- You recreate the files thanks by launching the same parameters (random_seed, n_jobs)

- 3- Reverberate the synthetic data generated in 2) thanks to [FUSS] RIRs.
Reverberated data are used to train the baseline not using Source separation.

- 4- Apply the FUSS baseline model on the reverberated data + recorded soundscapes from DESED.
These data are used to train the baseline using Source separation.


[desed_synth]: https://zenodo.org/record/3700195
[desed_repo]: https://github.com/turpaultn/DESED
[desed_website]: https://project.inria.fr/desed/dcase-challenge/dcase-2020-task-4/
[FUSS]: https://zenodo.org/record/3694384/