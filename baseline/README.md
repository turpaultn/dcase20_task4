## Baseline

See in config.py the different paths if you want to modify them for your own data.

### Train SED model without source separation

- `python main.py`

### Train SED model with separated sources already extracted

Make sure you extracted the separated sources with the FUSS model (see [4_separate_mixtures.sh])

- `python main.py -ss`


### System description
The baseline model is inspired by last year 2nd best submission system of DCASE 2019 task 4:
L. Delphin-Poulat & C. Plapous [[1]].

It is an improvement of [dcase 2019 baseline][dcase2019-baseline]. The model is a mean-teacher model [[2]][2].

The main differences of the baseline system (without source separation) compared to dcase 2019:
- The sampling rate becomes 16kHz.
- Features:
	- 2048 fft window, 255 hop size, 8000 max frequency for mel, 128 mel bins.
- Different synthetic dataset is used.
- The architecture (number of layers) is taken from L. Delphin-Poulat & C. Plapous [[1]].
- There is rampup for the learning rate for 50 epochs.
- Median window of 0.45s.
- Early stopping (10 epochs).


**Note:** The performance might not be exactly reproducible on a GPU based system.
That is why, you can download the [weights of the networks][model-weights]
used for the experiments and run `TestModel.py --model_path="Path_of_model" ` to reproduce the results.

### References
 - [[1]] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.
 - [[2]]  Tarvainen, A. and Valpola, H., 2017.
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
 In Advances in neural information processing systems (pp. 1195-1204).

[1]: http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Delphin_15.pdf
[2]: https://arxiv.org/pdf/1703.01780.pdf
[4_separate_mixtures.sh]: ../scripts/4_separate_mixtures.sh

[dcase2019-baseline]: https://github.com/turpaultn/DCASE2019_task4
[model-weights]: https://zenodo.org/record/3726376