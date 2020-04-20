# Baseline

See in config.py the different paths if you want to modify them for your own data.

## Train SED model without source separation

- `python main.py`

## Testing baseline models
### SED only
```bash
python TestModel.py -m "model_path" -g ../dataset/metadata/validation/validation.tsv  \
-ga ../dataset/audio/validation -s stored_data/baseline/validation_predictions.tsv 
```

### Sound separation and SED
This assume you extracted the sources as described in [4_separate_mixtures.sh].
```bash
python TestModel_ss_late_integration.py -m "model_path" -g ../dataset/metadata/validation/validation.tsv  \
-ga ../dataset/audio/validation -s stored_data/baseline/validation_predictions.tsv \
-a ../dataset/audio/validation_ss/separated_sources/ -k "1"
```
The `-k "1"` means that we are using only the 2nd sources of the sound separation model.
The sound separation model has been trained on soundscapes being a mix of FUSS and DESED data. 
It has 3 sources:
- DESED background
- DESED foreground (the one used with SED)
- FUSS mixture

To combine SS and SED, we average the predictions of the mixture (usual SED) and 
the estimated DESED foreground (before binarization).

Multiple experiments have been made to combine SS and SED and will be presented in the baselne paper.

**Note:** The performance might not be exactly reproducible on a GPU based system.
That is why, you can download the [weights of the networks][model-weights] used for the experiments.


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


### References
 - [[1]] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.
 - [[2]]  Tarvainen, A. and Valpola, H., 2017.
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
 In Advances in neural information processing systems (pp. 1195-1204).

[1]: http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Delphin_15.pdf
[2]: https://arxiv.org/pdf/1703.01780.pdf
[4_separate_mixtures.sh]: ../scripts/4_separate_mixtures.sh

[dcase2019-baseline]: https://github.com/turpaultn/DCASE2019_task4
[model-weights]: https://doi.org/10.5281/zenodo.3726375