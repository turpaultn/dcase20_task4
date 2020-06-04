import argparse
import glob
import os
import warnings
from pprint import pformat

import yaml


def get_data_yaml(yaml_path):
    with open(yaml_path, 'r', encoding="utf-8") as stream:
        # Problem with tab in files, so making sure there is not at the end
        lines = []
        for a in stream.readlines():
            line = a.split("#")[0]
            line = line.rstrip()
            lines.append(line)

        # Read YAML
        data = yaml.safe_load("\n".join(lines))
    return data


def _validate_general(submission):
    if submission["label"] in ["Turpault_INR_task4_SED_1", "Wisdom_GOO_task4_SS_1"]:
        raise ValueError("Please change the label of the submission with your name")
    for key in ["label", "name", "abbreviation"]:
        if "baseline" in submission[key].lower():
            raise ValueError("Please do not put 'baseline' in your system label, name or abbreviation")


def _validate_authors(list_authors):
    corresponding = False
    for author in list_authors:
        if author.get("corresponding") is not None:
            corresponding = True
        if author.get("firstname") is None or author.get("lastname") is None:
            raise ValueError("An author need to have a first name and a last name")

    if not corresponding:
        raise ValueError("Please put a corresponding author")


def _validate_system(system):
    if not isinstance(system["description"]["input_sampling_rate"], (int, float)):
        raise TypeError("The sampling rate needs to be a number (float or int)")

    ac_feat = system["description"]["acoustic_features"]
    if ac_feat is not None:
        if not isinstance(ac_feat, list):
            assert isinstance(ac_feat, str), "acoustic_features is a string if not a list"
            ac_feat = [ac_feat]
        common_values = ["mfcc", "log-mel energies", "log-mel amplitude", "spectrogram", "CQT", "raw waveform"]
        for ac_f in ac_feat:
            if ac_f.lower() not in common_values:
                warnings.warn(f"Please check you don't have a typo if "
                              f"you use common acoustic features: {common_values}")

    if not isinstance(system["complexity"]["total_parameters"], int):
        raise TypeError("the number of total_parameters needs to be an integer")

    if system["source_code"] == "https://github.com/turpaultn/dcase20_task4/tree/public_branch/baseline":
        raise ValueError("If you do not share your source code, please put '!!null'")


def _validate_ss_system(system):
    if system["ensemble_method_subsystem_count"] is not None:
        if not isinstance(system["ensemble_method_subsystem_count"], (int, float)):
            raise TypeError("The ensemble_method_subsystem_count needs to be a number (float or int)")
    if system["source_code"] == "https://github.com/google-research/sound-separation/tree/master/models/dcase2020_fuss_baseline":
        raise ValueError("If you do not share your source code, please put '!!null'")


def _validate_results(results):
    overall = results["development_dataset"]["overall"]
    if not isinstance(overall["F-score"], (int, float)):
        raise TypeError("The F-score on development set needs to be a float or integer")

    per_class = results["development_dataset"]["class_wise"]
    for label in per_class:
        if not isinstance(per_class[label]["F-score"], (int, float)):
            raise TypeError("The F-score on development set needs to be a float or integer")


def _validate_ss_results(results):
    for dataset in results:
        for result in results[dataset]:
            if not isinstance(results[dataset][result], (int, float)):
                raise TypeError(f"The {result} on {dataset} set needs to be a float or integer")


def validate_data(dict_data):
    _validate_general(dict_data["submission"])
    _validate_authors(dict_data["submission"]["authors"])
    if dict_data.get("system") is not None:
        _validate_system(dict_data["system"])
        _validate_results(dict_data["results"])
    if dict_data.get("sed_system") is not None:
        _validate_system(dict_data["sed_system"])
        _validate_results(dict_data["sed_results"])
    if dict_data.get("ss_system") is not None:
        _validate_ss_system(dict_data["ss_system"])
        _validate_ss_results(dict_data["ss_results"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Submission dir to be validated.")
    args = parser.parse_args()
    print(pformat(vars(args)))

    yaml_files = glob.glob(os.path.join(args.input_dir, "*", "*.yaml"))
    for yaml_path in yaml_files:
        data = get_data_yaml(yaml_path)
        validate_data(data)
        print(f"{yaml_path} is validated, continuing...")

    tsv_files = glob.glob(os.path.join(args.input_dir, "*", "*.yaml"))
    if len(tsv_files) < len(yaml_files):
        raise ValueError("Some tsv files are missing, nb yaml != nb tsv files")

    pdf_files = glob.glob(os.path.join(args.input_dir, "*.pdf"))
    if len(pdf_files) == 0:
        raise IndexError("You need to upload a report in your submission")

    with open(os.path.join(args.input_dir, "validated"), "w") as f:
        f.write("Submission validated")

    print("Submission validated")
