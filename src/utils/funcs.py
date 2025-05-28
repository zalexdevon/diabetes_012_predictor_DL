from Mylib import myfuncs
import os
import itertools


SCORINGS_PREFER_MININUM = ["log_loss", "mse", "mae"]
SCORINGS_PREFER_MAXIMUM = ["accuracy"]


def gather_result_from_model_training():
    components = []
    model_training_path = "artifacts/model_training"

    list_models_folder_path = [
        f"{model_training_path}/{item}" for item in os.listdir(model_training_path)
    ]
    for models_folder_path in list_models_folder_path:
        list_model_path = [
            f"{models_folder_path}/{item}" for item in os.listdir(models_folder_path)
        ]
        for model_path in list_model_path:
            result = myfuncs.load_python_object(f"{model_path}/result.pkl")
            components.append(result)

    return components


def gather_result_from_model_training_for_1folder(folder):
    components = []
    folder_path = f"artifacts/model_training/{folder}"

    list_model_path = [f"{folder_path}/{item}" for item in os.listdir(folder_path)]

    for model_path in list_model_path:
        result = myfuncs.load_python_object(f"{model_path}/result.pkl")
        components.append(result)

    return components


def gather_result_from_model_training_for_many_folders(folders):
    list_components_for_1folder = [
        gather_result_from_model_training_for_1folder(item) for item in folders
    ]
    return list(itertools.chain(*list_components_for_1folder))


def get_reverse_param_in_sorted(scoring):
    return scoring in SCORINGS_PREFER_MAXIMUM
