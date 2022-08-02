import pandas as pd
import os
import multiprocessing
from deepface import DeepFace
from functools import partial
import json


def results_to_performance_comparable_dataframe(performance: list[dict]) -> pd.DataFrame:
    # getting only the values needed in a format pandas can easily use
    filtered = [[result['age'], result['gender'], result['dominant_race']] for result in performance]

    # adjusting the values to match the classifications used in the provided correct answers
    for i, entry in enumerate(filtered):
        if entry[0] <= 2:
            entry[0] = "0-2"
        elif entry[0] < 10:
            entry[0] = "3-9"
        elif entry[0] < 20:
            entry[0] = "10-19"
        elif entry[0] < 30:
            entry[0] = "20-29"
        elif entry[0] < 40:
            entry[0] = "30-39"
        elif entry[0] < 50:
            entry[0] = "40-49"
        elif entry[0] < 60:
            entry[0] = "50-59"
        elif entry[0] < 70:
            entry[0] = "60-69"
        elif entry[0] < 20:
            entry[0] = "more than 70"

        if entry[1] == "Man":
            entry[1] = "Male"
        else:
            entry[1] = "Female"

        if entry[2] == "asian":
            entry[2] = "East Asian"
        elif entry[2] == "latino hispanic":
            entry[2] = "Latino_Hispanic"
        elif entry[2] == "middle eastern":
            entry[2] = "Middle Eastern"
        else:
            entry[2] = entry[2].capitalize()

        entry.insert(0, f"train/{i + 1}.jpg")
    return pd.DataFrame(columns=['file', 'age', 'gender', 'race'], data=filtered)




def evaluate_model_performance():
    # Runs the DeepFace model against the provided faces and then evaluates their performance
    val = pd.read_csv("../../face_tests/fairface_label_train.csv")

    default_analyze_kwargs = {"detector_backend": 'skip', "actions": ('age', 'gender', 'race'), "prog_bar": False,
                              "enforce_detection": False}
    photo_paths = []
    directory = '../../face_tests/train_small_500'
    for filename in os.listdir(directory):
        photo_paths.append(os.path.join(directory, filename))

    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(partial(DeepFace.analyze, **default_analyze_kwargs), photo_paths)
    with open("train_small_results.json", "w") as f:
        json.dump(results, f)

    results_to_performance_comparable_dataframe(results)


if __name__ == "__main__":
    with open("train_small_results.json", "r") as f:
        results = results_to_performance_comparable_dataframe(json.load(f))
    results.to_csv("./deepface_classifications.csv")
