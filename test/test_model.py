import pandas as pd
import os
import multiprocessing
from deepface import DeepFace
from functools import partial
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn


def results_to_performance_comparable_dataframe(performance: list[dict]) -> pd.DataFrame:
    # getting only the values needed in a format pandas can easily use
    filtered = [[result['age'], result['gender'], result['dominant_race']] for result in performance]

    # adjusting the values to match the classifications used in the provided correct answers
    # change to use python match/case syntax during refactor
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
        else:
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


def compare_performance(pred: pd.DataFrame, true: pd.DataFrame):
    def generate_graph(attribute: str):
        attribute_confusion_matrix = confusion_matrix(true.loc[:499, attribute], pred[attribute])
        labels = true[attribute].unique()
        labels.sort()
        disp = ConfusionMatrixDisplay(confusion_matrix=attribute_confusion_matrix, display_labels=labels)
        disp.plot()
        plt.xticks(rotation=45)
        plt.title(f"{attribute} Confusion Matrix")
        plt.savefig(f"./figures/{attribute}_confusion_matrix.png", bbox_inches="tight")

    generate_graph("gender")
    generate_graph("race")
    generate_graph("age")


def evaluate_model_performance():
    # Runs the DeepFace model against the provided faces and then evaluates their performance
    true_classifications = pd.read_csv("../../face_tests/fairface_label_train.csv")

    default_analyze_kwargs = {"detector_backend": 'skip', "actions": ('age', 'gender', 'race'), "prog_bar": False,
                              "enforce_detection": False}
    photo_paths = []
    directory = '../../face_tests/train_small_500'
    for filename in os.listdir(directory):
        photo_paths.append(os.path.join(directory, filename))
    # the paths are sorted alphabetically, we really want them numerically
    photo_paths_sorted = sorted(photo_paths, key=lambda x: int(x.split('\\')[-1][:-4]))

    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(partial(DeepFace.analyze, **default_analyze_kwargs), photo_paths_sorted)
    with open("train_small_results.json", "w") as f:
        json.dump(results, f)

    pred_classifications = results_to_performance_comparable_dataframe(results)
    pred_classifications.to_csv("./deepface_classifications.csv")
    compare_performance(pred_classifications, true_classifications)


if __name__ == "__main__":
    # for a full run, use evaluate_model_performance. below is shortened for when you already have done the processing
    # would probably work better as a jupyter notebook
    true_classifications = pd.read_csv("../../face_tests/fairface_label_train.csv")
    pred_classifications = pd.read_csv("./deepface_classifications.csv")
    compare_performance(pred_classifications, true_classifications)
