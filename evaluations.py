import os

import pandas as pd

from src.dataset import CoraCitationDataset
from sklearn.metrics import accuracy_score


def performance_evaluations(truth_df, prediction_df):
    # Merge truth and prediction DataFrames
    merged_df = pd.merge(truth_df, prediction_df, on="paper_id")

    # Extract true labels and predicted labels
    true_labels = merged_df["subject"]
    predicted_labels = merged_df["class_label"]

    # Calculate accuracy using sklearn's accuracy_score
    accuracy = round(accuracy_score(true_labels, predicted_labels), 2)

    # Print the accuracy
    print("Accuracy on complete dataset:", accuracy * 100)

    # Save the merged DataFrame
    merged_df.to_csv("results/pred_comparison.tsv", sep="\t", index=False)


# File path to the TSV file
pred_file_path = "results/predictions.tsv"
file_path_true = "data/cora/cora.content"

if os.path.exists(pred_file_path):
    # Read the TSV file into a DataFrame using pandas
    prediction_df = pd.read_csv(pred_file_path, delimiter="\t")
    if os.path.exists(file_path_true):
        with open(file_path_true, "r") as f:
            content_size = len(f.readline().split("\t"))
            unique_words = content_size - 2
            column_names = (
                ["paper_id"]
                + [f"term_{idx}" for idx in range(unique_words)]
                + ["subject"]
            )
            truth_df = pd.read_csv(
                file_path_true,
                sep="\t",
                header=None,
                names=column_names,
            ).iloc[:, [0, -1]]
    else:
        dataset = CoraCitationDataset()
        truth_df = dataset.truth_df
    performance_evaluations(truth_df, prediction_df)


else:
    print("No exising results found. Initiating the training.")
    from src.runner import run

    run(model_name="SAGE", k_folds=10)
