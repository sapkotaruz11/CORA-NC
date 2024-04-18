import csv
import os
import time

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from src.dataset import CoraCitationDataset
from src.models import GAT, GCN, SAGE


def perfromance_evaluations(truth_df, prediction_df):
    merged_df = pd.merge(truth_df, prediction_df, on="paper_id")
    # Calculate the number of correct predictions
    correct_predictions = (merged_df["subject"] == merged_df["class_label"]).sum()
    # Calculate the total number of predictions
    total_predictions = len(merged_df)
    # Calculate accuracy
    accuracy = round(correct_predictions / total_predictions, 2)
    # Print the accuracy
    print("Accuracy on complete dataset:", (accuracy * 100))
    merged_df.to_csv("results/pred_comparision.tsv", sep="\t", index=False)


class EarlyStopping:
    """Class to implement early stopping during training based on validation accuracy.

    Args:
    - patience (int): Number of epochs to wait before stopping if validation accuracy does not improve.

    Attributes:
    - patience (int): Number of epochs to wait before stopping.
    - counter (int): Counter to track the number of epochs without improvement in validation accuracy.
    - best_score (float or None): Best validation accuracy observed so far.
    - early_stop (bool): Flag to indicate if early stopping criteria are met.

    Methods:
    - step(acc): Method to update early stopping criteria based on the current validation accuracy.
    """

    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc):
        score = acc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        acc = correct.item() * 1.0 / len(labels)
        return indices, acc


def gen_predictions(preds, idx_map, node_idx, classes):
    reverse_dict = {value: key for key, value in idx_map.items()}
    predictions = {}
    for item, p in zip(node_idx, preds):
        predictions[reverse_dict[item.item()]] = classes[p]
    return predictions


def run(model_name="SAGE", k_folds=10):
    model_name = model_name
    k_fold = k_folds
    dataset = CoraCitationDataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = dataset.g.to(device)
    node_features = graph.ndata["feat"].to(device)
    node_labels = graph.ndata["label"]
    n_features = node_features.shape[1]
    n_labels = dataset.num_classes
    num_nodes = graph.num_nodes()
    idx_map = dataset.idx_map
    class_names = dataset.classes
    # print dataset statistics
    print(
        """'----Dataset statistics------'
      #Edges %d
      #Nodes %d
      #Classes %d
      #Features %d
        """
        % (
            graph.num_edges(),
            num_nodes,
            n_labels,
            n_features,
        )
    )
    graph = dgl.add_self_loop(graph)
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    X = torch.tensor(list(dataset.idx_map.values()))
    y = node_labels.cpu()
    prediction_dict = {}
    accuracies = []
    for fold, (train, test) in enumerate(skf.split(X, y)):
        # Reinitialize the model for each run
        print(f"Starting fold : {fold}")
        if model_name == "SAGE":
            model = SAGE(
                in_feats=n_features,
                hid_feats=200,
                out_feats=n_labels,
                agg="gcn",
                activation=F.elu,
            ).to(device)
        if model_name == "GCN":
            model = GCN(in_size=n_features, hid_size=100, out_size=n_labels).to(device)
        if model_name == "GAT":
            model = GAT(
                in_size=n_features, hid_size=100, out_size=n_labels, heads=[3, 1]
            ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
        stopper = EarlyStopping(patience=20)

        # Split the training set further into train and validation sets
        train, val = train_test_split(train, test_size=0.3, random_state=42)

        # Initialize train, validation, and test masks based on train, validation, and test splits
        train_mask = torch.tensor([False] * num_nodes, dtype=torch.bool).to(device)
        train_mask[X[train]] = True
        val_mask = torch.tensor([False] * num_nodes, dtype=torch.bool).to(device)
        val_mask[X[val]] = True
        test_mask = torch.tensor([False] * num_nodes, dtype=torch.bool).to(device)
        test_mask[X[test]] = True

        # start model training
        for epoch in range(200):
            model.train()
            logits = model(graph, node_features)
            loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (epoch + 1) % 50 == 0:

                _, train_acc = evaluate(
                    model, graph, node_features, node_labels, train_mask
                )
                _, val_acc = evaluate(
                    model, graph, node_features, node_labels, val_mask
                )
                if stopper.step(val_acc):
                    break
                print(
                    "Epoch {:03d} |  Loss {:.4f} | TrainAcc {:.4f} |"
                    " ValAcc {:.4f} |".format(
                        (epoch + 1),
                        loss.item(),
                        train_acc,
                        val_acc,
                    )
                )
        pred_indices, test_acc = evaluate(
            model, graph, node_features, node_labels, test_mask
        )
        accuracies.append(test_acc)
        predictions = gen_predictions(
            preds=pred_indices, idx_map=idx_map, node_idx=X[test], classes=class_names
        )
        prediction_dict.update(predictions)
    # Calculate mean and standard deviation
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    # Print mean accuracy and standard deviation
    print(
        f"Mean Test Set Accuracy across {k_fold} folds: {mean_accuracy:.2f}  ± {std_accuracy:.2f}"
    )

    # save the prediciton file as a tsc file with <paper_id>, <predictions>
    prediction_df = pd.DataFrame(
        list(prediction_dict.items()), columns=["paper_id", "class_label"]
    )
    file_name = f"results/predictions.tsv"
    # Extract the directory path
    directory = os.path.dirname(file_name)

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    prediction_df.to_csv(file_name, sep="\t", index=False)

    print("Predictions saved successfully as TSV file at {}.".format(file_name))
    perfromance_evaluations(dataset.truth_df, prediction_df)
