import os
import tarfile
from collections import defaultdict

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import requests
import torch


class CoraCitationDataset(object):
    """
    Class for loading and preprocessing the Cora citation dataset.

    Attributes:
    - name (str): Name of the dataset.
    - url (str): URL where the dataset can be downloaded from.
    - raw_dir (str): Directory path to store the raw dataset files.
    - save_dir (str): Directory path to save the preprocessed dataset.
    - backup_file (str): File path of the backup archive containing the dataset.
    - features (torch.Tensor): Features of the nodes in the graph.
    - truth_df (pd.DataFrame): DataFrame containing ground truth values (paper_id, subject).
    - idx_map (dict): Mapping of paper indices to their corresponding positions.
    - classes (numpy.ndarray): Array containing unique class labels.
    - labels (torch.Tensor): Tensor containing label indices for each node.
    - num_classes (int): Number of unique classes in the dataset.
    - g (dgl.DGLGraph): Graph object representing the dataset with node features and labels.
    """

    def __init__(self):
        self.name = "cora"
        self.url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        self.raw_dir = "data/backup"
        self.save_dir = "data"
        self.backup_file = "data/backup/cora.tgz"
        self._load()

        content_dir = os.path.join(self.save_dir, "cora/cora.content")
        citation_dir = os.path.join(self.save_dir, "cora/cora.cites")

        idx_features_labels = np.genfromtxt(content_dir, dtype=np.dtype(str))

        self.features = self._preprocess_features(idx_features_labels)
        idx = idx_features_labels[:, 0].astype(np.int32)
        subjects = idx_features_labels[:, -1].astype(str)

        # Create a DataFrame for ground truth values from indices and subjects
        self.truth_df = pd.DataFrame({"paper_id": idx, "subject": subjects})
        self.idx_map = {j: i for i, j in enumerate(idx)}
        # Extract unique classes and map labels to indices
        self.classes, labels = np.unique(
            idx_features_labels[:, -1], return_inverse=True
        )
        self.labels = torch.tensor(labels)  # Convert labels to a tensor
        self.num_classes = self.classes.shape[0]
        edges_unordered = np.genfromtxt(citation_dir, dtype=np.int32)
        edges_mapped = np.array(
            [[self.idx_map[k], self.idx_map[v]] for k, v in edges_unordered],
            dtype=np.int32,
        )

        # default dict to buid a networkx graph
        result_dict = defaultdict(list)

        # Assuming edges_mapped_array has the format [source, target]
        for target, source in edges_mapped:
            result_dict[source].append(target)

        graph = nx.DiGraph(nx.from_dict_of_lists(result_dict))
        g = dgl.from_networkx(graph)
        g.ndata["label"] = self.labels
        g.ndata["feat"] = self.features
        self.g = g

    def _preprocess_features(self, idx_features_labels):
        # Extract feature values
        # Compute sum of features for each node
        # Compute reciprocal of the sums
        # Handle division by zero cases
        # Convert the scale vector to a sparse diagonal matrix
        # Scale the features using the scale vector
        features = torch.FloatTensor(idx_features_labels[:, 1:-1].astype(np.int32))
        scale_vector = torch.sum(features, dim=1)
        scale_vector = 1 / scale_vector
        scale_vector[scale_vector == float("inf")] = 0
        scale_vector = torch.diag(scale_vector).to_sparse()
        features = scale_vector @ features
        return features

    def _load(self):
        if os.path.exists(self.backup_file):
            print("Exisitng raw files found.")
            self._create_directory(self.save_dir)
            with tarfile.open(self.backup_file, "r:gz") as tar:
                tar.extractall(path=self.save_dir)
            print("File contents extracted successfully.")
        else:
            # Send a GET request to the URL
            response = requests.get(self.url)
            print("Downloading file from {}.".format(self.url))
            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Create the directory if it doesn't exist
                self._create_directory(self.raw_dir)
                # Extract the file name from the URL
                file_name = self.url.split("/")[-1]
                file_path = os.path.join(self.raw_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(response.content)

                print(
                    "File downloaded successfully and stored at {}/.".format(
                        self.raw_dir
                    )
                )
                self._create_directory(self.save_dir)
                # Extract the contents of the downloaded .tgz file
                with tarfile.open(file_path, "r:gz") as tar:
                    tar.extractall(path=self.save_dir)
                print(
                    "File contents extracted successfully and save at {}/.".format(
                        self.save_dir
                    )
                )

            else:
                print(f"Failed to download file. Status code: {response.status_code}")

    def _create_directory(self, directory):
        if os.path.exists(directory):
            # Clear existing files
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        else:
            os.makedirs(directory, exist_ok=True)
