# CORA-NC
Repository for Node Classification task on CORA dataset

## Approach

### Dataset
The Cora dataset consists of Machine Learning papers classified into one of the following seven classes: Case-Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, and Theory. The papers were selected in a way such that in the final corpus, every paper cites or is cited by at least one other paper. There are 2708 papers in the whole corpus and 5429 links.

The dataset for this task is built as the CoraCitationDataset class, which is a self-built Python class. It uses the nodes, features and labels from the cora.content file and links from cora.cites file. The features consist of 1433 words mapped into binary labels. The features are passed through a preprocessing pipeline. 

In this dataset preprocessing, feature values are extracted from the dataset. The sum of features for each node is computed, followed by the calculation of reciprocals of these sums. Special consideration is given to handling cases where division by zero may occur. The scale vector is then transformed into a sparse diagonal matrix, which is used to scale the features accordingly.

The nodes and citation links are used to create a dgl graph object (a homogenous graph, as there is only 1 type of node and edge). The processed features are assigned to the graph nodes respectively.

### Model
The default model for the training is a SAGE model, which uses DGL implementation of Sage Convolution Layer adapted from [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf). GraphSAGE (Graph Sample and Aggregation) convolution operates by iteratively updating node representations in a graph. 

It begins by sampling a fixed number of neighbours for each node, then aggregates feature information from these neighbours using a neural network function. The aggregated features are combined with the central node's features to create a new representation, allowing it to incorporate information from its neighbourhood. This process is iterated multiple times, enabling nodes to refine their representations by considering increasingly distant neighbours. During training, the aggregation function's parameters are learned via backpropagation, optimizing the model to generate node representations useful for tasks like node classification or link prediction.

In the experiments, we used GCN as an aggregation function for the SAGE convolution and ELU as the activation function for the first layer. ELU (Exponential Linear Unit) activation function is a variant of the ReLU function, which introduces a non-linearity by smoothly handling negative values with an exponential decay.

The input size for the model was given as the size of the features and the output size as the no of types of classes in the dataset for node classification. 

Apart from the SAGE, we also implemented other GNN models, such as Graph Attention Network(GAT) based on [GRAPH ATTENTION NETWORKS](https://arxiv.org/pdf/1710.10903.pdf) and  [Graph Convolution Network(GCN)](https://docs.dgl.ai/en/2.0.x/tutorials/models/1_gnn/1_gcn.html)  adapted from DGL the implementation. 

### Training 
The training process follows a k-fold (10-fold) cross-validation approach, ensuring robust evaluation. Each fold begins by initializing the new instance of the model to ensure the new weights are initialized for each run with parameters and activation functions as described above. The model is optimized using the Adam optimizer with a learning rate of 5e-3, and the weight decay rate set to 5e-4. The learning rate (lr) determines the optimization step size, with higher values enabling faster convergence but risking overshooting or oscillation. Conversely, lower rates ensure stability but slower convergence. The chosen 5e-3 balances speed and stability. Weight decay acts as L2 regularization, penalizing large parameter values to prevent overfitting. A weight decay of 5e-4 controls model complexity, aiding generalization.

To ensure comprehensive training and evaluation, a 10-fold data split is implemented, where each fold involves masking different nodes for training and testing. This iterative process guarantees that all nodes are included in the test set exactly once across the training runs. In the experiments, we have used the Stratifed variant of 10-fold validation, adapted from the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) library. Using stratified k-fold cross-validation ensures that each fold maintains a representative class distribution, enhancing the reliability of model evaluation across different data subsets. Setting a random seed allows for the reproducibility of data distribution in various runs, facilitating consistent experimentation and comparison of different model configurations.


The training is done for 200 epochs and aims to minimize the cross-entropy loss. Early stopping is employed to prevent overfitting, monitor training accuracy and halt training if it does not improve for a set number of epochs. At the end of each fold, model predictions are generated for the validation set, and accuracy is evaluated. This process repeats for each fold, accumulating accuracies for subsequent analysis. Finally, the mean and standard deviation of accuracies across folds are calculated, providing insights into model performance and variability.

### Inference
As the entire dataset is divided into 10 folds training/test sets, each paper_id is guaranteed to fall once in the test set and we extract the predictions for the paper_id from the test sets only. The test sets are only used to measure the validation accuracy during certain times but are not provided to interfere with the training and are hence unseen data. The predictions for all the paper_ids are then saved in a predictions file as predictions.tsv in the results directory. Moreover, a file called "pred_comparision" containing the prediction and the original label for all nodes is also stored in the results directory, where the subject is the true class and the node label is the predicted class for the paper.

### Performace 
The Accuracy on the entire dataset is found to be 89 % for the default model with (0.89  ± 0.02) averaged over different runs.

### Execution
The repository is formatted following standard practice for a Python framework. The code is written following Python PEP-8 standards and validated using static analysis tools. The experiments can be carried out using a single command. Most of the cases for missing directories and data files are handled properly.
 
## Installation Guide for the CORA-NC Framework

Follow these steps to set up the virtual environment on your system:

### Step 1: Clone the GitHub Repository

First, clone the repository from GitHub using the following command:

```shell
git clone https://github.com/sapkotaruz11/CORA-NC.git
```


### Step 2: Install Conda

If you don't have Conda installed, download and install it from [Anaconda's official website](https://www.anaconda.com/products/individual).

### Step 3: Create the Conda Environment

Open your terminal or command prompt and run the following command to create a Conda environment named `cora` ( or any names you prefer) with Python 3.10:

```shell
conda create --name cora python=3.10
```

### Step 4: Activate the Environment

Activate the newly created environment using:

```shell
conda activate cora
```

### Step 5: Install Dependencies

Get inside the cloned directory using the command:

```shell
cd CORA-NC
```

Next step is to install all the dependencies required to execute the experiments. First, install the [DGL library](https://www.dgl.ai/pages/start.html) according to the GPU availability and Pytorch Version(2.1.x).

 To install rest of the required dependencies, run:

```shell
pip install -r requirements.txt
```

This command will automatically install all the libraries and packages listed in the `requirements.txt` file.

After completing these steps, the `cora` environment should be set up with all the necessary dependencies.

## Running the Node Classification Tasks

### Dataset
The dataset for the project was downloaded from the provided URL and saved at `data/backup/`. If you want to carry out the experiments with a fresh copy of dataset, simply delete the data directory and run the experiments. The dataset will be downloaded automatically from  [source]("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz").


### Experiments
To run the experiments, use the command 
```shell
python main.py
```
This will execute the node classification on CORA dataset using 10 fold data split. The mean validation accuracy  with standard deviation for different runs will be printed. Finally, the accuracy on the entire dataset will be printed as well.

The default machine learning model for the task is SAGE. However, there are other models such as Graph Attention Network(GAT) and  Graph Convolution Network(GCN)  available if you wish to run the experiments. To use those models, run the experiments with:

```shell
python main.py --model GAT
```

There is also a provision to set the number of folds for data split, which is kept at 10 as default value. To run the experiments with different number of folds, run with command:
```shell
python main.py --model GAT --k_folds 5
```
The number of folds should be greater or equal to 2.

### Evaluations

If you wish to evaluate the perfromance without carrying out the experiments using the previous predicitons, run the command:
```shell
python evaluations.py
```