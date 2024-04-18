# CORA-NC
Repository for Node Classification task on CORA dataset

## Approach

### Dataset
The Cora dataset consists of Machine Learning papers classified into one of the following seven classes: Case-Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, and Theory. The papers were selected in a way such that in the final corpus, every paper cites or is cited by at least one other paper. The dataset is available to download at the [source]("https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz").

#### Dataset Statistics

- Papers &#8594; 2780
- Citation Links &#8594; 5429
- Subjects &#8594; 7
- Unique Words &#8594; 1433


The dataset used for this task is implemented as the CoraCitationDataset class, which is a custom Python class we developed specifically for this task. The dataset class uses the paper_id, features and labels from the cora.content file and citation links from cora.cites file. The features consist of 1433 unique words mapped into binary labels. The features are passed through a preprocessing pipeline. In this preprocessing pipeline, feature values are extracted from the dataset. The sum of features for each paper is computed, followed by the calculation of reciprocals of these sums while also handling cases where division by zero may occur. The scale vector is then transformed into a sparse diagonal matrix, which is used to scale the features accordingly.

The paper_ids and citation links are used to create a DGL graph (homogenous graph, as there is only 1 type of node and edge), paper as nodes and citation links as edges. The paper_ids are mapped to node_id as series of natural number for easier implementation. The processed features are assigned to the graph nodes respectively. The COraCitationGraph dataset contains the graph, node features and all the other information like node_id to paper_id mapping etc. 

### Model
The default model for the training is a SAGE model, which uses DGL implementation of SAGE Convolution Layer and is adapted from [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216.pdf). GraphSAGE (Graph Sample and Aggregation) convolution operates by iteratively updating node representations in a graph. 

It begins by sampling a fixed number of neighbours for each node, then aggregates feature information from these neighbours using a neural network function. The aggregated features are combined with the central node's features to create a new representation, allowing it to incorporate information from its neighbourhood. This process is iterated multiple times, enabling nodes to refine their representations by considering increasingly distant neighbours. During training, the aggregation function's parameters are learned via backpropagation, optimizing the model to generate node representations useful for tasks like node classification or link prediction.

In the experiments, we used GCN as an aggregation function for the SAGE convolution and ELU as the activation function for the first layer. ELU (Exponential Linear Unit) activation function is a variant of the ReLU function, which introduces a non-linearity by smoothly handling negative values with an exponential decay.

The input size for the model was given as the size of the features and the output size as the number of unique classes the papers belong to in the dataset. 

Apart from the SAGE, we also implemented other GNN models, such as Graph Attention Network(GAT) based on [GRAPH ATTENTION NETWORKS](https://arxiv.org/pdf/1710.10903.pdf) and  [Graph Convolution Network(GCN)](https://docs.dgl.ai/en/2.0.x/tutorials/models/1_gnn/1_gcn.html)  adapted from DGL the implementation. 

### Training 
The training process follows a k-fold (10-fold) cross-validation approach. Each fold is trained separately, which begins by initializing the new instance of the model to ensure new weights are initialized for each run with parameters and activation functions as described above. The model is optimized using the Adam optimizer with a learning rate of 5e-3, and the weight decay rate set to 5e-4. The learning rate (lr) dictates the optimization step size: higher values ensure faster convergence but risk overshooting or oscillation, while lower rates offer stability but slower convergence. The selected 5e-3, being relatively small, strikes a balance between speed and stability. Weight decay functions as L2 regularization, penalizing large parameter values to prevent overfitting. A weight decay of 5e-4 manages model complexity, promoting generalization.

We followed Trasnductive learning approach in which the features for both the train and test set are provided during the training. It is a general learning approach, especially for node classification tasks on graph based learning as Graph Neural Networks require entire graph data to learn in order make predictions on the graph nodes. Eventhough the entire graph and features are passed to the model during training, the loss function for the model is updated only using the training set(the training loss), so the test set can still be considered to be unseen data. The dataset is divided into training and test set using node masking. The masking technique is a common approach used in machine learning to split a dataset into training and testing sets. In this process, we create a mask, which is essentially a binary array or a set of indices, where True represents the data points intended for the training set and False represents those intended for the testing set. By applying this mask to the dataset, we can separate the data accordingly. This helps ensure that the model is trained on a portion of the data while being evaluated on unseen data to assess its generalization performance. 


We used 10 fold validation technique to divide data into 10 folds of train and test set. The indices of the nodes from the split were then taken to create the training and test masks. In the experiments, we have used the Stratifed variant of 10-fold validation, adapted from the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) library. Using stratified k-fold cross-validation ensures that each fold maintains a representative class distribution, enhancing the reliability of model evaluation across different data subsets. Setting a random seed allows for the reproducibility of data distribution.


We trained the model for 200 epochs by minimizing the cross-entropy loss function. Early stopping is employed to prevent overfitting, monitoring validation accuracy and halt training if it does not improve for a set number of epochs. We extracted a small portion from the train set as validation set which was used to measure the validation accuracy and stop the training early if Early Stopping criteria wad met. At the end of each fold, model predictions were generated for the test set, and evaluated using accuracy metric. This process repeated for each fold, accumulating accuracies for subsequent analysis. Finally, we calculated the mean and standard deviation of accuracies across folds of data distribution, providing insights into model performance and variability.

### Inference
As the entire dataset was divided into 10 folds training/test sets, each paper_id was guaranteed to fall once in the test set and the predictions for the paper_id were aquired from the test sets only.  The predictions for all the paper_ids were then saved in a predictions file as `predictions.tsv` in the results directory. Moreover, a file called `pred_comparison.tsv` containing both the prediction and the original label for all nodes is also stored in the results directory, where the subject is the true class and the node label is the predicted class for the paper.

### Performace 
The Accuracy on the entire dataset was found to be 89 % with the SAGE(default) model and (0.89  ± 0.02) averaged test accuracy with standard deviation over different runs.

### Execution
The repository is structured following standard practice for a Python framework. The code is written following Python PEP-8 standards and validated using static analysis tools. 

The experiments can be carried out using a single command. The test experiments were carried out on Ubuntu 22.04 LTS using a NVIDIA RTX 3070 laptop GPU. 
 
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

If you wish to evaluate the perfromance without carrying out the experiments using the previous predictions, run the command:
```shell
python evaluations.py
```