import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import jaccard_score

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed value to use for numpy and torch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SimpleUnsupervisedNN(nn.Module):
    """
    A simple unsupervised neural network with configurable activation function.
    
    Parameters:
    activation (str): The activation function to use ('relu' or 'sin').
    """
    def __init__(self, activation='relu'):
        super().__init__()
        self.layer1 = nn.Linear(2, 6)
        self.layer2 = nn.Linear(6, 3)
        self.output = nn.Linear(3, 1)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        if self.activation == 'relu':
            self.act_fn = nn.ReLU()
        elif self.activation == 'sin':
            self.act_fn = torch.sin
        else:
            raise ValueError('Choose relu or sin')

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        x = self.act_fn(self.layer1(x))
        x = self.act_fn(self.layer2(x))
        x = self.output(x)
        x = self.sigmoid(x)
        return x

def balance_clusters(cluster0, cluster1):
    """
    Determine the number of samples needed to augment each cluster for balancing.

    Parameters:
    cluster0 (np.ndarray): Data points in cluster 0.
    cluster1 (np.ndarray): Data points in cluster 1.

    Returns:
    tuple: Number of samples needed to augment cluster 0 and cluster 1.
    """
    num_to_augment_cluster0 = len(cluster1)
    num_to_augment_cluster1 = len(cluster0)

    print(f"augment_cluster_0: {num_to_augment_cluster0}, augment_cluster_1: {num_to_augment_cluster1}")
    return num_to_augment_cluster0, num_to_augment_cluster1

def augment_with_mixup(x, y, num_to_augment):
    """
    Augment the dataset using mixup technique.

    Parameters:
    x (np.ndarray): Input features.
    y (np.ndarray): Labels.
    num_to_augment (int): Number of samples to generate.

    Returns:
    tuple: Augmented input features and labels.
    """
    mixed_x = []
    mixed_y = []

    while len(mixed_x) < num_to_augment:
        idx1, idx2 = np.random.choice(range(x.shape[0]), 2, replace=False)
        x1, y1 = x[idx1], y[idx1]
        x2, y2 = x[idx2], y[idx2]
        mx, my = mixup_data(x1, y1, x2, y2)
        mixed_x.append(mx)
        mixed_y.append(my)

    mixed_x = np.array(mixed_x)
    mixed_y = np.array(mixed_y)

    augmented_x = np.concatenate([x, mixed_x])
    augmented_y = np.concatenate([y, mixed_y])

    return augmented_x, augmented_y

def train_model(cluster0, samples, mixup=False, n_epochs=20, batch_size=10, k_ratio=None, T=1, num_to_augment=0):
    """
    Train the model using the specified parameters.

    Parameters:
    cluster0 (np.ndarray): Data points in cluster 0.
    samples (np.ndarray): Additional samples to include in training (or this could be the other cluster)
    mixup (bool): Whether to use mixup augmentation.
    n_epochs (int): Number of training epochs.
    batch_size (int): Batch size for training.
    k_ratio (float): Ratio for determining the number of nearest neighbors.
    T (float): Temperature parameter for distance calculation.
    num_to_augment (int): Number of samples to augment using mixup.

    Returns:
    SimpleUnsupervisedNN: Trained model.
    """
    ds = np.concatenate([cluster0, samples])

    # Finding nearest neighbors & similarity
    k = int(k_ratio * len(cluster0)) if k_ratio is not None else len(cluster0)
    neighbors = NearestNeighbors(n_neighbors=k).fit(cluster0)
    distances, indices = neighbors.kneighbors(ds)
    exp_distances = np.exp(-T * np.square(distances))
    avg_exp_distance = np.mean(exp_distances, axis=1)

    y_ds = avg_exp_distance

    if mixup and num_to_augment > 0:
        ds, y_ds = augment_with_mixup(ds, y_ds, num_to_augment)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_train = torch.tensor(ds, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_ds, dtype=torch.float32).reshape(-1, 1).to(device)
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleUnsupervisedNN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_mse = float('inf')
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            mse = float(loss_fn(y_pred, y_train))
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    return model

def compare_models(model0, model1, dataset):
    """
    Compare two models to determine which performs better on the given dataset.

    Parameters:
    model0 (SimpleUnsupervisedNN): The first model to compare.
    model1 (SimpleUnsupervisedNN): The second model to compare.
    dataset (np.ndarray): The dataset to evaluate the models on.

    Returns:
    np.ndarray: A boolean array indicating which model performed better for each sample.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model0.to(device)
    model1.to(device)
    dataset = torch.tensor(dataset, dtype=torch.float32).to(device)

    model0.eval()
    model1.eval()

    with torch.no_grad():
        scores_model0 = model0(dataset)
        scores_model1 = model1(dataset)

    higher_score_model0 = (scores_model0 > scores_model1).cpu().numpy().flatten()

    return higher_score_model0

def mixup_data(x1, y1, x2, y2):
    """
    Mix two samples using the mixup technique.

    Parameters:
    x1 (np.ndarray): The first sample's features.
    y1 (float): The first sample's label.
    x2 (np.ndarray): The second sample's features.
    y2 (float): The second sample's label.

    Returns:
    tuple: Mixed features and label.
    """
    lam = np.random.rand()
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

def test(X, initial_labels, samples=None, num_iterations=5, mixup=False, k_ratio=None):
    """
    Perform the iterative training and comparison process.

    Parameters:
    X (np.ndarray): The dataset.
    initial_labels (np.ndarray): Initial labels for the dataset.
    samples (np.ndarray): Additional samples to include in training.
    num_iterations (int): Number of iterations to run.
    mixup (bool): Whether to use mixup augmentation.
    k_ratio (float): Ratio for determining the number of nearest neighbors.

    Returns:
    tuple: Trained models and the number of iterations to convergence.
    """
    labels = initial_labels
    prev_labels = None
    for i in range(num_iterations):
        print(f"Iteration {i+1}")

        cluster0 = X[labels == 0]
        cluster1 = X[labels == 1]

        if len(cluster0) == 0 or len(cluster1) == 0:
            print(f"Collapsed at iteration {i+1}")
            return None, None, -1  # collapsed

        num_to_augment_cluster0, num_to_augment_cluster1 = balance_clusters(cluster0, cluster1)

        try:
            if samples is not None:
                trained_model0 = train_model(cluster0, samples, mixup=mixup, n_epochs=50, batch_size=10, k_ratio=k_ratio, T=1, num_to_augment=num_to_augment_cluster0)
                trained_model1 = train_model(cluster1, samples, mixup=mixup, n_epochs=50, batch_size=10, k_ratio=k_ratio, T=1, num_to_augment=num_to_augment_cluster1)
            else:
                trained_model0 = train_model(cluster1, cluster0, mixup=mixup, n_epochs=50, batch_size=10, k_ratio=k_ratio, T=1, num_to_augment=num_to_augment_cluster1)
                trained_model1 = train_model(cluster0, cluster1, mixup=mixup, n_epochs=50, batch_size=10, k_ratio=k_ratio, T=1, num_to_augment=num_to_augment_cluster0)
        except ValueError as e:
            print(f"Training failed at iteration {i+1} with error: {e}")
            return None, None, -1  # collapse

        new_labels = compare_models(trained_model0, trained_model1, X)
        if np.array_equal(new_labels, prev_labels):
            print(f"Converged at iteration {i+1}")
            return trained_model0, trained_model1, i + 1  # number of iterations it took to converge 
        prev_labels = labels
        labels = new_labels

    return trained_model0, trained_model1, num_iterations 

def initialize_labels(X, method, model, device):
    """
    Initialize labels for the dataset.

    Parameters:
    X (np.ndarray): The dataset.
    method (str): The method to use for initialization ('random', 'run forward', 'two points').
    model (SimpleUnsupervisedNN): The model to use for 'run forward' initialization.
    device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
    np.ndarray: Initialized labels.
    """
    if method == 'random':
        # Choose completely random labels 
        return np.random.choice([0, 1], size=len(X))
    elif method == 'run forward':
        # Initialize two models and without training allow the random weights to assign clusters and take the best out of the two
        model1 = SimpleUnsupervisedNN().to(device)
        model1.eval()
        labels1 = model1(torch.tensor(X, dtype=torch.float32).to(device))
        labels1 = labels1.detach().cpu().numpy().flatten()  

        model2 = SimpleUnsupervisedNN().to(device)
        model2.eval()
        labels2 = model2(torch.tensor(X, dtype=torch.float32).to(device))
        labels2 = labels2.detach().cpu().numpy().flatten()  

        labels = (labels1 > labels2)
        return labels
    elif method == 'two points':
        # Choose two random points and assign labels based on proximity to those two points 
        idx = np.random.choice(X.shape[0], 2, replace=False)
        point1, point2 = X[idx[0]], X[idx[1]]
        dist_to_point1 = np.linalg.norm(X - point1, axis=1)
        dist_to_point2 = np.linalg.norm(X - point2, axis=1)
        return np.where(dist_to_point1 < dist_to_point2, 1, 0)
    else:
        raise ValueError('Unknown initialization method')

def is_bad_initialization(labels, y):
    """
    Check if the initialization is bad based on the similarity with true labels.
    "Bad" means too good of an initialization/too similar to the true labels. 

    Parameters:
    labels (np.ndarray): Initialized labels.
    y (np.ndarray): True labels.

    Returns:
    bool: Whether the initialization is bad.
    """
    similarity = np.mean(labels == y)
    similarity_complement = np.mean(labels == (1 - y))
    return similarity >= 0.97 or similarity_complement >= 0.97

def experiment(initialization_methods, use_samples_options, use_mixup_options, k_ratios, file_path, dataset='moons', rounds=200, variable=False):
    """
    Conduct experiments with various configurations and save the results.

    Parameters:
    initialization_methods (list): List of initialization methods to test.
    use_samples_options (list): List of boolean options for using samples 
    use_mixup_options (list): List of boolean options for using mixup augmentation.
    k_ratios (list): List of k ratios to test.
    file_path (str): Path to save the results CSV file.
    dataset (str): Dataset type ('moons' or 'blobs').
    rounds (int): Number of rounds to run for each configuration.
    variable (bool): Whether to use a fixed (variable=False) or variable dataset 

    Returns:
    pd.DataFrame: Dataframe containing the results of the experiments.
    """
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    if variable is False:
        if dataset == 'moons':
            X, y = make_moons(n_samples=200, noise=0.05)
        elif dataset == 'blobs':
            X, y = make_blobs(n_samples=200, centers=2, n_features=2, shuffle=False)
        else:
            raise ValueError('Unknown dataset type')

    for k_ratio in k_ratios:
        for init_method in initialization_methods:
            for use_samples in use_samples_options:
                for use_mixup in use_mixup_options:
                    print(f"Running experiment: initialization_method={init_method}, use_samples={use_samples}, use_mixup={use_mixup}, k_ratio={k_ratio}")
                    
                    failures = 0
                    good_convergences = 0
                    total_iterations = 0
                    jaccard_scores = []
                    bad_initializations = 0
                    collapses = 0

                    for round_num in range(rounds):
                        set_seed(42 + round_num)
                        
                        if variable:
                            if dataset == 'moons':
                                X, y = make_moons(n_samples=200, noise=0.05)
                            elif dataset == 'blobs':
                                X, y = make_blobs(n_samples=200, centers=2, n_features=2, shuffle=False)
                            else:
                                raise ValueError('Unknown dataset type')
                        
                        midpoint = np.mean(X, axis=0)
                        ds_std = np.std(X)
                        model = SimpleUnsupervisedNN().to(device)

                        while True:
                            labels = initialize_labels(X, init_method, model, device)
                            if not is_bad_initialization(labels, y):
                                break
                            print("bad initialization")
                            bad_initializations += 1

                        samples = np.random.randn(X.shape[0], X.shape[1]) * ds_std + midpoint if use_samples else None

                        trained_model0, trained_model1, iterations_to_converge = test(X, labels, samples=samples, num_iterations=10, mixup=use_mixup, k_ratio=k_ratio)
                        
                        if iterations_to_converge == -1:
                            collapses += 1
                        elif iterations_to_converge < 10:
                            good_convergences += 1
                            total_iterations += iterations_to_converge
                            current_labels = compare_models(trained_model0, trained_model1, X)
                            jaccard = max(jaccard_score(current_labels, y), jaccard_score(1 - current_labels, y))
                            jaccard_scores.append(jaccard)
                        else:
                            failures += 1

                    result = {
                        'k_ratio': k_ratio,
                        'initialization_method': init_method,
                        'use_samples': use_samples,
                        'use_mixup': use_mixup,
                        'total_rounds': rounds,
                        'convergence_rate': good_convergences / rounds,
                        'doesnt_converge_rate': failures / rounds,
                        'collapse_rate': collapses / rounds,
                        'average_iterations': total_iterations / good_convergences if good_convergences else 0,
                        'average_jaccard_score': np.mean(jaccard_scores) if jaccard_scores else 0,
                        'bad_initialization_count': bad_initializations
                    }

                    results.append(result)
                    results_df = pd.DataFrame(results)
                    results_df = results_df[['k_ratio', 'initialization_method', 'use_samples', 'use_mixup', 'total_rounds', 'convergence_rate', 'doesnt_converge_rate', 'collapse_rate', 'average_iterations', 'average_jaccard_score', 'bad_initialization_count']]
                    results_df.to_csv(file_path, index=False)

    return results_df

# EXPERIMENT #1 -- fixed dataset, make blobs
initialization_methods = ['run forward', 'two points', 'random']
use_samples_options = [True, False]
use_mixup_options = [True, False]
k_ratios = [None]  
file_path = "./exp_esults/blobs_fixed_dataset.csv"
results_df = experiment(initialization_methods, use_samples_options, use_mixup_options, k_ratios, file_path, dataset='blobs', variable=False)


# EXPERIMENT #2 -- variable dataset, make blobs
initialization_methods = ['run forward', 'two points', 'random']
use_samples_options = [True, False]
use_mixup_options = [True, False]
k_ratios = [None]  
file_path = "./exp_results/blobs_variable_dataset.csv"
results_df = experiment(initialization_methods, use_samples_options, use_mixup_options, k_ratios, file_path, dataset='blobs', variable=True)

# EXPERIMENT #3 -- fixed dataset, make moons
initialization_methods = ['run forward', 'two points', 'random']
use_samples_options = [True, False]
use_mixup_options = [True, False]
k_ratios = [0.1,0.2,0.3,0.4,0.5]  # tested various k ratio configurations 
file_path = './exp_results/moons_fixed_dataset_exp.csv'
results_df = experiment(initialization_methods, use_samples_options, use_mixup_options, k_ratios, file_path, dataset='moons', variable=False)


# EXPERIMENT #4 -- variable dataset, make moons
initialization_methods = ['run forward', 'two points', 'random']
use_samples_options = [True, False]
use_mixup_options = [True, False]
k_ratios = [0.1,0.2,0.3,0.4,0.5]  
file_path = './exp_results/moons_variable_dataset_exp.csv'
results_df = experiment(initialization_methods, use_samples_options, use_mixup_options, k_ratios, file_path, dataset='moons', variable=True)
