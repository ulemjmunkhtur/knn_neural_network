import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SimpleUnsupervisedNN(nn.Module):
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
        x = self.act_fn(self.layer1(x))
        x = self.act_fn(self.layer2(x))
        x = self.output(x)
        x = self.sigmoid(x)
        return x

def plot(cluster1, samples, save_path, plot_name="plot", iteration=0):
    plt.figure(figsize=(10, 5))
    plt.scatter(cluster1[:, 0], cluster1[:, 1], c="b", label="Cluster", s=50)
    plt.scatter(samples[:, 0], samples[:, 1], c="g", label="Samples", s=50)
    plt.gca().set_aspect("equal", adjustable='box')
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{plot_name}_iteration_{iteration}.png"))
    plt.show()

def train_model(cluster0, samples, mixup=False, n_epochs=20, batch_size=10, k=10, T=1, save_path=None, iteration=0):
    plot(cluster0, samples, save_path=save_path, plot_name="initial_plot", iteration=iteration)
    ds = np.concatenate([cluster0, samples])

    # Finding nearest neighbors & similarity
    neighbors = NearestNeighbors(n_neighbors=min(k, cluster0.shape[0])).fit(cluster0)
    distances, indices = neighbors.kneighbors(ds)
    exp_distances = np.exp(-T * np.square(distances))
    avg_exp_distance = np.mean(exp_distances, axis=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    y_ds = avg_exp_distance

    if mixup:
        ds, y_ds = augment_with_mixup(ds, y_ds)
    X_train = torch.tensor(ds, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_ds, dtype=torch.float32).reshape(-1, 1).to(device)
    dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Move the model to the device
    model = SimpleUnsupervisedNN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Finding the best model
    best_mse = float('inf')
    best_weights = None

    history = []

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Evaluating model after each epoch
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            mse = float(loss_fn(y_pred, y_train))
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}, MSE: {mse}')

    print(f'Best MSE: {best_mse}')

    model.load_state_dict(best_weights)

    # Plot training progress
    plt.plot(history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training Progress')
    if save_path:
        plt.savefig(os.path.join(save_path, f"training_progress_iteration_{iteration}.png"))
    plt.show()

    return model

def compare_models(model0, model1, dataset):
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

def plot_comparison(cluster0, cluster1, higher_score_model0, save_path, iteration):
    # Concatenate to form entire dataset
    test_set = np.concatenate([cluster0, cluster1])
    test_labels = np.array(['cluster0'] * len(cluster0) + ['cluster1'] * len(cluster1))

    # Plot comparison
    plt.figure(figsize=(10, 5))
    colors = np.where(higher_score_model0, 'b', 'r').flatten()
    plt.scatter(test_set[:, 0], test_set[:, 1], c=colors, s=50)
    plt.gca().set_aspect("equal", adjustable='box')
    plt.title("Model Comparison: Blue = cluster0_model, Red = cluster1_model")
    plt.savefig(os.path.join(save_path, f"comparison_plot_iteration_{iteration}.png"))
    plt.show()

    # Plot (but distinguishing)
    plt.figure(figsize=(10, 5))
    for label, marker in zip(['cluster0', 'cluster1'], ['o', 's']):
        indices = (test_labels == label)
        colors = np.where(higher_score_model0[indices], 'b', 'r').flatten()
        plt.scatter(test_set[indices, 0], test_set[indices, 1], c=colors, marker=marker, label=label, s=50)

    plt.gca().set_aspect("equal", adjustable='box')
    plt.title("Model Comparison (labeled): Blue = cluster0_model, Red = cluster1_model")
    plt.legend()
    plt.savefig(os.path.join(save_path, f"labeled_comparison_plot_iteration_{iteration}.png"))
    plt.show()

    # Numerical summary
    count_model0 = np.sum(higher_score_model0)
    count_model1 = len(higher_score_model0) - count_model0
    print(f"\n SUMMARY")
    print(f"cluster0's model has higher scores for {count_model0} points.")
    print(f"cluster1's model has higher scores for {count_model1} points.")

def mixup_data(x1, y1, x2, y2):
    lam = np.random.rand()  # lambda from uniform distribution between 0 and 1
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y

def augment_with_mixup(x, y):
    mixed_x = []
    mixed_y = []

    while len(mixed_x) < x.shape[0]:
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

def plot_meshgrid(model, X_in, title="", save_path=None, iteration=0):
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    len_x = len(x)

    X, Y = np.meshgrid(x, y)
    X_f = np.expand_dims(X.flatten('C'), axis=1)
    Y_f = np.expand_dims(Y.flatten('C'), axis=1)
    data = np.concatenate([X_f, Y_f], axis=1)

    m_out = model(torch.tensor(data).to(device).float()).detach().cpu().numpy()
    m_out = m_out.reshape((len_x, -1), order='C')

    plt.pcolor(X, Y, m_out)
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}_meshgrid_iteration_{iteration}.png"))
    plt.show()

    x_out = model(torch.tensor(X_in).to(device).float()).detach().cpu().numpy()
    plt.scatter(X_in[:, 0], X_in[:, 1], c=x_out, edgecolors='black')
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}_scatter_iteration_{iteration}.png"))
    plt.show()

    plt.plot(x_out)
    plt.title(f"Predicted Similarities using {title}")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{title}_similarities_iteration_{iteration}.png"))
    plt.show()

# Data generation
N, D = 200, 2
centers = np.array([[2, 2], [-2, -2]])
std_dev = 0.5

seed = 50

X, labels = make_blobs(n_samples=N*2, centers=centers, n_features=D, cluster_std=std_dev, random_state=2, shuffle=False)

midpoint = np.mean(centers, axis=0)
ds_std = np.std(centers)
samples = np.random.randn(N, D) * ds_std + midpoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(X, samples=None, num_iterations=5, type="two points", mixup=False, save_path="plots"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    Xtest = X.copy()
    for i in range(num_iterations):
        print(f"Iteration {i+1}")
        X = X.copy()
        labels = None

        # First iteration
        if i == 0:
            if type == "random":
                set_seed(seed)
                total_points = X.shape[0]
                labels = np.random.choice([0, 1], size=total_points)
            elif type == "run forward":
                # Run forward pass
                model1 = SimpleUnsupervisedNN().to(device)
                model1.eval()
                labels1 = model1(torch.tensor(X, dtype=torch.float32).to(device))
                labels1 = labels1.detach().cpu().numpy().flatten()  # 1D array

                model2 = SimpleUnsupervisedNN().to(device)
                model2.eval()
                labels2 = model2(torch.tensor(X, dtype=torch.float32).to(device))
                labels2 = labels2.detach().cpu().numpy().flatten()  # 1D array

                labels = (labels1 > labels2)
            elif type == "two points":
                idx = np.random.choice(X.shape[0], 2, replace=False)
                point1, point2 = X[idx[0]], X[idx[1]]
                dist_to_point1 = np.linalg.norm(X - point1, axis=1)
                dist_to_point2 = np.linalg.norm(X - point2, axis=1)
                labels = np.where(dist_to_point1 < dist_to_point2, 1, 0)
        else:
            # Assign labels based on the highest score from the previous iteration
            labels = higher_score_model0

        cluster0 = X[labels == 0]
        cluster1 = X[labels == 1]

        # Train models
        if samples is not None:
            trained_model0 = train_model(cluster0, samples, mixup=mixup, n_epochs=50, batch_size=10, k=200, T=1, save_path=save_path, iteration=i+1)
            trained_model1 = train_model(cluster1, samples, mixup=mixup, n_epochs=50, batch_size=10, k=200, T=1, save_path=save_path, iteration=i+1)
        else:
            trained_model0 = train_model(cluster1, cluster0, mixup=mixup, n_epochs=50, batch_size=10, k=200, T=1, save_path=save_path, iteration=i+1)
            trained_model1 = train_model(cluster0, cluster1, mixup=mixup, n_epochs=50, batch_size=10, k=200, T=1, save_path=save_path, iteration=i+1)

        # Compare models on the original data
        higher_score_model0 = compare_models(trained_model0, trained_model1, X)

        # Plot comparison
        plot_comparison(cluster0, cluster1, higher_score_model0, save_path=save_path, iteration=i+1)

    plot_meshgrid(trained_model0, X, "model0", save_path=save_path, iteration=i+1)
    plot_meshgrid(trained_model1, X, "model1", save_path=save_path, iteration=i+1)

    return trained_model0, trained_model1

save_path = "two_points_samples"
trained_model0, trained_model1 = test(X, samples, num_iterations=5, type="two points", mixup=False, save_path=save_path)