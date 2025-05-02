import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Data Preprocessing for ML]
# Learn how to preprocess ML datasets with NumPy.
# Covers loading datasets, normalization, standardization, and train/test splitting.

print("NumPy version:", np.__version__)

# %% [2. Loading Datasets]
# Load Iris dataset or use synthetic CSV data.
if load_iris:
    iris = load_iris()
    X = iris.data
    y = iris.target
else:
    # Synthetic CSV-like data
    data = np.array([[5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], 
                     [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1]])
    X = data[:, :-1]  # Features
    y = data[:, -1].astype(int)  # Labels

print("\nLoaded Dataset:")
print("Features (X) Shape:", X.shape)
print("Labels (y) Shape:", y.shape)
print("First 3 Samples:\n", np.hstack((X[:3], y[:3].reshape(-1, 1))))

# %% [3. Normalization]
# Scale features to [0, 1] using min-max normalization.
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X_normalized = (X - X_min) / (X_max - X_min)
print("\nNormalized Features (first 3 rows):\n", X_normalized[:3])

# %% [4. Standardization]
# Scale features to mean=0, std=1.
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean) / X_std
print("\nStandardized Features (first 3 rows):\n", X_standardized[:3])

# %% [5. Train/Test Splitting]
# Split dataset into training and testing sets.
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
train_size = int(0.8 * X.shape[0])
train_idx, test_idx = indices[:train_size], indices[train_size:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print("\nTrain/Test Split:")
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# %% [6. Visualizing Preprocessing]
# Visualize standardized vs. original features.
if load_iris:
    plt.figure(figsize=(8, 4))
    plt.scatter(X[:, 0], X[:, 2], c='blue', alpha=0.5, label='Original')
    plt.scatter(X_standardized[:, 0], X_standardized[:, 2], c='red', alpha=0.5, label='Standardized')
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.title('Original vs. Standardized Iris Features')
    plt.legend()
    plt.savefig('preprocessing_scatter.png')

# %% [7. Practical ML Application]
# Prepare a synthetic dataset for ML classification.
np.random.seed(42)
X_synthetic = np.random.rand(100, 2)  # 100 samples, 2 features
y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] > 1).astype(int)
X_synthetic_std = (X_synthetic - np.mean(X_synthetic, axis=0)) / np.std(X_synthetic, axis=0)
train_idx = np.random.choice(100, 80, replace=False)
test_idx = np.setdiff1d(np.arange(100), train_idx)
X_train_synthetic = X_synthetic_std[train_idx]
X_test_synthetic = X_synthetic_std[test_idx]
print("\nSynthetic ML Dataset:")
print("Standardized X_train Shape:", X_train_synthetic.shape)
print("X_test Shape:", X_test_synthetic.shape)

# %% [8. Interview Scenario: Preprocessing]
# Discuss preprocessing for ML pipelines.
print("\nInterview Scenario: Preprocessing")
print("Q: How would you preprocess a dataset for ML with NumPy?")
print("A: Normalize or standardize features, split into train/test sets.")
print("Key: Standardization (mean=0, std=1) is common for ML algorithms.")
print("Example: (X - np.mean(X, axis=0)) / np.std(X, axis=0) for standardization.")