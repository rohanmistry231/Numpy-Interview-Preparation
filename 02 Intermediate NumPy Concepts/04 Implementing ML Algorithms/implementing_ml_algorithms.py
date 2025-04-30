import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import make_blobs
except ImportError:
    make_blobs = None

# %% [1. Introduction to Implementing ML Algorithms]
# Learn to implement ML algorithms with NumPy.
# Covers linear regression, logistic regression, and K-means clustering.

print("NumPy version:", np.__version__)

# %% [2. Linear Regression with Normal Equations]
# Implement linear regression on synthetic data.
np.random.seed(42)
if make_blobs:
    X, _ = make_blobs(n_samples=100, centers=1, n_features=2)
else:
    X = np.random.rand(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1  # Linear relationship
X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
theta = np.linalg.solve(np.dot(X_b.T, X_b), np.dot(X_b.T, y))  # Normal equations
print("\nLinear Regression Coefficients:", theta)

# Predict
y_pred = np.dot(X_b, theta)

# %% [3. Logistic Regression with Gradient Descent]
# Implement logistic regression.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X_log = X
y_log = (y > np.median(y)).astype(int)  # Binary labels
X_log_b = np.c_[np.ones((100, 1)), X_log]
theta_log = np.zeros(3)
lr = 0.1
for _ in range(1000):
    z = np.dot(X_log_b, theta_log)
    h = sigmoid(z)
    gradient = np.dot(X_log_b.T, (h - y_log)) / 100
    theta_log -= lr * gradient
print("\nLogistic Regression Coefficients:", theta_log)

# Predict
y_pred_log = sigmoid(np.dot(X_log_b, theta_log)) > 0.5

# %% [4. K-means Clustering]
# Implement K-means clustering.
if make_blobs:
    X_cluster, _ = make_blobs(n_samples=100, centers=3, n_features=2)
else:
    X_cluster = np.random.rand(100, 2) * 10
K = 3
centroids = X_cluster[np.random.choice(100, K, replace=False)]
for _ in range(10):
    distances = np.sqrt(((X_cluster - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    centroids = np.array([X_cluster[labels == k].mean(axis=0) for k in range(K)])
print("\nK-means Centroids:\n", centroids)

# %% [5. Visualizing ML Algorithms]
# Visualize linear regression predictions.
plt.figure(figsize=(8, 4))
plt.scatter(X[:, 0], y, c='blue', alpha=0.5, label='Data')
plt.plot(X[:, 0], y_pred, c='red', label='Linear Regression')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Linear Regression Fit')
plt.legend()
plt.savefig('ml_algorithms_linear.png')

# Visualize K-means clusters
plt.figure(figsize=(8, 4))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.legend()
plt.savefig('ml_algorithms_kmeans.png')

# %% [6. Practical ML Application]
# Evaluate linear regression performance.
mse = np.mean((y_pred - y)**2)
print("\nLinear Regression Performance:")
print("Mean Squared Error:", mse)

# %% [7. Interview Scenario: Gradient Descent]
# Discuss implementing gradient descent.
print("\nInterview Scenario: Gradient Descent")
print("Q: How would you implement logistic regression with NumPy?")
print("A: Use gradient descent to minimize loss, compute gradients with np.dot.")
print("Key: Sigmoid function and iterative updates are critical.")
print("Example: theta -= lr * np.dot(X.T, (sigmoid(X @ theta) - y)) / n.")