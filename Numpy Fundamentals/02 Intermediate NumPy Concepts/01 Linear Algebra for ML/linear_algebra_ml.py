import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Linear Algebra for ML]
# Learn NumPy's linear algebra tools for ML tasks.
# Covers matrix operations, solving linear systems, eigenvalues, and SVD.

print("NumPy version:", np.__version__)

# %% [2. Matrix Operations]
# Perform matrix operations using Iris data.
if load_iris:
    iris = load_iris()
    X = iris.data[:100]  # First 100 samples for simplicity
else:
    X = np.random.rand(100, 4)  # Synthetic data

# Matrix multiplication (dot product)
X_T = np.transpose(X)  # Transpose
X_TX = np.dot(X_T, X)  # X^T * X
print("\nX^T * X Matrix (4x4):\n", X_TX)

# Matrix multiplication with matmul
X_matmul = np.matmul(X_T, X)
print("\nX^T * X with matmul:\n", X_matmul)

# %% [3. Solving Linear Systems]
# Solve a linear system: Ax = b
A = np.array([[3, 1], [1, 2]])  # Coefficient matrix
b = np.array([9, 8])  # Constants
x = np.linalg.solve(A, b)  # Solve for x
print("\nLinear System Solution (Ax = b):")
print("A:\n", A)
print("b:", b)
print("x:", x)

# Verify solution
print("Verification (Ax):\n", np.dot(A, x))

# %% [4. Eigenvalues and Eigenvectors]
# Compute eigenvalues/vectors for covariance matrix.
cov_matrix = np.cov(X.T)  # Covariance of Iris features
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nCovariance Matrix Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# %% [5. Singular Value Decomposition (SVD)]
# Apply SVD for dimensionality reduction.
U, S, Vt = np.linalg.svd(X, full_matrices=False)
X_reduced = np.dot(U[:, :2], np.diag(S[:2]))  # Reduce to 2 dimensions
print("\nSVD Reduced Data Shape:", X_reduced.shape)
print("First 3 Reduced Samples:\n", X_reduced[:3])

# %% [6. Visualizing Linear Algebra]
# Visualize SVD-reduced data.
plt.figure(figsize=(8, 4))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='blue', alpha=0.5)
plt.title('SVD: Iris Data in 2D')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig('linear_algebra_svd.png')

# Visualize covariance matrix
plt.figure(figsize=(6, 4))
plt.imshow(cov_matrix, cmap='viridis')
plt.colorbar()
plt.title('Covariance Matrix Heatmap')
plt.savefig('linear_algebra_cov_matrix.png')

# %% [7. Practical ML Application]
# Use matrix operations for feature transformation.
np.random.seed(42)
X_synthetic = np.random.rand(100, 3)  # Synthetic data
W = np.random.rand(3, 2)  # Transformation matrix
X_transformed = np.dot(X_synthetic, W)  # Linear transformation
print("\nSynthetic ML Dataset:")
print("Transformed Features Shape:", X_transformed.shape)
print("First 3 Transformed Samples:\n", X_transformed[:3])

# %% [8. Interview Scenario: SVD for PCA]
# Discuss SVD for dimensionality reduction.
print("\nInterview Scenario: SVD for PCA")
print("Q: How would you implement PCA with NumPy?")
print("A: Use np.linalg.svd to decompose data, select top components.")
print("Key: SVD reduces dimensionality while preserving variance.")
print("Example: U, S, Vt = np.linalg.svd(X); X_reduced = U[:, :k] @ np.diag(S[:k]).")