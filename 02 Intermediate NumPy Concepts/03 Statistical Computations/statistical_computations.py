import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Statistical Computations]
# Learn NumPy's statistical tools for ML analysis.
# Covers descriptive statistics, correlation/covariance, and random sampling.

print("NumPy version:", np.__version__)

# %% [2. Descriptive Statistics]
# Compute statistics on Iris data.
if load_iris:
    iris = load_iris()
    X = iris.data
else:
    X = np.random.rand(150, 4)  # Synthetic data

mean = np.mean(X, axis=0)
median = np.median(X, axis=0)
variance = np.var(X, axis=0)
std = np.std(X, axis=0)
print("\nDescriptive Statistics (Iris Features):")
print("Mean:", mean)
print("Median:", median)
print("Variance:", variance)
print("Standard Deviation:", std)

# %% [3. Correlation and Covariance]
# Compute correlation and covariance matrices.
corr_matrix = np.corrcoef(X.T)
cov_matrix = np.cov(X.T)
print("\nCorrelation Matrix:\n", corr_matrix)
print("\nCovariance Matrix:\n", cov_matrix)

# %% [4. Random Sampling for Data Augmentation]
# Generate augmented data with random sampling.
np.random.seed(42)
indices = np.random.choice(X.shape[0], size=50, replace=True)
augmented_data = X[indices]
print("\nAugmented Data Shape:", augmented_data.shape)
print("First 3 Augmented Samples:\n", augmented_data[:3])

# %% [5. Visualizing Statistics]
# Visualize correlation matrix.
plt.figure(figsize=(6, 4))
plt.imshow(corr_matrix, cmap='coolwarm')
plt.colorbar()
plt.title('Correlation Matrix of Iris Features')
plt.savefig('statistical_computations_corr.png')

# Visualize augmented data distribution
plt.figure(figsize=(8, 4))
plt.hist(augmented_data[:, 0], bins=20, color='purple', alpha=0.7)
plt.title('Augmented Data: Feature 1 Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('statistical_computations_hist.png')

# %% [6. Practical ML Application]
# Use statistics for feature selection.
np.random.seed(42)
X_synthetic = np.random.rand(100, 3)
y_synthetic = np.random.randint(0, 2, 100)
correlations = np.array([np.corrcoef(X_synthetic[:, i], y_synthetic)[0, 1] for i in range(3)])
print("\nFeature Selection:")
print("Feature Correlations with Target:", correlations)
print("Selected Feature (highest correlation):", np.argmax(np.abs(correlations)))

# %% [7. Interview Scenario: Correlation Analysis]
# Discuss correlation for feature selection.
print("\nInterview Scenario: Correlation Analysis")
print("Q: How would you select features using NumPy?")
print("A: Compute np.corrcoef to find feature-target correlations, select high values.")
print("Key: High correlation indicates predictive power.")
print("Example: np.corrcoef(X.T, y) for feature-target correlations.")