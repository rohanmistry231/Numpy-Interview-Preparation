import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Indexing and Slicing]
# Learn how to access and manipulate NumPy arrays using indexing and slicing.
# Covers basic indexing, boolean indexing, fancy indexing, and slicing for ML.

print("NumPy version:", np.__version__)

# %% [2. Basic Indexing]
# Access elements and subarrays using indices.
if load_iris:
    iris = load_iris()
    data = iris.data
else:
    data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4]])

print("\nIris Array (first 3 rows):\n", data[:3])
print("Single Element (row 0, col 0):", data[0, 0])  # Sepal length of first sample
print("First Row:", data[0])  # All features of first sample
print("First Column:", data[:, 0])  # Sepal length for all samples

# %% [3. Slicing]
# Extract subarrays using slices.
subset = data[:5, 1:3]  # First 5 rows, columns 1 and 2
print("\nSliced Subarray (first 5 rows, cols 1-2):\n", subset)

# Step slicing for downsampling
downsampled = data[::2, :]  # Every other row
print("\nDownsampled Array (every other row):\n", downsampled[:3])

# %% [4. Boolean Indexing]
# Filter arrays based on conditions.
sepal_length = data[:, 0]
long_sepal = data[sepal_length > 6.0]  # Samples with sepal length > 6.0
print("\nSamples with Sepal Length > 6.0:\n", long_sepal[:3])

# Combine conditions
mask = (sepal_length > 5.0) & (data[:, 2] < 2.0)  # Sepal length > 5.0 and petal length < 2.0
filtered = data[mask]
print("\nFiltered Samples (sepal > 5.0, petal < 2.0):\n", filtered)

# %% [5. Fancy Indexing]
# Use arrays of indices to select elements.
rows = np.array([0, 2, 4])
cols = np.array([1, 3])
selected = data[rows, cols]  # Elements at (0,1), (2,3), (4,3)
print("\nFancy Indexing (selected elements):\n", selected)

# Select specific rows
selected_rows = data[[0, 10, 20]]
print("\nSelected Rows (0, 10, 20):\n", selected_rows)

# %% [6. Visualizing Indexing]
# Visualize filtered data.
if load_iris:
    plt.figure(figsize=(8, 4))
    plt.scatter(data[:, 0], data[:, 2], c='blue', alpha=0.5, label='All Samples')
    plt.scatter(long_sepal[:, 0], long_sepal[:, 2], c='red', label='Sepal Length > 6.0')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Iris Dataset: Boolean Indexing')
    plt.legend()
    plt.savefig('indexing_scatter.png')

# %% [7. Practical ML Application]
# Use indexing to prepare ML features.
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = np.random.randint(0, 2, 100)  # Binary labels
positive_samples = X[y == 1]  # Select samples with label 1
print("\nML Dataset: Positive Samples Shape:", positive_samples.shape)
print("First 3 Positive Samples:\n", positive_samples[:3])

# %% [8. Interview Scenario: Indexing]
# Discuss indexing for ML data selection.
print("\nInterview Scenario: Indexing")
print("Q: How would you filter a dataset for ML preprocessing?")
print("A: Use boolean indexing for conditions (e.g., arr[arr[:, 0] > 5]) and slicing for subsets.")
print("Key: Boolean indexing is efficient for outlier removal and feature selection.")
print("Example: arr[arr[:, 0] > np.mean(arr[:, 0])] for above-average values.")