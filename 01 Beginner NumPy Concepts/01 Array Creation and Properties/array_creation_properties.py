import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Array Creation and Properties]
# Learn how to create NumPy arrays and explore their properties for ML data handling.
# Covers np.array, np.zeros, np.ones, np.random, array attributes, and reshaping.

print("NumPy version:", np.__version__)

# %% [2. Creating Arrays]
# Create arrays using different methods.
# From a Python list (e.g., Iris features).
if load_iris:
    iris = load_iris()
    data = iris.data  # Shape: (150, 4)
else:
    # Fallback synthetic data
    data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4]])

array_from_list = np.array(data)
print("\nArray from List (Iris features):\n", array_from_list[:3])

# Zeros, ones, and random arrays
zeros_array = np.zeros((3, 4))  # 3x4 array of zeros
ones_array = np.ones((2, 5))    # 2x5 array of ones
random_array = np.random.rand(3, 3)  # 3x3 array of random values [0, 1)
print("\nZeros Array:\n", zeros_array)
print("\nOnes Array:\n", ones_array)
print("\nRandom Array:\n", random_array)

# %% [3. Array Attributes]
# Explore array properties: shape, dtype, ndim.
print("\nArray Attributes (Iris array):")
print("Shape:", array_from_list.shape)  # (n_samples, n_features)
print("Data Type:", array_from_list.dtype)
print("Dimensions:", array_from_list.ndim)

# Example with random integer array
int_array = np.random.randint(0, 10, size=(4, 3))
print("\nInteger Array:\n", int_array)
print("Shape:", int_array.shape)
print("Data Type:", int_array.dtype)
print("Dimensions:", int_array.ndim)

# %% [4. Reshaping and Flattening]
# Reshape arrays for ML tasks (e.g., flattening features).
reshaped_array = np.reshape(array_from_list, (50, 12))  # Reshape to 50x12
print("\nReshaped Array (50x12):\n", reshaped_array[:2])

flattened_array = np.ravel(array_from_list)  # Flatten to 1D
print("\nFlattened Array (first 10 elements):\n", flattened_array[:10])

# Example: Reshape for image-like data
image_array = np.random.rand(16, 16)  # Simulate 16x16 grayscale image
reshaped_image = np.reshape(image_array, (4, 64))  # Reshape to 4x64
print("\nImage Array Shape:", image_array.shape)
print("Reshaped Image Shape:", reshaped_image.shape)

# %% [5. Visualizing Arrays]
# Visualize a random 2D array as a heatmap.
plt.figure(figsize=(6, 4))
plt.imshow(random_array, cmap='viridis')
plt.colorbar()
plt.title('Random 2D Array Heatmap')
plt.savefig('array_creation_heatmap.png')

# Visualize Iris feature distribution
if load_iris:
    plt.figure(figsize=(8, 4))
    plt.hist(array_from_list[:, 0], bins=20, color='blue', alpha=0.7)
    plt.title('Iris Sepal Length Distribution')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.savefig('iris_sepal_histogram.png')

# %% [6. Practical ML Application]
# Create a synthetic dataset for ML classification.
np.random.seed(42)
X = np.random.rand(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Binary labels
print("\nSynthetic ML Dataset:")
print("Features (X) Shape:", X.shape)
print("Labels (y) Shape:", y.shape)
print("First 5 samples:\n", np.hstack((X[:5], y[:5].reshape(-1, 1))))

# %% [7. Interview Scenario: Array Creation]
# Discuss creating arrays for ML tasks.
print("\nInterview Scenario: Array Creation")
print("Q: How would you create a dataset for ML with NumPy?")
print("A: Use np.random for features, np.array for structured data, and np.reshape for correct shapes.")
print("Key: Ensure correct shape and dtype for ML model compatibility.")
print("Example: np.random.rand(100, 2) for 100 samples with 2 features.")