import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Basic Operations]
# Learn NumPy’s element-wise operations, broadcasting, and universal functions (ufuncs).
# Essential for ML computations like feature scaling and loss calculations.

print("NumPy version:", np.__version__)

# %% [2. Element-wise Operations]
# Perform arithmetic operations on arrays.
if load_iris:
    iris = load_iris()
    data = iris.data
else:
    data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4]])

scaled_data = data * 2  # Multiply all elements by 2
print("\nScaled Data (first 3 rows):\n", scaled_data[:3])

added_data = data + 10  # Add 10 to all elements
print("\nAdded Data (first 3 rows):\n", added_data[:3])

# Combine arrays
combined = data[:, 0] + data[:, 1]  # Sum of sepal length and width
print("\nSum of Sepal Length and Width (first 5):\n", combined[:5])

# %% [3. Broadcasting]
# Apply operations across arrays of different shapes.
bias = np.array([1, -1, 0, 0.5])  # Bias for each feature
biased_data = data + bias  # Broadcasting bias to all rows
print("\nBroadcasted Bias Data (first 3 rows):\n", biased_data[:3])

# Broadcasting scalar
normalized = data / np.max(data, axis=0)  # Normalize by column max
print("\nNormalized Data (first 3 rows):\n", normalized[:3])

# %% [4. Universal Functions (ufuncs)]
# Apply mathematical functions element-wise.
sin_data = np.sin(data)  # Sine of all elements
print("\nSine of Data (first 3 rows):\n", sin_data[:3])

exp_data = np.exp(data[:, 0])  # Exponential of sepal length
print("\nExponential of Sepal Length (first 5):\n", exp_data[:5])

mean_data = np.mean(data, axis=0)  # Mean of each feature
print("\nMean of Each Feature:", mean_data)

# %% [5. Visualizing Operations]
# Visualize normalized data distribution.
if load_iris:
    plt.figure(figsize=(8, 4))
    plt.hist(normalized[:, 0], bins=20, color='green', alpha=0.7)
    plt.title('Normalized Sepal Length Distribution')
    plt.xlabel('Normalized Value')
    plt.ylabel('Frequency')
    plt.savefig('operations_histogram.png')

# %% [6. Practical ML Application]
# Compute a loss function for ML.
np.random.seed(42)
y_true = np.random.randint(0, 2, 100)  # True binary labels
y_pred = np.random.rand(100)  # Predicted probabilities
mse = np.mean((y_true - y_pred) ** 2)  # Mean squared error
print("\nML Loss Calculation:")
print("Mean Squared Error:", mse)

# Visualize predictions vs. true labels
plt.figure(figsize=(8, 4))
plt.scatter(range(100), y_true, c='blue', label='True Labels', alpha=0.5)
plt.scatter(range(100), y_pred, c='red', label='Predictions', alpha=0.5)
plt.title('True vs. Predicted Labels')
plt.legend()
plt.savefig('operations_loss.png')

# %% [7. Interview Scenario: Broadcasting]
# Discuss broadcasting for ML computations.
print("\nInterview Scenario: Broadcasting")
print("Q: How does broadcasting simplify ML feature scaling?")
print("A: Broadcasting applies operations (e.g., normalization) to arrays without loops.")
print("Key: Ensures shape compatibility for efficient computations.")
print("Example: arr / np.max(arr, axis=0) normalizes columns without explicit iteration.")