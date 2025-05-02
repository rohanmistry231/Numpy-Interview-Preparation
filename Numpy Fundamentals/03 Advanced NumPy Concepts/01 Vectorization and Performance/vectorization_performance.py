import numpy as np
import matplotlib.pyplot as plt
import time
try:
    from scipy import sparse
except ImportError:
    sparse = None

# %% [1. Introduction to Vectorization and Performance]
# Learn advanced NumPy techniques for performance optimization.
# Covers vectorization, memory-efficient computations, and profiling.

print("NumPy version:", np.__version__)

# %% [2. Replacing Loops with Vectorized Operations]
# Compare loop vs. vectorized operations.
np.random.seed(42)
X = np.random.rand(10000, 100)  # Large dataset
y = np.random.rand(10000)

# Loop-based dot product
start_time = time.time()
result_loop = np.zeros(100)
for i in range(100):
    result_loop[i] = np.sum(X[:, i] * y)
loop_time = time.time() - start_time

# Vectorized dot product
start_time = time.time()
result_vectorized = np.dot(X.T, y)
vectorized_time = time.time() - start_time
print("\nLoop Time:", loop_time, "seconds")
print("Vectorized Time:", vectorized_time, "seconds")
print("Speedup:", loop_time / vectorized_time)

# Verify results
print("Results Match:", np.allclose(result_loop, result_vectorized))

# %% [3. Memory-efficient Computations]
# Use np.memmap for large datasets.
large_array = np.memmap('large_array.dat', dtype='float32', mode='w+', shape=(10000, 1000))
large_array[:] = np.random.rand(10000, 1000)
print("\nMemory-mapped Array Shape:", large_array.shape)

# Stride tricks for sliding windows
from numpy.lib.stride_tricks import as_strided
X_small = np.random.rand(100, 10)
window_size = 3
strided = as_strided(X_small, shape=(X_small.shape[0] - window_size + 1, window_size, X_small.shape[1]),
                     strides=(X_small.strides[0], X_small.strides[0], X_small.strides[1]))
print("\nStrided Array Shape (sliding windows):", strided.shape)

# %% [4. Profiling and Optimization]
# Profile a computation-heavy operation.
def compute_distances(X):
    return np.sqrt(((X[:, np.newaxis] - X)**2).sum(axis=2))

X_profile = np.random.rand(1000, 5)
start_time = time.time()
distances = compute_distances(X_profile)
profile_time = time.time() - start_time
print("\nDistance Computation Time:", profile_time, "seconds")

# %% [5. Visualizing Performance]
# Plot loop vs. vectorized times.
plt.figure(figsize=(8, 4))
plt.bar(['Loop', 'Vectorized'], [loop_time, vectorized_time], color=['red', 'green'])
plt.title('Loop vs. Vectorized Performance')
plt.ylabel('Time (seconds)')
plt.savefig('vectorization_performance_bar.png')

# Visualize distance matrix
plt.figure(figsize=(6, 4))
plt.imshow(distances, cmap='viridis')
plt.colorbar()
plt.title('Distance Matrix')
plt.savefig('vectorization_performance_distances.png')

# %% [6. Practical ML Application]
# Optimize a feature scaling operation.
np.random.seed(42)
X_ml = np.random.rand(5000, 50)
start_time = time.time()
X_scaled = (X_ml - np.mean(X_ml, axis=0)) / np.std(X_ml, axis=0)
scaling_time = time.time() - start_time
print("\nML Feature Scaling Time:", scaling_time, "seconds")
print("Scaled Features Shape:", X_scaled.shape)

# %% [7. Interview Scenario: Vectorization]
# Discuss vectorization benefits.
print("\nInterview Scenario: Vectorization")
print("Q: Why use vectorized operations in ML?")
print("A: Vectorization replaces slow loops with optimized C-based operations.")
print("Key: Improves performance for large datasets.")
print("Example: np.dot(X.T, y) vs. loop-based summation.")