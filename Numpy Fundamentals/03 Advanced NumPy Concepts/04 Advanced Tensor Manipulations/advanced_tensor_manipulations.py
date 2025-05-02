import numpy as np
import matplotlib.pyplot as plt
try:
    from scipy import sparse
    import tensorly as tl
except ImportError:
    sparse, tl = None, None

# %% [1. Introduction to Advanced Tensor Manipulations]
# Learn advanced NumPy tensor operations for ML.
# Covers batch processing, sparse arrays, and tensor decompositions.

print("NumPy version:", np.__version__)

# %% [2. Batch Processing for Deep Learning]
# Process image batches for CNNs.
np.random.seed(42)
images = np.random.rand(32, 3, 64, 64)  # 32 RGB images, 64x64
batch_mean = np.mean(images, axis=(2, 3), keepdims=True)
images_normalized = images - batch_mean  # Batch normalization
print("\nBatch Processed Images Shape:", images_normalized.shape)
print("Batch Mean Shape:", batch_mean.shape)

# %% [3. Sparse Arrays for Large-scale Data]
# Use sparse arrays for memory efficiency.
if sparse:
    sparse_matrix = sparse.csr_matrix(np.random.rand(1000, 1000) > 0.9)  # 90% sparsity
    print("\nSparse Matrix Shape:", sparse_matrix.shape)
    print("Non-zero Elements:", sparse_matrix.nnz)
else:
    print("\nScipy.sparse not available; skipping sparse matrix.")

# %% [4. Tensor Decompositions]
# Perform CP decomposition for compression.
if tl:
    tensor = np.random.rand(10, 20, 30)
    factors = tl.decomposition.parafac(tensor, rank=5)
    reconstructed = tl.kruskal_to_tensor(factors)
    error = np.mean((tensor - reconstructed)**2)
    print("\nCP Decomposition Error:", error)
else:
    print("\nTensorly not available; using simple sum reduction.")
    tensor = np.random.rand(10, 20, 30)
    reduced = np.sum(tensor, axis=2)
    print("Reduced Tensor Shape:", reduced.shape)

# %% [5. Visualizing Tensor Manipulations]
# Visualize normalized image slice.
plt.figure(figsize=(6, 4))
plt.imshow(images_normalized[0, 0], cmap='gray')
plt.title('Normalized Image Slice (Batch Processing)')
plt.colorbar()
plt.savefig('tensor_manipulations_image.png')

# Visualize sparse matrix (if available)
if sparse:
    plt.figure(figsize=(6, 4))
    plt.spy(sparse_matrix, markersize=1)
    plt.title('Sparse Matrix Structure')
    plt.savefig('tensor_manipulations_sparse.png')

# %% [6. Practical ML Application]
# Prepare a large tensor for deep learning.
np.random.seed(42)
large_tensor = np.random.rand(100, 3, 128, 128)  # 100 RGB images
large_tensor_flattened = np.reshape(large_tensor, (100, -1))  # Flatten for dense layer
print("\nDeep Learning Tensor Preparation:")
print("Original Tensor Shape:", large_tensor.shape)
print("Flattened Tensor Shape:", large_tensor_flattened.shape)

# %% [7. Interview Scenario: Tensor Decomposition]
# Discuss tensor decomposition for ML.
print("\nInterview Scenario: Tensor Decomposition")
print("Q: How would you compress a tensor for ML?")
print("A: Use CP decomposition to reduce dimensionality with tensorly.")
print("Key: Preserves structure while reducing memory.")
print("Example: factors = tl.decomposition.parafac(tensor, rank=5).")