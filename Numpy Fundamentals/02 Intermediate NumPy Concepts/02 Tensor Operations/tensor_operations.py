import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Tensor Operations]
# Learn NumPy's tensor operations for ML tasks.
# Covers multi-dimensional arrays, reshaping, transposing, and contractions.

print("NumPy version:", np.__version__)

# %% [2. Multi-dimensional Arrays]
# Create 3D tensor (e.g., batch of images).
np.random.seed(42)
tensor_3d = np.random.rand(10, 32, 32)  # 10 samples, 32x32 images
print("\n3D Tensor Shape:", tensor_3d.shape)
print("First Sample (first 3x3):\n", tensor_3d[0, :3, :3])

# 4D tensor (e.g., batch of RGB images)
tensor_4d = np.random.rand(5, 3, 64, 64)  # 5 samples, 3 channels, 64x64
print("\n4D Tensor Shape:", tensor_4d.shape)

# %% [3. Tensor Reshaping and Transposing]
# Reshape tensor for ML input.
reshaped_tensor = np.reshape(tensor_3d, (10, 32 * 32))  # Flatten images
print("\nReshaped Tensor Shape:", reshaped_tensor.shape)
print("First Sample (first 10 elements):\n", reshaped_tensor[0, :10])

# Transpose tensor
transposed_tensor = np.transpose(tensor_3d, (1, 2, 0))  # Swap axes
print("\nTransposed Tensor Shape:", transposed_tensor.shape)

# Moveaxis and swapaxes
moved_tensor = np.moveaxis(tensor_4d, 1, 3)  # Move channel axis
print("\nMoved Tensor Shape:", moved_tensor.shape)

# %% [4. Tensor Contractions and Reductions]
# Tensor contraction with tensordot
tensor_a = np.random.rand(5, 3, 4)
tensor_b = np.random.rand(4, 2)
contracted = np.tensordot(tensor_a, tensor_b, axes=([2], [0]))
print("\nTensor Contraction Shape:", contracted.shape)

# Reduction with sum
sum_tensor = np.sum(tensor_3d, axis=(1, 2))  # Sum over image dimensions
print("\nSum Reduction Shape:", sum_tensor.shape)
print("Sum Values:", sum_tensor)

# %% [5. Visualizing Tensors]
# Visualize a 3D tensor slice.
plt.figure(figsize=(6, 4))
plt.imshow(tensor_3d[0], cmap='gray')
plt.title('3D Tensor: First Image Slice')
plt.colorbar()
plt.savefig('tensor_operations_image.png')

# Visualize reduction results
plt.figure(figsize=(8, 4))
plt.bar(range(len(sum_tensor)), sum_tensor)
plt.title('Sum Reduction of 3D Tensor')
plt.xlabel('Sample Index')
plt.ylabel('Sum Value')
plt.savefig('tensor_operations_reduction.png')

# %% [6. Practical ML Application]
# Prepare tensor for CNN input.
np.random.seed(42)
images = np.random.rand(20, 1, 28, 28)  # 20 grayscale images, 28x28
images_reshaped = np.reshape(images, (20, 28 * 28))  # Flatten for dense layer
print("\nCNN Input Preparation:")
print("Original Tensor Shape:", images.shape)
print("Reshaped Tensor Shape:", images_reshaped.shape)

# %% [7. Interview Scenario: Tensor Reshaping]
# Discuss reshaping for deep learning.
print("\nInterview Scenario: Tensor Reshaping")
print("Q: How would you prepare a tensor for a neural network?")
print("A: Reshape to match input layer (e.g., flatten images with np.reshape).")
print("Key: Ensure shape compatibility with model architecture.")
print("Example: np.reshape(tensor, (n_samples, height * width)) for dense layers.")