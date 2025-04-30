import numpy as np
import matplotlib.pyplot as plt
try:
    import tensorflow as tf
    import torch
    from sklearn.preprocessing import StandardScaler
except ImportError:
    tf, torch, StandardScaler = None, None, None

# %% [1. Introduction to Integration with ML Frameworks]
# Learn to integrate NumPy with TensorFlow, PyTorch, and scikit-learn.
# Covers tensor conversion, data pipelines, and preprocessing.

print("NumPy version:", np.__version__)

# %% [2. Converting NumPy Arrays to Tensors]
# Convert to TensorFlow and PyTorch tensors.
np.random.seed(42)
X_np = np.random.rand(100, 5)

if tf:
    X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)
    print("\nTensorFlow Tensor Shape:", X_tf.shape)
else:
    print("\nTensorFlow not available; skipping.")

if torch:
    X_torch = torch.from_numpy(X_np).float()
    print("PyTorch Tensor Shape:", X_torch.shape)
else:
    print("PyTorch not available; skipping.")

# %% [3. NumPy as a Backend for Data Pipelines]
# Create a TensorFlow data pipeline from NumPy arrays.
if tf:
    y_np = np.random.randint(0, 2, 100)
    dataset = tf.data.Dataset.from_tensor_slices((X_np, y_np)).batch(32).shuffle(100)
    for X_batch, y_batch in dataset.take(1):
        print("\nTensorFlow Dataset Batch Shapes:", X_batch.shape, y_batch.shape)
else:
    print("\nTensorFlow not available; skipping pipeline.")

# %% [4. Interfacing with scikit-learn]
# Use NumPy with scikit-learn preprocessing.
if StandardScaler:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    print("\nScikit-learn Scaled Features Shape:", X_scaled.shape)
    print("First 3 Scaled Samples:\n", X_scaled[:3])
else:
    X_scaled = (X_np - np.mean(X_np, axis=0)) / np.std(X_np, axis=0)
    print("\nScikit-learn not available; using NumPy scaling.")
    print("First 3 Scaled Samples:\n", X_scaled[:3])

# %% [5. Visualizing Integration]
# Visualize scaled features.
plt.figure(figsize=(8, 4))
plt.scatter(X_np[:, 0], X_np[:, 1], c='blue', alpha=0.5, label='Original')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='red', alpha=0.5, label='Scaled')
plt.title('Original vs. Scaled Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.savefig('integration_ml_scaled.png')

# %% [6. Practical ML Application]
# Prepare a NumPy dataset for a deep learning model.
np.random.seed(42)
X_ml = np.random.rand(1000, 10)
y_ml = np.random.randint(0, 2, 1000)
if tf:
    dataset_ml = tf.data.Dataset.from_tensor_slices((X_ml, y_ml)).batch(64)
    print("\nDeep Learning Dataset:")
    print("Batch Size: 64")
else:
    print("\nTensorFlow not available; skipping deep learning dataset.")

# %% [7. Interview Scenario: Framework Integration]
# Discuss NumPy integration with ML frameworks.
print("\nInterview Scenario: Framework Integration")
print("Q: How do you prepare NumPy data for TensorFlow?")
print("A: Convert to tensors with tf.convert_to_tensor, use tf.data.Dataset.")
print("Key: Ensures compatibility with ML framework APIs.")
print("Example: dataset = tf.data.Dataset.from_tensor_slices((X_np, y_np)).batch(32).")