import numpy as np
import matplotlib.pyplot as plt
try:
    from numba import jit
except ImportError:
    jit = lambda x: x

# %% [1. Introduction to Custom Functions and Ufuncs]
# Learn to create custom NumPy functions for ML tasks.
# Covers np.frompyfunc, np.vectorize, numba, and gradient computations.

print("NumPy version:", np.__version__)

# %% [2. Writing Custom Ufuncs with np.frompyfunc]
# Create a custom activation function.
def custom_activation(x):
    return np.clip(x, -1, 1)

ufunc_activation = np.frompyfunc(custom_activation, 1, 1)
X = np.linspace(-5, 5, 100)
y_ufunc = ufunc_activation(X).astype(float)
print("\nCustom Ufunc Output (first 5):", y_ufunc[:5])

# %% [3. Vectorizing Complex Operations]
# Vectorize a non-trivial function.
def complex_function(x, threshold=0.5):
    return x**2 if x > threshold else np.sin(x)

vectorized_func = np.vectorize(complex_function)
X_complex = np.linspace(-2, 2, 100)
y_vectorized = vectorized_func(X_complex)
print("\nVectorized Function Output (first 5):", y_vectorized[:5])

# %% [4. Numba for Performance]
# Optimize a gradient computation with numba.
@jit(nopython=True)
def compute_gradient(X, y, theta):
    return np.dot(X.T, (np.dot(X, theta) - y)) / len(y)

np.random.seed(42)
X_grad = np.random.rand(1000, 5)
y_grad = np.random.rand(1000)
theta = np.random.rand(5)
gradient = compute_gradient(X_grad, y_grad, theta)
print("\nNumba Gradient Shape:", gradient.shape)
print("Gradient Values:", gradient)

# %% [5. Gradient Computations for ML]
# Compute gradients for a custom loss.
def custom_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def loss_gradient(X, y, theta):
    y_pred = np.dot(X, theta)
    return -2 * np.dot(X.T, (y - y_pred)) / len(y)

X_ml = np.random.rand(500, 3)
y_ml = np.random.rand(500)
theta_ml = np.random.rand(3)
grad = loss_gradient(X_ml, y_ml, theta_ml)
print("\nCustom Loss Gradient Shape:", grad.shape)
print("Gradient Values:", grad)

# %% [6. Visualizing Custom Functions]
# Plot custom activation function.
plt.figure(figsize=(8, 4))
plt.plot(X, y_ufunc, label='Custom Activation (clipped)')
plt.plot(X, X, '--', label='Input')
plt.title('Custom Ufunc: Clipped Activation')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.savefig('custom_functions_ufunc.png')

# Plot vectorized function
plt.figure(figsize=(8, 4))
plt.plot(X_complex, y_vectorized, label='Vectorized Function')
plt.title('Vectorized Complex Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.savefig('custom_functions_vectorized.png')

# %% [7. Practical ML Application]
# Apply custom ufunc to preprocess ML features.
np.random.seed(42)
X_features = np.random.rand(1000, 10) * 10 - 5
X_processed = ufunc_activation(X_features).astype(float)
print("\nML Feature Preprocessing:")
print("Processed Features Shape:", X_processed.shape)
print("First 3 Processed Samples:\n", X_processed[:3])

# %% [8. Interview Scenario: Numba Optimization]
# Discuss numba for ML performance.
print("\nInterview Scenario: Numba Optimization")
print("Q: How would you optimize a gradient computation in NumPy?")
print("A: Use numba's @jit to compile Python code to machine code.")
print("Key: Numba accelerates loops and numerical operations.")
print("Example: @jit def compute_gradient(X, y, theta): ...")