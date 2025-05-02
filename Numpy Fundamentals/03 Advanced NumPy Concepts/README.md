# 🌐 Advanced NumPy Concepts (`numpy`)

## 📖 Introduction
NumPy’s advanced concepts focus on performance optimization, custom functions, integration with ML frameworks, and complex tensor manipulations for AI and machine learning. This section covers **Vectorization and Performance**, **Custom Functions and Ufuncs**, **Integration with ML Frameworks**, and **Advanced Tensor Manipulations**, with practical examples and interview insights to elevate your NumPy expertise.

## 🎯 Learning Objectives
- Optimize NumPy code with vectorization and memory-efficient techniques.
- Create custom functions and ufuncs for ML tasks.
- Integrate NumPy with TensorFlow, PyTorch, and scikit-learn.
- Manipulate tensors for deep learning and large-scale data.

## 🔑 Key Concepts
- **Vectorization and Performance**:
  - Replacing loops with vectorized operations.
  - Memory-efficient computations (`np.memmap`, `np.lib.stride_tricks`).
  - Profiling and optimizing code.
- **Custom Functions and Ufuncs**:
  - Writing ufuncs with `np.frompyfunc` or `numba`.
  - Vectorizing operations (`np.vectorize`).
  - Gradient computations for ML.
- **Integration with ML Frameworks**:
  - Converting arrays to tensors (`tf.convert_to_tensor`, `torch.from_numpy`).
  - NumPy-based data pipelines.
  - Interfacing with scikit-learn.
- **Advanced Tensor Manipulations**:
  - Batch processing for deep learning.
  - Sparse arrays (`scipy.sparse`).
  - Tensor decompositions (Tucker, CP).

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`vectorization_performance.py`**:
   - Compares loop vs. vectorized operations (`np.dot`).
   - Uses `np.memmap` and `np.lib.stride_tricks` for memory efficiency.
   - Visualizes performance (bar plot) and distance matrix (heatmap).

   Example code:
   ```python
   import numpy as np
   X = np.random.rand(10000, 100)
   result = np.dot(X.T, y)  # Vectorized
   ```

2. **`custom_functions_ufuncs.py`**:
   - Creates custom ufuncs (`np.frompyfunc`) and vectorized functions (`np.vectorize`).
   - Optimizes gradients with `numba`.
   - Visualizes custom activation and vectorized function outputs.

   Example code:
   ```python
   import numpy as np
   ufunc = np.frompyfunc(lambda x: np.clip(x, -1, 1), 1, 1)
   y = ufunc(X).astype(float)
   ```

3. **`integration_ml_frameworks.py`**:
   - Converts NumPy arrays to TensorFlow/PyTorch tensors.
   - Builds a TensorFlow data pipeline and uses scikit-learn preprocessing.
   - Visualizes original vs. scaled features.

   Example code:
   ```python
   import tensorflow as tf
   X_np = np.random.rand(100, 5)
   X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)
   ```

4. **`advanced_tensor_manipulations.py`**:
   - Normalizes image batches and processes sparse arrays.
   - Performs CP decomposition with `tensorly`.
   - Visualizes normalized images and sparse matrix structure.

   Example code:
   ```python
   import numpy as np
   images = np.random.rand(32, 3, 64, 64)
   images_normalized = images - np.mean(images, axis=(2, 3), keepdims=True)
   ```

## 🛠️ Practical Tasks
1. **Vectorization**:
   - Optimize a loop-based computation (e.g., dot product) with vectorization.
   - Process a large dataset with `np.memmap`.
2. **Custom Functions**:
   - Write a custom ufunc for a non-standard activation function.
   - Optimize a gradient computation with `numba`.
3. **Framework Integration**:
   - Convert a NumPy dataset to a TensorFlow `tf.data.Dataset`.
   - Preprocess features with scikit-learn and NumPy.
4. **Tensor Manipulations**:
   - Normalize a batch of images for a CNN.
   - Apply CP decomposition to compress a tensor.

## 💡 Interview Tips
- **Common Questions**:
  - Why is vectorization faster than loops in NumPy?
  - How do you optimize a NumPy computation with `numba`?
  - How do you integrate NumPy with TensorFlow/PyTorch?
  - What are the benefits of tensor decomposition in ML?
- **Tips**:
  - Explain vectorization’s use of C-based operations for speed.
  - Highlight `numba`’s JIT compilation for ML optimizations.
  - Be ready to code a data pipeline or tensor decomposition.
- **Coding Tasks**:
  - Vectorize a loop-based computation.
  - Convert a NumPy array to a TensorFlow dataset.
  - Implement a sparse matrix operation.

## 📚 Resources
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/performance.html)
- [NumPy and Numba](https://numba.pydata.org/)
- [TensorFlow Data Pipeline](https://www.tensorflow.org/guide/data)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Tensorly Documentation](http://tensorly.org/stable/)
- [SciPy Sparse Arrays](https://docs.scipy.org/doc/scipy/reference/sparse.html)