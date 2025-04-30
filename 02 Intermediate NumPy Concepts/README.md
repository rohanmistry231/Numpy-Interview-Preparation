# 🏋️ Intermediate NumPy Concepts (`numpy`)

## 📖 Introduction
NumPy’s intermediate concepts build on beginner skills, focusing on linear algebra, tensor operations, statistical computations, and ML algorithm implementation for AI and machine learning. This section covers **Linear Algebra for ML**, **Tensor Operations**, **Statistical Computations**, and **Implementing ML Algorithms**, with practical examples and interview insights to deepen your NumPy proficiency.

## 🎯 Learning Objectives
- Perform matrix operations and linear algebra for ML tasks.
- Manipulate multi-dimensional tensors for deep learning workflows.
- Compute statistical metrics and augment data for ML analysis.
- Implement ML algorithms (e.g., linear regression, K-means) from scratch.

## 🔑 Key Concepts
- **Linear Algebra for ML**:
  - Matrix operations (`np.dot`, `np.matmul`, `np.transpose`).
  - Solving linear systems (`np.linalg.solve`).
  - Eigenvalues/vectors (`np.linalg.eig`) and SVD (`np.linalg.svd`).
- **Tensor Operations**:
  - Multi-dimensional arrays (3D+ tensors).
  - Reshaping/transposing (`np.reshape`, `np.moveaxis`, `np.swapaxes`).
  - Contractions/reductions (`np.tensordot`, `np.sum`).
- **Statistical Computations**:
  - Descriptive statistics (`np.mean`, `np.median`, `np.var`).
  - Correlation/covariance (`np.corrcoef`, `np.cov`).
  - Random sampling (`np.random.choice`).
- **Implementing ML Algorithms**:
  - Linear regression with normal equations.
  - Logistic regression with gradient descent.
  - K-means clustering from scratch.

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`linear_algebra_ml.py`**:
   - Performs matrix operations (`np.dot`, `np.matmul`) on Iris data.
   - Solves linear systems and computes eigenvalues/SVD for dimensionality reduction.
   - Visualizes SVD-reduced data and covariance matrix heatmap.

   Example code:
   ```python
   import numpy as np
   X = np.random.rand(100, 4)
   U, S, Vt = np.linalg.svd(X, full_matrices=False)
   X_reduced = np.dot(U[:, :2], np.diag(S[:2]))
   ```

2. **`tensor_operations.py`**:
   - Creates 3D/4D tensors for image data and reshapes/transposes them.
   - Performs tensor contractions (`np.tensordot`) and reductions (`np.sum`).
   - Visualizes tensor slices and reduction results.

   Example code:
   ```python
   import numpy as np
   tensor = np.random.rand(10, 32, 32)
   reshaped = np.reshape(tensor, (10, 32 * 32))
   ```

3. **`statistical_computations.py`**:
   - Computes descriptive statistics (`np.mean`, `np.var`) and correlation/covariance on Iris data.
   - Augments data with random sampling (`np.random.choice`).
   - Visualizes correlation matrix and augmented data distribution.

   Example code:
   ```python
   import numpy as np
   X = np.random.rand(150, 4)
   corr_matrix = np.corrcoef(X.T)
   ```

4. **`implementing_ml_algorithms.py`**:
   - Implements linear regression (normal equations), logistic regression (gradient descent), and K-means clustering.
   - Uses synthetic or blob data for simplicity.
   - Visualizes linear regression fit and K-means clusters.

   Example code:
   ```python
   import numpy as np
   X = np.random.rand(100, 2)
   X_b = np.c_[np.ones((100, 1)), X]
   theta = np.linalg.solve(np.dot(X_b.T, X_b), np.dot(X_b.T, y))
   ```

## 🛠️ Practical Tasks
1. **Linear Algebra**:
   - Compute the covariance matrix of a dataset and find its eigenvalues.
   - Apply SVD to reduce a dataset to 2 dimensions.
2. **Tensor Operations**:
   - Reshape a 3D tensor (e.g., image batch) for a dense layer.
   - Perform a tensor contraction between two tensors.
3. **Statistical Computations**:
   - Calculate feature correlations and select the most predictive feature.
   - Augment a dataset with random sampling.
4. **ML Algorithms**:
   - Implement linear regression for a synthetic dataset.
   - Code K-means clustering and visualize the results.

## 💡 Interview Tips
- **Common Questions**:
  - How does SVD enable PCA in ML?
  - What’s the difference between `np.dot` and `np.matmul`?
  - How would you implement logistic regression with NumPy?
  - Why use random sampling for data augmentation?
- **Tips**:
  - Explain SVD’s role in dimensionality reduction (e.g., `U @ np.diag(S)` for PCA).
  - Highlight gradient descent’s iterative nature for logistic regression.
  - Be ready to code linear regression or K-means from scratch.
- **Coding Tasks**:
  - Implement PCA using SVD.
  - Code gradient descent for logistic regression.
  - Compute a correlation matrix for feature selection.

## 📚 Resources
- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [NumPy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html)
- [SciPy Lecture Notes: NumPy for ML](https://scipy-lectures.org/intro/numpy/index.html)
- [NumPy Array Manipulation](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)
- [Kaggle: Machine Learning with NumPy](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781491957653/)