# 🌱 Beginner NumPy Concepts (`numpy`)

## 📖 Introduction
NumPy is the cornerstone of numerical computing in Python, essential for AI and machine learning (ML) data manipulation and preprocessing. This section introduces the fundamentals of NumPy, focusing on array creation, indexing, operations, and ML preprocessing. It covers **Array Creation and Properties**, **Indexing and Slicing**, **Basic Operations**, and **Data Preprocessing for ML**, with practical examples and interview insights tailored to beginners.

## 🎯 Learning Objectives
- Create and manipulate NumPy arrays for ML datasets.
- Access and filter data using indexing and slicing.
- Perform element-wise operations, broadcasting, and universal functions (ufuncs).
- Preprocess ML datasets with normalization, standardization, and train/test splitting.

## 🔑 Key Concepts
- **Array Creation and Properties**:
  - Create arrays with `np.array`, `np.zeros`, `np.ones`, `np.random`.
  - Understand attributes (`shape`, `dtype`, `ndim`) and reshaping (`np.reshape`, `np.ravel`).
- **Indexing and Slicing**:
  - Use basic indexing (`arr[0]`), slicing (`arr[:5, 1:3]`), boolean, and fancy indexing.
- **Basic Operations**:
  - Perform element-wise operations (e.g., `arr + 1`), broadcasting, and ufuncs (`np.sin`, `np.mean`).
- **Data Preprocessing for ML**:
  - Load datasets (`np.loadtxt`), normalize/standardize features, and split train/test sets.

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`array_creation_properties.py`**:
   - Creates arrays from Iris data and synthetic datasets (`np.array`, `np.random.rand`).
   - Explores attributes (`shape`, `dtype`) and reshaping for ML tasks.
   - Visualizes a random array as a heatmap and Iris feature distribution.

   Example code:
   ```python
   import numpy as np
   data = np.array([[5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2]])
   print("Shape:", data.shape)  # (2, 4)
   reshaped = np.reshape(data, (4, 2))
   ```

2. **`indexing_slicing.py`**:
   - Demonstrates basic indexing, slicing, boolean, and fancy indexing on Iris data.
   - Filters samples (e.g., sepal length > 6.0) and selects specific rows/columns.
   - Visualizes filtered data with a scatter plot.

   Example code:
   ```python
   import numpy as np
   data = np.array([[5.1, 3.5, 1.4], [4.9, 3.0, 1.4]])
   long_sepal = data[data[:, 0] > 5.0]
   ```

3. **`basic_operations.py`**:
   - Performs element-wise operations (e.g., scaling), broadcasting (e.g., bias addition), and ufuncs (e.g., `np.sin`).
   - Computes a mean squared error for ML.
   - Visualizes normalized feature distribution and true vs. predicted labels.

   Example code:
   ```python
   import numpy as np
   data = np.array([[5.1, 3.5], [4.9, 3.0]])
   normalized = data / np.max(data, axis=0)
   ```

4. **`data_preprocessing_ml.py`**:
   - Loads Iris or synthetic data, normalizes/standardizes features, and splits train/test sets.
   - Prepares a synthetic ML dataset with standardization.
   - Visualizes original vs. standardized features.

   Example code:
   ```python
   import numpy as np
   X = np.array([[5.1, 3.5], [4.9, 3.0]])
   X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
   ```

## 🛠️ Practical Tasks
1. **Array Creation**:
   - Create a 3x4 array of random values and print its shape and dtype.
   - Reshape a 1D array into a 2D matrix for ML input.
2. **Indexing and Slicing**:
   - Filter a dataset to select samples with a feature value above the mean.
   - Extract the first and last columns of a 2D array using slicing.
3. **Basic Operations**:
   - Normalize a dataset’s features to [0, 1] using broadcasting.
   - Compute the mean and standard deviation of each feature in a dataset.
4. **Data Preprocessing**:
   - Load a CSV dataset with `np.loadtxt` and standardize its features.
   - Split a dataset into 80% training and 20% testing sets.

## 💡 Interview Tips
- **Common Questions**:
  - How do you create a NumPy array for an ML dataset?
  - What is broadcasting, and how is it used in ML preprocessing?
  - How would you filter outliers using boolean indexing?
  - Why standardize features before training an ML model?
- **Tips**:
  - Explain broadcasting’s efficiency for feature scaling (e.g., `arr / np.max(arr)`).
  - Highlight boolean indexing for data cleaning (e.g., removing outliers).
  - Be ready to code normalization or train/test splitting with NumPy.
- **Coding Tasks**:
  - Create a 2D array and normalize its columns.
  - Filter a dataset using a boolean condition.
  - Split a NumPy array into train/test sets.

## 📚 Resources
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Basics](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [NumPy Array Creation](https://numpy.org/doc/stable/reference/routines.array-creation.html)
- [NumPy Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [SciPy Lecture Notes: NumPy](https://scipy-lectures.org/intro/numpy/index.html)
- [Kaggle: Python and NumPy](https://www.kaggle.com/learn/python)