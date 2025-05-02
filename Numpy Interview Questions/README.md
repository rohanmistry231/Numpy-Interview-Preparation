# NumPy Interview Questions for AI/ML Roles

This README provides 170 NumPy interview questions tailored for AI/ML roles, focusing on numerical computing with NumPy in Python. The questions cover **core NumPy concepts** (e.g., array creation, operations, indexing, broadcasting, linear algebra) and their applications in AI/ML tasks like data preprocessing, feature engineering, and model input preparation. Questions are categorized by topic and divided into **Basic**, **Intermediate**, and **Advanced** levels to support candidates preparing for roles requiring NumPy in AI/ML workflows.

## Array Creation and Manipulation

### Basic
1. **What is NumPy, and why is it important in AI/ML?**  
   NumPy provides efficient array operations for numerical computing in AI/ML.  
   ```python
   import numpy as np
   array = np.array([1, 2, 3])
   ```

2. **How do you create a NumPy array from a Python list?**  
   Converts lists to arrays for fast computation.  
   ```python
   import numpy as np
   list_data = [1, 2, 3]
   array = np.array(list_data)
   ```

3. **How do you create a NumPy array with zeros or ones?**  
   Initializes arrays for placeholders.  
   ```python
   zeros = np.zeros((2, 3))
   ones = np.ones((2, 3))
   ```

4. **What is the role of `np.arange` in NumPy?**  
   Creates arrays with a range of values.  
   ```python
   array = np.arange(0, 10, 2)
   ```

5. **How do you create a NumPy array with random values?**  
   Generates random data for testing.  
   ```python
   random_array = np.random.rand(2, 3)
   ```

6. **How do you reshape a NumPy array?**  
   Changes array dimensions for ML inputs.  
   ```python
   array = np.array([1, 2, 3, 4, 5, 6])
   reshaped = array.reshape(2, 3)
   ```

#### Intermediate
7. **Write a function to create a 2D NumPy array with a given shape.**  
   Initializes arrays dynamically.  
   ```python
   def create_2d_array(rows, cols, fill=0):
       return np.full((rows, cols), fill)
   ```

8. **How do you create a NumPy array with evenly spaced values?**  
   Uses `linspace` for uniform intervals.  
   ```python
   array = np.linspace(0, 10, 5)
   ```

9. **Write a function to initialize a NumPy array with random integers.**  
   Generates integer arrays for simulations.  
   ```python
   def random_int_array(shape, low, high):
       return np.random.randint(low, high, shape)
   ```

10. **How do you create a diagonal matrix in NumPy?**  
    Initializes matrices for linear algebra.  
    ```python
    diag_matrix = np.diag([1, 2, 3])
    ```

11. **Write a function to visualize a NumPy array as a heatmap.**  
    Displays array values graphically.  
    ```python
    import matplotlib.pyplot as plt
    def plot_heatmap(array):
        plt.imshow(array, cmap='viridis')
        plt.colorbar()
        plt.savefig('heatmap.png')
    ```

12. **How do you concatenate two NumPy arrays?**  
    Combines arrays for data aggregation.  
    ```python
    array1 = np.array([[1, 2], [3, 4]])
    array2 = np.array([[5, 6]])
    concatenated = np.concatenate((array1, array2), axis=0)
    ```

#### Advanced
13. **Write a function to create a NumPy array with a custom pattern.**  
    Generates structured arrays.  
    ```python
    def custom_pattern(shape, pattern='checkerboard'):
        array = np.zeros(shape)
        if pattern == 'checkerboard':
            array[::2, ::2] = 1
            array[1::2, 1::2] = 1
        return array
    ```

14. **How do you optimize array creation for large datasets?**  
    Uses efficient initialization methods.  
    ```python
    large_array = np.empty((10000, 10000))
    ```

15. **Write a function to create a block matrix in NumPy.**  
    Constructs matrices from subarrays.  
    ```python
    def block_matrix(blocks):
        return np.block(blocks)
    ```

16. **How do you handle memory-efficient array creation?**  
    Uses sparse arrays or generators.  
    ```python
    from scipy.sparse import csr_matrix
    sparse_array = csr_matrix((1000, 1000))
    ```

17. **Write a function to create a NumPy array with padded borders.**  
    Adds padding for convolutional tasks.  
    ```python
    def pad_array(array, pad_width):
        return np.pad(array, pad_width, mode='constant')
    ```

18. **How do you create a NumPy array with a specific memory layout?**  
    Controls C or Fortran order for performance.  
    ```python
    array = np.array([[1, 2], [3, 4]], order='F')
    ```

## Array Operations

### Basic
19. **How do you perform element-wise addition in NumPy?**  
   Adds arrays for data transformations.  
   ```python
   array1 = np.array([1, 2, 3])
   array2 = np.array([4, 5, 6])
   result = array1 + array2
   ```

20. **What is broadcasting in NumPy, and how does it work?**  
   Aligns arrays for operations.  
   ```python
   array = np.array([[1, 2], [3, 4]])
   scalar = 2
   result = array * scalar
   ```

21. **How do you compute the dot product of two NumPy arrays?**  
   Performs matrix multiplication.  
   ```python
   array1 = np.array([1, 2])
   array2 = np.array([3, 4])
   dot_product = np.dot(array1, array2)
   ```

22. **How do you calculate the mean of a NumPy array?**  
   Computes statistics for analysis.  
   ```python
   array = np.array([1, 2, 3, 4])
   mean = np.mean(array)
   ```

23. **How do you perform matrix transposition in NumPy?**  
   Flips rows and columns.  
   ```python
   array = np.array([[1, 2], [3, 4]])
   transposed = array.T
   ```

24. **How do you visualize array operations in NumPy?**  
   Plots operation results.  
   ```python
   import matplotlib.pyplot as plt
   array = np.array([1, 2, 3, 4])
   plt.plot(array, array**2)
   plt.savefig('array_operation.png')
   ```

#### Intermediate
25. **Write a function to perform element-wise operations on NumPy arrays.**  
    Applies custom operations.  
    ```python
    def element_wise_op(array1, array2, op='add'):
        if op == 'add':
            return array1 + array2
        elif op == 'multiply':
            return array1 * array2
    ```

26. **How do you implement broadcasting for custom operations?**  
    Aligns shapes dynamically.  
    ```python
    array = np.array([[1, 2], [3, 4]])
    vector = np.array([1, 2])
    result = array + vector
    ```

27. **Write a function to compute the outer product of two NumPy arrays.**  
    Generates matrix from vectors.  
    ```python
    def outer_product(vec1, vec2):
        return np.outer(vec1, vec2)
    ```

28. **How do you perform batch operations on NumPy arrays?**  
    Processes multiple arrays efficiently.  
    ```python
    arrays = [np.array([1, 2]), np.array([3, 4])]
    results = [arr * 2 for arr in arrays]
    ```

29. **Write a function to normalize a NumPy array.**  
    Scales values for ML preprocessing.  
    ```python
    def normalize_array(array):
        return (array - np.mean(array)) / np.std(array)
    ```

30. **How do you handle numerical stability in NumPy operations?**  
    Uses safe computations for large numbers.  
    ```python
    array = np.array([1e10, 2e10])
    result = np.log1p(array)
    ```

#### Advanced
31. **Write a function to implement matrix factorization in NumPy.**  
    Decomposes matrices for dimensionality reduction.  
    ```python
    def matrix_factorization(matrix, k):
        U, S, Vt = np.linalg.svd(matrix)
        return U[:, :k], np.diag(S[:k]), Vt[:k, :]
    ```

32. **How do you optimize NumPy operations for performance?**  
    Uses vectorized operations.  
    ```python
    array = np.random.rand(1000, 1000)
    result = np.einsum('ij,ij->i', array, array)
    ```

33. **Write a function to perform sliding window operations in NumPy.**  
    Applies operations over windows.  
    ```python
    def sliding_window(array, window_size):
        return np.lib.stride_tricks.sliding_window_view(array, window_size)
    ```

34. **How do you implement custom reductions in NumPy?**  
    Defines specialized aggregations.  
    ```python
    def custom_reduction(array, op='sum'):
        if op == 'sum':
            return np.sum(array, axis=0)
        elif op == 'prod':
            return np.prod(array, axis=0)
    ```

35. **Write a function to handle sparse array operations in NumPy.**  
    Optimizes for sparse data.  
    ```python
    from scipy.sparse import csr_matrix
    def sparse_operation(array):
        sparse = csr_matrix(array)
        return sparse.dot(sparse.T)
    ```

36. **How do you parallelize NumPy operations?**  
    Uses libraries like Numba or Dask.  
    ```python
    from numba import jit
    @jit
    def fast_operation(array):
        return array * 2
    ```

## Indexing and Slicing

### Basic
37. **How do you access elements in a NumPy array?**  
   Uses indices for data retrieval.  
   ```python
   array = np.array([[1, 2], [3, 4]])
   element = array[0, 1]
   ```

38. **What is array slicing in NumPy?**  
   Extracts subarrays with ranges.  
   ```python
   array = np.array([1, 2, 3, 4])
   slice = array[1:3]
   ```

39. **How do you use boolean indexing in NumPy?**  
   Filters arrays with conditions.  
   ```python
   array = np.array([1, 2, 3, 4])
   filtered = array[array > 2]
   ```

40. **How do you access rows and columns in a 2D NumPy array?**  
   Uses slicing for matrix operations.  
   ```python
   array = np.array([[1, 2], [3, 4]])
   row = array[0, :]
   ```

41. **What is fancy indexing in NumPy?**  
   Uses arrays as indices.  
   ```python
   array = np.array([10, 20, 30, 40])
   indices = [0, 2]
   selected = array[indices]
   ```

42. **How do you visualize sliced NumPy arrays?**  
   Plots subarray data.  
   ```python
   import matplotlib.pyplot as plt
   array = np.random.rand(5, 5)
   plt.imshow(array[:3, :3], cmap='Blues')
   plt.savefig('sliced_array.png')
   ```

#### Intermediate
43. **Write a function to extract a subarray using NumPy slicing.**  
    Retrieves specific regions.  
    ```python
    def extract_subarray(array, rows, cols):
        return array[rows[0]:rows[1], cols[0]:cols[1]]
    ```

44. **How do you use advanced indexing with NumPy?**  
    Combines integer and boolean indexing.  
    ```python
    array = np.array([[1, 2], [3, 4]])
    rows = np.array([0, 1])
    cols = np.array([1, 0])
    selected = array[rows, cols]
    ```

45. **Write a function to filter a NumPy array with conditions.**  
    Selects elements dynamically.  
    ```python
    def filter_array(array, threshold):
        return array[array > threshold]
    ```

46. **How do you modify array elements using indexing?**  
    Updates values conditionally.  
    ```python
    array = np.array([1, 2, 3, 4])
    array[array < 3] = 0
    ```

47. **Write a function to extract diagonal elements in NumPy.**  
    Retrieves matrix diagonals.  
    ```python
    def get_diagonal(array):
        return np.diagonal(array)
    ```

48. **How do you handle out-of-bounds indexing in NumPy?**  
    Uses safe indexing techniques.  
    ```python
    def safe_index(array, index):
        return array[index] if 0 <= index < len(array) else None
    ```

#### Advanced
49. **Write a function to implement multi-dimensional indexing in NumPy.**  
    Accesses complex array structures.  
    ```python
    def multi_dim_index(array, indices):
        return array[tuple(indices)]
    ```

50. **How do you optimize indexing for large NumPy arrays?**  
    Uses strides or views.  
    ```python
    array = np.random.rand(1000, 1000)
    view = array[::2, ::2]
    ```

51. **Write a function to perform conditional indexing with multiple criteria.**  
    Filters with complex logic.  
    ```python
    def multi_condition_index(array, cond1, cond2):
        return array[np.logical_and(array > cond1, array < cond2)]
    ```

52. **How do you implement custom indexing for NumPy arrays?**  
    Defines specialized access patterns.  
    ```python
    def custom_index(array, pattern='even'):
        if pattern == 'even':
            return array[::2]
        return array[1::2]
    ```

53. **Write a function to reorder NumPy array elements.**  
    Rearranges based on indices.  
    ```python
    def reorder_array(array, order):
        return array[np.argsort(order)]
    ```

54. **How do you handle sparse array indexing in NumPy?**  
    Uses sparse formats for efficiency.  
    ```python
    from scipy.sparse import csr_matrix
    def sparse_index(sparse_array, row, col):
        return sparse_array[row, col]
    ```

## Broadcasting and Vectorization

### Basic
55. **What is vectorization in NumPy, and why is it important?**  
   Replaces loops with array operations for speed.  
   ```python
   array = np.array([1, 2, 3])
   result = array * 2
   ```

56. **How do you perform broadcasting with mismatched shapes?**  
   Aligns arrays automatically.  
   ```python
   array = np.array([[1, 2], [3, 4]])
   vector = np.array([1, 2])
   result = array + vector
   ```

57. **How do you compute element-wise operations without loops?**  
   Uses vectorized functions.  
   ```python
   array = np.array([1, 2, 3])
   squared = np.square(array)
   ```

58. **What is the role of `np.vectorize` in NumPy?**  
   Applies scalar functions to arrays.  
   ```python
   def my_func(x):
       return x * 2
   vectorized = np.vectorize(my_func)
   result = vectorized(np.array([1, 2, 3]))
   ```

59. **How do you visualize broadcasting results?**  
   Plots operation outputs.  
   ```python
   import matplotlib.pyplot as plt
   array = np.ones((3, 3)) + np.array([1, 2, 3])
   plt.imshow(array, cmap='Greys')
   plt.savefig('broadcasting_result.png')
   ```

60. **How do you check broadcasting compatibility in NumPy?**  
   Verifies shape alignment.  
   ```python
   def check_broadcasting(shape1, shape2):
       try:
           np.broadcast_arrays(np.empty(shape1), np.empty(shape2))
           return True
       except ValueError:
           return False
   ```

#### Intermediate
61. **Write a function to perform broadcasting with custom arrays.**  
    Applies operations across shapes.  
    ```python
    def broadcast_operation(array, vector):
        return array + vector
    ```

62. **How do you optimize vectorized operations in NumPy?**  
    Minimizes memory overhead.  
    ```python
    array = np.random.rand(1000)
    result = np.sin(array, out=np.empty_like(array))
    ```

63. **Write a function to apply vectorized operations conditionally.**  
    Uses masks for selective computation.  
    ```python
    def conditional_vectorize(array, threshold):
        return np.where(array > threshold, array * 2, array)
    ```

64. **How do you handle broadcasting with higher-dimensional arrays?**  
    Aligns multi-dimensional shapes.  
    ```python
    array = np.ones((3, 4, 5))
    vector = np.array([1, 2, 3, 4])
    result = array + vector[:, np.newaxis]
    ```

65. **Write a function to vectorize a custom computation.**  
    Applies scalar logic to arrays.  
    ```python
    def vectorized_custom(array):
        return np.vectorize(lambda x: x**2 if x > 0 else 0)(array)
    ```

66. **How do you visualize vectorized operation performance?**  
    Compares loop vs. vectorized times.  
    ```python
    import matplotlib.pyplot as plt
    import time
    sizes = [100, 1000, 10000]
    times = []
    for n in sizes:
        array = np.random.rand(n)
        start = time.time()
        np.sin(array)
        times.append(time.time() - start)
    plt.plot(sizes, times)
    plt.savefig('vectorized_performance.png')
    ```

#### Advanced
67. **Write a function to implement complex broadcasting rules.**  
    Handles intricate shape alignments.  
    ```python
    def complex_broadcast(array, shape):
        return array + np.ones(shape)
    ```

68. **How do you optimize broadcasting for memory efficiency?**  
    Uses in-place operations.  
    ```python
    array = np.random.rand(1000, 1000)
    array += 1
    ```

69. **Write a function to vectorize matrix operations.**  
    Applies matrix computations efficiently.  
    ```python
    def vectorized_matrix_op(matrix1, matrix2):
        return np.einsum('ij,jk->ik', matrix1, matrix2)
    ```

70. **How do you handle broadcasting with sparse arrays?**  
    Uses sparse formats for efficiency.  
    ```python
    from scipy.sparse import csr_matrix
    def sparse_broadcast(sparse, dense):
        return sparse + dense
    ```

71. **Write a function to debug broadcasting issues.**  
    Logs shape mismatches.  
    ```python
    import logging
    def debug_broadcast(array1, array2):
        logging.basicConfig(filename='numpy.log', level=logging.INFO)
        try:
            return array1 + array2
        except ValueError as e:
            logging.error(f"Broadcasting error: {e}")
            raise
    ```

72. **How do you implement broadcasting with custom dtypes?**  
    Handles specialized data types.  
    ```python
    array = np.array([1, 2], dtype=np.float32)
    result = array + np.array([1, 2], dtype=np.int16)
    ```

## Linear Algebra

### Basic
73. **How do you compute the matrix inverse in NumPy?**  
   Inverts matrices for solving systems.  
   ```python
   matrix = np.array([[1, 2], [3, 4]])
   inverse = np.linalg.inv(matrix)
   ```

74. **What is the determinant of a matrix in NumPy?**  
   Measures matrix properties.  
   ```python
   matrix = np.array([[1, 2], [3, 4]])
   det = np.linalg.det(matrix)
   ```

75. **How do you solve a linear system in NumPy?**  
   Finds solutions to Ax = b.  
   ```python
   A = np.array([[1, 2], [3, 4]])
   b = np.array([5, 6])
   x = np.linalg.solve(A, b)
   ```

76. **How do you compute eigenvalues in NumPy?**  
   Analyzes matrix properties.  
   ```python
   matrix = np.array([[1, 2], [3, 4]])
   eigenvalues = np.linalg.eigvals(matrix)
   ```

77. **How do you perform singular value decomposition (SVD) in NumPy?**  
   Decomposes matrices for ML.  
   ```python
   matrix = np.array([[1, 2], [3, 4]])
   U, S, Vt = np.linalg.svd(matrix)
   ```

78. **How do you visualize matrix operations in NumPy?**  
   Plots matrix transformations.  
   ```python
   import matplotlib.pyplot as plt
   matrix = np.random.rand(5, 5)
   plt.imshow(matrix, cmap='hot')
   plt.savefig('matrix_plot.png')
   ```

#### Intermediate
79. **Write a function to solve a batch of linear systems in NumPy.**  
    Handles multiple systems efficiently.  
    ```python
    def batch_solve(A_batch, b_batch):
        return np.linalg.solve(A_batch, b_batch)
    ```

80. **How do you compute the matrix rank in NumPy?**  
    Determines linear independence.  
    ```python
    matrix = np.array([[1, 2], [2, 4]])
    rank = np.linalg.matrix_rank(matrix)
    ```

81. **Write a function to perform QR decomposition in NumPy.**  
    Decomposes matrices for stability.  
    ```python
    def qr_decomposition(matrix):
        Q, R = np.linalg.qr(matrix)
        return Q, R
    ```

82. **How do you compute the condition number of a matrix?**  
    Assesses numerical stability.  
    ```python
    matrix = np.array([[1, 2], [3, 4]])
    cond = np.linalg.cond(matrix)
    ```

83. **Write a function to compute the Cholesky decomposition.**  
    Factorizes symmetric matrices.  
    ```python
    def cholesky_decomp(matrix):
        return np.linalg.cholesky(matrix)
    ```

84. **How do you visualize eigenvalues of a matrix?**  
    Plots eigenvalue distributions.  
    ```python
    import matplotlib.pyplot as plt
    matrix = np.random.rand(5, 5)
    eigvals = np.linalg.eigvals(matrix)
    plt.scatter(eigvals.real, eigvals.imag)
    plt.savefig('eigenvalues_plot.png')
    ```

#### Advanced
85. **Write a function to implement iterative linear solvers in NumPy.**  
    Solves large systems efficiently.  
    ```python
    from scipy.sparse.linalg import cg
    def iterative_solve(A, b):
        x, _ = cg(A, b)
        return x
    ```

86. **How do you optimize linear algebra operations in NumPy?**  
    Uses BLAS/LAPACK for speed.  
    ```python
    matrix = np.random.rand(1000, 1000)
    result = np.linalg.inv(matrix)
    ```

87. **Write a function to compute the pseudo-inverse in NumPy.**  
    Handles non-square matrices.  
    ```python
    def pseudo_inverse(matrix):
        return np.linalg.pinv(matrix)
    ```

88. **How do you implement tensor operations in NumPy?**  
    Extends linear algebra to tensors.  
    ```python
    tensor = np.random.rand(3, 3, 3)
    result = np.tensordot(tensor, tensor, axes=([2], [2]))
    ```

89. **Write a function to handle ill-conditioned matrices.**  
    Stabilizes computations.  
    ```python
    def safe_inverse(matrix, tol=1e-10):
        if np.linalg.cond(matrix) < 1/tol:
            return np.linalg.inv(matrix)
        return np.linalg.pinv(matrix)
    ```

90. **How do you parallelize linear algebra operations?**  
    Uses multi-core processing.  
    ```python
    from joblib import Parallel, delayed
    def parallel_matrix_inv(matrices):
        return Parallel(n_jobs=-1)(delayed(np.linalg.inv)(m) for m in matrices)
    ```

## Integration with AI/ML Workflows

### Basic
91. **How do you preprocess data with NumPy for AI/ML?**  
   Normalizes and reshapes inputs.  
   ```python
   data = np.random.rand(100, 10)
   normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
   ```

92. **How do you create feature matrices in NumPy?**  
   Structures data for ML models.  
   ```python
   features = np.array([[1, 2], [3, 4], [5, 6]])
   ```

93. **How do you split data into train/test sets in NumPy?**  
   Prepares data for evaluation.  
   ```python
   data = np.random.rand(100, 5)
   train = data[:80]
   test = data[80:]
   ```

94. **How do you compute pairwise distances in NumPy?**  
   Used in clustering algorithms.  
   ```python
   from scipy.spatial.distance import cdist
   points = np.random.rand(10, 2)
   distances = cdist(points, points)
   ```

95. **How do you one-hot encode labels in NumPy?**  
   Prepares categorical data.  
   ```python
   labels = np.array([0, 1, 2])
   one_hot = np.eye(3)[labels]
   ```

96. **How do you visualize data distributions in NumPy?**  
   Plots histograms for analysis.  
   ```python
   import matplotlib.pyplot as plt
   data = np.random.randn(1000)
   plt.hist(data, bins=30)
   plt.savefig('data_distribution.png')
   ```

#### Intermediate
97. **Write a function to preprocess images with NumPy for ML.**  
    Normalizes and reshapes images.  
    ```python
    def preprocess_image(image):
        return (image / 255.0).reshape(-1)
    ```

98. **How do you implement data augmentation with NumPy?**  
    Generates synthetic data.  
    ```python
    def augment_data(array):
        return array + np.random.normal(0, 0.1, array.shape)
    ```

99. **Write a function to compute feature correlations in NumPy.**  
    Analyzes feature relationships.  
    ```python
    def feature_correlation(features):
        return np.corrcoef(features, rowvar=False)
    ```

100. **How do you handle missing data in NumPy for ML?**  
     Imputes or removes NaNs.  
     ```python
     def handle_missing(array):
         return np.where(np.isnan(array), np.mean(array, axis=0), array)
     ```

101. **Write a function to standardize features in NumPy.**  
     Scales features for ML models.  
     ```python
     def standardize_features(features):
         return (features - np.mean(features, axis=0)) / np.std(features, axis=0)
     ```

102. **How do you integrate NumPy with Scikit-learn?**  
     Prepares data for ML pipelines.  
     ```python
     from sklearn.linear_model import LogisticRegression
     X = np.random.rand(100, 5)
     y = np.random.randint(0, 2, 100)
     model = LogisticRegression().fit(X, y)
     ```

#### Advanced
103. **Write a function to implement PCA with NumPy.**  
     Reduces dimensionality for ML.  
     ```python
     def pca_transform(data, n_components):
         cov = np.cov(data.T)
         eigvals, eigvecs = np.linalg.eigh(cov)
         top_k = eigvecs[:, -n_components:]
         return data @ top_k
     ```

104. **How do you optimize NumPy for large-scale ML datasets?**  
     Uses chunked processing.  
     ```python
     def process_chunks(data, chunk_size=1000):
         for i in range(0, len(data), chunk_size):
             yield standardize_features(data[i:i + chunk_size])
     ```

105. **Write a function to compute gradients in NumPy.**  
     Supports optimization in ML.  
     ```python
     def compute_gradient(X, y, w):
         return X.T @ (X @ w - y) / len(y)
     ```

106. **How do you implement k-means clustering with NumPy?**  
     Groups data points.  
     ```python
     def kmeans(X, k, max_iters=100):
         centroids = X[np.random.choice(len(X), k)]
         for _ in range(max_iters):
             distances = cdist(X, centroids)
             labels = np.argmin(distances, axis=1)
             centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
         return labels, centroids
     ```

107. **Write a function to handle imbalanced datasets in NumPy.**  
     Resamples data for balance.  
     ```python
     def balance_data(X, y, minority_class):
         minority = X[y == minority_class]
         majority = X[y != minority_class]
         minority_upsampled = minority[np.random.choice(len(minority), len(majority))]
         return np.vstack([majority, minority_upsampled]), np.hstack([y[y != minority_class], np.full(len(majority), minority_class)])
     ```

108. **How do you integrate NumPy with deep learning frameworks?**  
     Converts data for TensorFlow/PyTorch.  
     ```python
     import tensorflow as tf
     array = np.random.rand(100, 10)
     tensor = tf.convert_to_tensor(array)
     ```

## Debugging and Error Handling

### Basic
109. **How do you debug NumPy array shapes?**  
     Logs shape information.  
     ```python
     def debug_shape(array):
         print(f"Shape: {array.shape}")
         return array
     ```

110. **What is a try-except block in NumPy applications?**  
     Handles numerical errors.  
     ```python
     try:
         result = np.linalg.inv(np.array([[1, 2], [2, 4]]))
     except np.linalg.LinAlgError as e:
         print(f"Error: {e}")
     ```

111. **How do you validate NumPy array inputs?**  
     Ensures correct shapes and types.  
     ```python
     def validate_array(array, expected_shape):
         if array.shape != expected_shape:
             raise ValueError(f"Expected shape {expected_shape}, got {array.shape}")
         return array
     ```

112. **How do you handle NaN values in NumPy?**  
     Detects and replaces NaNs.  
     ```python
     array = np.array([1, np.nan, 3])
     cleaned = np.nan_to_num(array, nan=0)
     ```

113. **What is the role of logging in NumPy debugging?**  
     Tracks errors and operations.  
     ```python
     import logging
     logging.basicConfig(filename='numpy.log', level=logging.INFO)
     logging.info("Starting NumPy operation")
     ```

114. **How do you handle overflow errors in NumPy?**  
     Uses safe numerical ranges.  
     ```python
     array = np.array([1e308], dtype=np.float64)
     result = np.clip(array, -1e308, 1e308)
     ```

#### Intermediate
115. **Write a function to retry NumPy operations on failure.**  
     Handles transient errors.  
     ```python
     def retry_operation(func, array, max_attempts=3):
         for attempt in range(max_attempts):
             try:
                 return func(array)
             except Exception as e:
                 if attempt == max_attempts - 1:
                     raise
                 print(f"Attempt {attempt+1} failed: {e}")
     ```

116. **How do you debug NumPy operation outputs?**  
     Inspects intermediate results.  
     ```python
     def debug_operation(array):
         result = array * 2
         print(f"Input: {array[:5]}, Output: {result[:5]}")
         return result
     ```

117. **Write a function to validate NumPy array dtypes.**  
     Ensures correct data types.  
     ```python
     def validate_dtype(array, expected_dtype):
         if array.dtype != expected_dtype:
             raise ValueError(f"Expected dtype {expected_dtype}, got {array.dtype}")
         return array
     ```

118. **How do you profile NumPy operation performance?**  
     Measures execution time.  
     ```python
     import time
     def profile_operation(array):
         start = time.time()
         result = np.sin(array)
         print(f"Operation took {time.time() - start}s")
         return result
     ```

119. **Write a function to handle memory errors in NumPy.**  
     Manages large arrays.  
     ```python
     def safe_operation(array, max_size=1e6):
         if array.size > max_size:
             raise MemoryError("Array too large")
         return array * 2
     ```

120. **How do you debug broadcasting errors in NumPy?**  
     Logs shape mismatches.  
     ```python
     def debug_broadcasting(array1, array2):
         try:
             return array1 + array2
         except ValueError as e:
             print(f"Broadcasting error: {e}, Shapes: {array1.shape}, {array2.shape}")
             raise
     ```

#### Advanced
121. **Write a function to implement a custom NumPy error handler.**  
     Logs specific errors.  
     ```python
     import logging
     def custom_error_handler(array, operation):
         logging.basicConfig(filename='numpy.log', level=logging.ERROR)
         try:
             return operation(array)
         except Exception as e:
             logging.error(f"Operation error: {e}")
             raise
     ```

122. **How do you implement circuit breakers in NumPy applications?**  
     Prevents cascading failures.  
     ```python
     from pybreaker import CircuitBreaker
     breaker = CircuitBreaker(fail_max=3, reset_timeout=60)
     @breaker
     def safe_operation(array):
         return np.linalg.inv(array)
     ```

123. **Write a function to detect numerical instability in NumPy.**  
     Checks for large condition numbers.  
     ```python
     def detect_instability(matrix):
         cond = np.linalg.cond(matrix)
         if cond > 1e10:
             print("Warning: Matrix may be ill-conditioned")
         return matrix
     ```

124. **How do you implement logging for distributed NumPy operations?**  
     Centralizes logs for debugging.  
     ```python
     import logging.handlers
     def setup_distributed_logging():
         handler = logging.handlers.SocketHandler('log-server', 9090)
         logging.getLogger().addHandler(handler)
         logging.info("NumPy operation started")
     ```

125. **Write a function to handle version compatibility in NumPy.**  
     Checks library versions.  
     ```python
     import numpy as np
     def check_numpy_version():
         if np.__version__ < '1.20':
             raise ValueError("Unsupported NumPy version")
     ```

126. **How do you debug NumPy performance bottlenecks?**  
     Profiles operation stages.  
     ```python
     import time
     def debug_bottlenecks(array):
         start = time.time()
         result = np.dot(array, array.T)
         print(f"Matrix multiplication: {time.time() - start}s")
         return result
     ```

## Visualization and Interpretation

### Basic
127. **How do you visualize NumPy array distributions?**  
     Plots histograms for data analysis.  
     ```python
     import matplotlib.pyplot as plt
     array = np.random.randn(1000)
     plt.hist(array, bins=30)
     plt.savefig('array_distribution.png')
     ```

128. **How do you create a scatter plot with NumPy data?**  
     Visualizes relationships in data.  
     ```python
     import matplotlib.pyplot as plt
     x = np.random.rand(100)
     y = np.random.rand(100)
     plt.scatter(x, y)
     plt.savefig('scatter_plot.png')
     ```

129. **How do you visualize matrix data in NumPy?**  
     Uses heatmaps for matrices.  
     ```python
     import matplotlib.pyplot as plt
     matrix = np.random.rand(5, 5)
     plt.imshow(matrix, cmap='coolwarm')
     plt.colorbar()
     plt.savefig('matrix_heatmap.png')
     ```

130. **How do you plot NumPy array operations?**  
     Visualizes transformed data.  
     ```python
     import matplotlib.pyplot as plt
     array = np.linspace(0, 10, 100)
     plt.plot(array, np.sin(array))
     plt.savefig('sin_plot.png')
     ```

131. **How do you create a 3D plot with NumPy data?**  
     Visualizes multi-dimensional arrays.  
     ```python
     from mpl_toolkits.mplot3d import Axes3D
     import matplotlib.pyplot as plt
     x = np.linspace(-5, 5, 100)
     y = np.linspace(-5, 5, 100)
     X, Y = np.meshgrid(x, y)
     Z = np.sin(np.sqrt(X**2 + Y**2))
     fig = plt.figure()
     ax = fig.add_subplot(111, projection='3d')
     ax.plot_surface(X, Y, Z)
     plt.savefig('3d_plot.png')
     ```

132. **How do you visualize NumPy array statistics?**  
     Plots mean, std, etc.  
     ```python
     import matplotlib.pyplot as plt
     arrays = [np.random.randn(100) for _ in range(5)]
     means = [np.mean(arr) for arr in arrays]
     plt.bar(range(len(means)), means)
     plt.savefig('array_stats.png')
     ```

#### Intermediate
133. **Write a function to visualize NumPy array comparisons.**  
     Plots multiple arrays.  
     ```python
     import matplotlib.pyplot as plt
     def compare_arrays(arrays, labels):
         for arr, label in zip(arrays, labels):
             plt.plot(arr, label=label)
         plt.legend()
         plt.savefig('array_comparison.png')
     ```

134. **How do you visualize NumPy clustering results?**  
     Plots clustered data points.  
     ```python
     import matplotlib.pyplot as plt
     def plot_clusters(X, labels):
         plt.scatter(X[:, 0], X[:, 1], c=labels)
         plt.savefig('cluster_plot.png')
     ```

135. **Write a function to visualize NumPy feature importance.**  
     Plots feature weights.  
     ```python
     import matplotlib.pyplot as plt
     def plot_feature_importance(features, importances):
         plt.bar(features, importances)
         plt.xticks(rotation=45)
         plt.savefig('feature_importance.png')
     ```

136. **How do you visualize NumPy matrix transformations?**  
     Shows before/after effects.  
     ```python
     import matplotlib.pyplot as plt
     def plot_transformation(matrix, transformed):
         plt.subplot(1, 2, 1)
         plt.imshow(matrix, cmap='Blues')
         plt.subplot(1, 2, 2)
         plt.imshow(transformed, cmap='Blues')
         plt.savefig('transformation_plot.png')
     ```

137. **Write a function to visualize NumPy error distributions.**  
     Plots operation errors.  
     ```python
     import matplotlib.pyplot as plt
     def plot_errors(errors):
         plt.hist(errors, bins=20)
         plt.savefig('error_distribution.png')
     ```

138. **How do you visualize NumPy data trends?**  
     Plots time series or trends.  
     ```python
     import matplotlib.pyplot as plt
     data = np.cumsum(np.random.randn(100))
     plt.plot(data)
     plt.savefig('data_trend.png')
     ```

#### Advanced
139. **Write a function to visualize NumPy high-dimensional data.**  
     Uses PCA for projection.  
     ```python
     from sklearn.decomposition import PCA
     import matplotlib.pyplot as plt
     def plot_high_dim_data(data):
         pca = PCA(n_components=2)
         reduced = pca.fit_transform(data)
         plt.scatter(reduced[:, 0], reduced[:, 1])
         plt.savefig('high_dim_plot.png')
     ```

140. **How do you implement a dashboard for NumPy metrics?**  
     Displays real-time stats.  
     ```python
     from fastapi import FastAPI
     app = FastAPI()
     metrics = []
     @app.get('/metrics')
     async def get_metrics():
         return {'metrics': metrics}
     ```

141. **Write a function to visualize NumPy operation performance.**  
     Plots execution times.  
     ```python
     import matplotlib.pyplot as plt
     def plot_performance(sizes, times):
         plt.plot(sizes, times, marker='o')
         plt.savefig('performance_plot.png')
     ```

142. **How do you visualize NumPy data drift?**  
     Tracks data changes over time.  
     ```python
     import matplotlib.pyplot as plt
     def plot_data_drift(metrics):
         plt.plot(metrics, marker='o')
         plt.savefig('data_drift.png')
     ```

143. **Write a function to visualize NumPy uncertainty.**  
     Plots confidence intervals.  
     ```python
     import matplotlib.pyplot as plt
     def plot_uncertainty(data, std):
         plt.plot(data)
         plt.fill_between(range(len(data)), data - std, data + std, alpha=0.2)
         plt.savefig('uncertainty_plot.png')
     ```

144. **How do you visualize NumPy model errors by category?**  
     Analyzes error patterns.  
     ```python
     import matplotlib.pyplot as plt
     def plot_error_by_category(categories, errors):
         plt.bar(categories, errors)
         plt.savefig('error_by_category.png')
     ```

## Best Practices and Optimization

### Basic
145. **What are best practices for NumPy code organization?**  
     Modularizes array operations.  
     ```python
     def preprocess_data(data):
         return standardize_features(data)
     def compute_features(data):
         return np.dot(data, data.T)
     ```

146. **How do you ensure reproducibility in NumPy?**  
     Sets random seeds.  
     ```python
     np.random.seed(42)
     ```

147. **What is caching in NumPy pipelines?**  
     Stores intermediate results.  
     ```python
     from functools import lru_cache
     @lru_cache(maxsize=1000)
     def compute_matrix(array):
         return np.dot(array, array.T)
     ```

148. **How do you handle large-scale NumPy arrays?**  
     Uses chunked processing.  
     ```python
     def process_large_array(array, chunk_size=1000):
         for i in range(0, len(array), chunk_size):
             yield array[i:i + chunk_size]
     ```

149. **What is the role of environment configuration in NumPy?**  
     Manages settings securely.  
     ```python
     import os
     os.environ['NUMPY_DATA_PATH'] = 'data.npy'
     ```

150. **How do you document NumPy code?**  
     Uses docstrings for clarity.  
     ```python
     def normalize_array(array):
         """Normalizes array to zero mean and unit variance."""
         return (array - np.mean(array)) / np.std(array)
     ```

#### Intermediate
151. **Write a function to optimize NumPy memory usage.**  
     Limits memory allocation.  
     ```python
     def optimize_memory(array, max_size=1e6):
         if array.size > max_size:
             return array[:int(max_size)]
         return array
     ```

152. **How do you implement unit tests for NumPy code?**  
     Validates array operations.  
     ```python
     import unittest
     class TestNumPy(unittest.TestCase):
         def test_normalize(self):
             array = np.array([1, 2, 3])
             result = normalize_array(array)
             self.assertAlmostEqual(np.mean(result), 0)
     ```

153. **Write a function to create reusable NumPy templates.**  
     Standardizes array processing.  
     ```python
     def array_template(array, operation='normalize'):
         if operation == 'normalize':
             return normalize_array(array)
         return array
     ```

154. **How do you optimize NumPy for batch processing?**  
     Processes arrays in chunks.  
     ```python
     def batch_process(arrays, batch_size=100):
         for i in range(0, len(arrays), batch_size):
             yield [normalize_array(arr) for arr in arrays[i:i + batch_size]]
     ```

155. **Write a function to handle NumPy configuration.**  
     Centralizes settings.  
     ```python
     def configure_numpy():
         return {'dtype': np.float32, 'order': 'C'}
     ```

156. **How do you ensure NumPy pipeline consistency?**  
     Standardizes versions and settings.  
     ```python
     import numpy as np
     def check_numpy_env():
         print(f"NumPy version: {np.__version__}")
     ```

#### Advanced
157. **Write a function to implement NumPy pipeline caching.**  
     Reuses processed arrays.  
     ```python
     import joblib
     def cache_array(array, cache_file='cache.npy'):
         if os.path.exists(cache_file):
             return np.load(cache_file)
         result = normalize_array(array)
         np.save(cache_file, result)
         return result
     ```

158. **How do you optimize NumPy for high-throughput processing?**  
     Uses parallel execution.  
     ```python
     from joblib import Parallel, delayed
     def high_throughput_process(arrays):
         return Parallel(n_jobs=-1)(delayed(normalize_array)(arr) for arr in arrays)
     ```

159. **Write a function to implement NumPy pipeline versioning.**  
     Tracks changes in workflows.  
     ```python
     def version_pipeline(config, version):
         with open(f'numpy_pipeline_v{version}.json', 'w') as f:
             json.dump(config, f)
     ```

160. **How do you implement NumPy pipeline monitoring?**  
     Logs performance metrics.  
     ```python
     import logging
     def monitored_process(array):
         logging.basicConfig(filename='numpy.log', level=logging.INFO)
         start = time.time()
         result = normalize_array(array)
         logging.info(f"Processed array in {time.time() - start}s")
         return result
     ```

161. **Write a function to handle NumPy scalability.**  
     Processes large datasets efficiently.  
     ```python
     def scalable_process(array, chunk_size=1000):
         for i in range(0, len(array), chunk_size):
             yield normalize_array(array[i:i + chunk_size])
     ```

162. **How do you implement NumPy pipeline automation?**  
     Scripts end-to-end workflows.  
     ```python
     def automate_pipeline(data):
         processed = normalize_array(data)
         np.save('processed_data.npy', processed)
         return processed
     ```

## Ethical Considerations in NumPy

### Basic
163. **What are ethical concerns in NumPy applications?**  
   Includes bias in data processing and resource usage.  
   ```python
   def check_data_bias(data, labels):
       return np.mean(data[labels == 0]) - np.mean(data[labels == 1])
   ```

164. **How do you detect bias in NumPy data processing?**  
   Analyzes statistical disparities.  
   ```python
   def detect_bias(data, groups):
       return {g: np.mean(data[groups == g]) for g in np.unique(groups)}
   ```

165. **What is data privacy in NumPy, and how is it ensured?**  
   Protects sensitive data.  
   ```python
   def anonymize_data(data):
       return data + np.random.normal(0, 0.1, data.shape)
   ```

166. **How do you ensure fairness in NumPy data processing?**  
   Balances data across groups.  
   ```python
   def fair_processing(data, labels):
       return balance_data(data, labels, minority_class=1)
   ```

167. **What is explainability in NumPy applications?**  
   Clarifies data transformations.  
   ```python
   def explain_transformation(data, transformed):
       print(f"Mean before: {np.mean(data)}, Mean after: {np.mean(transformed)}")
       return transformed
   ```

168. **How do you visualize NumPy data bias?**  
   Plots group-wise statistics.  
   ```python
   import matplotlib.pyplot as plt
   def plot_bias(groups, means):
       plt.bar(groups, means)
       plt.savefig('bias_plot.png')
   ```

#### Intermediate
169. **Write a function to mitigate bias in NumPy data.**  
     Reweights or resamples data.  
     ```python
     def mitigate_bias(data, labels, minority_class):
         return balance_data(data, labels, minority_class)
     ```

170. **How do you implement differential privacy in NumPy?**  
     Adds noise to protect data.  
     ```python
     def private_processing(data, epsilon=1.0):
         noise = np.random.laplace(0, 1/epsilon, data.shape)
         return data + noise
     ```