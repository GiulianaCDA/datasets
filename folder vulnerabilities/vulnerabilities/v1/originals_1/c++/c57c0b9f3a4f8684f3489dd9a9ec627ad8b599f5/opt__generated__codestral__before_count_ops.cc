 This is a TensorFlow C++ code file containing implementations of three types of kernels: DenseCount, SparseCount and RaggedCount. Each kernel handles the count operation for different input data representations (dense, sparse, or ragged) while allowing multiple input and output data types.

   Here is a brief overview of each kernel's functionality:

   1. **DenseCount**: This kernel takes three inputs - values, weights, and splits - and performs the count operation on dense input data. It validates input shapes, computes batch-wise counts based on the provided splits, handles weighted or unweighted counting, and generates sparse output data.

   2. **SparseCount**: This kernel takes four inputs - indices, values, dense_shape, and weights (optional) - and performs the count operation on sparse input data. It validates input shapes, extracts unique row indices from the sparse indices, performs weighted or unweighted counting for each unique index, and generates dense output data.

   3. **RaggedCount**: This kernel takes three inputs - values, weights, and splits_dense_values - and performs the count operation on ragged input data. It validates input shapes, computes batch-wise counts based on the provided splits, handles weighted or unweighted counting, and generates dense output data.

   The code also uses macros to register the kernels for various combinations of input and output data types (int32, int64, float, double).