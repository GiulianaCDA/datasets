```

**Explanation:**

The provided code registers CPU and GPU kernels for the TensorFlow operation called `QuantizeAndDequantize`. The kernels are used to quantize and dequantize input data in a neural network model.

**Key Points:**

* **CPU Kernels:**
    * `QuantizeAndDequantizeV2Op`: Quantizes and dequantizes data using a single scale.
    * `QuantizeAndDequantizeV3Op`: Quantizes and dequantizes data using per-channel scaling.
    * `QuantizeAndDequantizeV4Op`: Quantizes and dequantizes data using a single scale.
    * `QuantizeAndDequantizeV4GradientOp`: Calculates gradients for the quantized data.
* **GPU Kernels:**
    * Similar to the CPU kernels, but optimized for GPU devices.
* **Template Instantiation:**
    * The kernels are instantiated for both float and double data types.
* **Kernel Registration:**
    * The kernels are registered using the `REGISTER_KERNEL_BUILDER` macro.
    * The kernels are associated with the `QuantizeAndDequantize` operation.
* **Host Memory:**
    * The GPU kernels require additional host memory for the `input_min`, `input_max`, and `num_bits` tensors.

**Usage:**

These kernels can be used within TensorFlow models to perform quantization and dequantization operations. They can be called using the `QuantizeAndDequantize` operation in TensorFlow code.

**Note:**

* The code assumes the availability of the `GOOGLE_CUDA` or `TENSORFLOW_USE_ROCM` macros to determine whether GPU support is available.
* The specific kernel used will depend on the device type (CPU or GPU) and data type.