
Here is the code from the `fully_connected` kernel with the `kReference` and `kGenericOptimized` kernels removed, as well as the `kLegacyPie` kernel replaced by a call to `Register_FULLY_CONNECTED_GENERIC_OPT()`:
```c++
#include "tensorflow/lite/kernels/internal/optimized/fully_connected.h"

namespace tflite {
namespace ops {
namespace builtin {

TfLiteRegistration* Register_FULLY_CONNECTED() {
  return Register_FULLY_CONNECTED_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
```
This code defines a single function `Register_FULLY_CONNECTED()` that returns a pointer to a `TfLiteRegistration` object. The registration function is defined in the `fully_connected` kernel, and it uses the optimized implementation of the fully connected layer for the `kGenericOptimized` kernel, which is defined in the `internal/optimized/` directory.

The `Register_FULLY_CONNECTED()` function is called during the initialization phase of the model interpreter to register the fully connected operation as a built-in TensorFlow Lite kernel.