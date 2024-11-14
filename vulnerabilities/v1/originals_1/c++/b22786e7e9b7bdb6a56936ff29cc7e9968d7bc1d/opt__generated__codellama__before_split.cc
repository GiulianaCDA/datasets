```
 
This code defines a custom TensorFlow Lite op called "SPLIT" that splits an input tensor into multiple output tensors. It uses the `tflite::ops::builtin::Split` class to perform the split operation, and it uses the `VectorOfTensors` class to handle the output tensors.

The `Prepare` function is responsible for determining the shapes of the output tensors based on the input tensor and the axis along which to split. It also sets the data type of all the output tensors to match that of the input tensor. If the "axis" tensor is constant, it resizes the output tensors immediately. Otherwise, it defers shape calculation until `Eval` is called.

The `Eval` function simply calls the `Split` operation with the appropriate parameters and data types. It also handles any error cases that may arise during execution.