/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 */

#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace expand_dims {

enum InputIndex { kInput = 0, kAxis };

TfLiteStatus ExpandTensorDim(TfLiteContext* context, const TfLiteTensor& input,
                             int axis, TfLiteTensor* output) {
  if (axis < 0) {
    axis += input.dims->size + 1;
  }

  TF_LITE_ENSURE(context, axis <= input.dims->size);

  TfLiteIntArray* output_dims = TfLiteIntArrayCreate(input.dims->size + 1);
  TF_LITE_ENSURE(context, output_dims != nullptr);

  for (int i = 0; i < output_dims->size; ++i) {
    if (i < axis) {
      output_dims->data[i] = input.dims->data[i];
    } else if (i == axis) {
      output_dims->data[i] = 1;
    } else {
      output_dims->data[i] = input.dims->data[i - 1];
    }
  }

  return context->ResizeTensor(context, output, output_dims);
}

TfLiteStatus GetAxisValueFromTensor(const TfLiteTensor& axis, int* axis_value) {
  if (axis.type == kTfLiteInt32) {
    *axis_value = *axis.data.i32;
  } else if (axis.type == kTfLiteInt64) {
    *axis_value = static_cast<int>(*axis.data.i64);
  } else {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto& params = *reinterpret_cast<TfLiteExpandDimsParams*>(node->builtin_data);
  TfLiteTensor* input = &context->tensors[node->inputs->data[kInput]];
  TfLiteTensor* axis = &context->tensors[node->inputs->data[kAxis]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  int axis_value;
  TF_LITE_ENSURE(context, GetAxisValueFromTensor(*axis, &axis_value) == kTfLiteOk);

  return ExpandTensorDim(context, *input, axis_value, output);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto& params = *reinterpret_cast<TfLiteExpandDimsParams*>(node->builtin_data);
  TfLiteTensor* input = &context->tensors[node->inputs->data[kInput]];
  TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

  memcpy(output->data.raw, input->data.raw, input->bytes);
  return kTfLiteOk;
}

TfLiteRegistration* Register_EXPAND_DIMS() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

} // namespace expand_dims
} // namespace builtin
} // namespace ops
} // namespace tflite
