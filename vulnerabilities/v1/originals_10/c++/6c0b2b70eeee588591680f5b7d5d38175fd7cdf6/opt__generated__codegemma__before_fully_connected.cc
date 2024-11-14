// ...

TfLiteStatus fully_connected::Prepare<kernel_type>(TfLiteContext* context,
                                               TfLiteNode* node) {
  // ...
  // Calculate filter dimensions based on input and filter tensors.
  filter_dims = ComputeOutputDimensions(input, filter, kWeightsDimension);

  // ...
  return kTfLiteOk;
}

// ...

TfLiteStatus fully_connected::Eval<kernel_type>(TfLiteContext* context,
                                              TfLiteNode* node) {
  // ...
  // Perform the fully connected operation using the calculated filter dimensions.
  // ...

  return kTfLiteOk;
}

// ...
