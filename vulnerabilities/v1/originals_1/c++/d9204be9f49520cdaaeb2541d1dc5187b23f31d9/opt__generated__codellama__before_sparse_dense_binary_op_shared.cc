#define REGISTER_KERNELS(T)                                                  \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseDenseCwiseMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseDenseBinaryOpShared<CPUDevice, T, functor::mul<T>>)              \
                                                                             \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseDenseCwiseDiv").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseDenseBinaryOpShared<CPUDevice, T, functor::div<T>>)              \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("SparseDenseCwiseAdd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SparseDenseBinaryOpShared<CPUDevice, T, functor::add<T>>)
