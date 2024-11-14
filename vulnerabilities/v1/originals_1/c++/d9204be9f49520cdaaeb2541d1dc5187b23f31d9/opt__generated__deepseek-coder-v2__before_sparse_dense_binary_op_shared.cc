REGISTER_KERNEL_BUILDER(Name("SparseDenseCwiseMul").Device(DEVICE_CPU).TypeConstraint<float>("T"), SparseDenseBinaryOpShared<CPUDevice, float, functor::mul<float>>);
REGISTER_KERNEL_BUILDER(Name("SparseDenseCwiseDiv").Device(DEVICE_CPU).TypeConstraint<float>("T"), SparseDenseBinaryOpShared<CPUDevice, float, functor::div<float>>);
REGISTER_KERNEL_BUILDER(Name("SparseDenseCwiseAdd").Device(DEVICE_CPU).TypeConstraint<float>("T"), SparseDenseBinaryOpShared<CPUDevice, float, functor::add<float>>);
