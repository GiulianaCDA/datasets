#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  void SpatialMaxPooling<Eigen::GpuDevice, T>::operator()(             \
      const Eigen::GpuDevice& d, typename TTypes<T, 4>::Tensor output, \
      typename TTypes<T, 4>::ConstTensor input, int window_rows,       \
      int window_cols, int row_stride, int col_stride,                 \
      const Eigen::PaddingType& padding);                              \
  extern template struct SpatialMaxPooling<Eigen::GpuDevice, T>;
