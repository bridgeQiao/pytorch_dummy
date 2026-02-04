#include <torch/torch.h>

#include <cstring>

#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"
#include "torch_dpu/csrc/core/dpu/dpu_storage_impl.h"

namespace at_dpu::native {

at::Scalar DPUNativeFunctions::_local_scalar_dense(const at::Tensor &self) {
  TORCH_CHECK(self.numel() == 1, "_local_scalar_dense: tensor must have exactly one element");

  // Get scalar value based on dtype using data_ptr
  auto dtype = self.scalar_type();
  const void *data_ptr = self.const_data_ptr();

  switch (dtype) {
    case at::ScalarType::Float:
      return at::Scalar(*static_cast<const float *>(data_ptr));
    case at::ScalarType::Double:
      return at::Scalar(*static_cast<const double *>(data_ptr));
    case at::ScalarType::Int:
      return at::Scalar(*static_cast<const int32_t *>(data_ptr));
    case at::ScalarType::Long:
      return at::Scalar(*static_cast<const int64_t *>(data_ptr));
    case at::ScalarType::Bool:
      return at::Scalar(*static_cast<const bool *>(data_ptr));
    case at::ScalarType::Half:
      return at::Scalar(static_cast<double>(*static_cast<const at::Half *>(data_ptr)));
    case at::ScalarType::BFloat16:
      return at::Scalar(static_cast<double>(*static_cast<const at::BFloat16 *>(data_ptr)));
    case at::ScalarType::Byte:
      return at::Scalar(*static_cast<const uint8_t *>(data_ptr));
    case at::ScalarType::Char:
      return at::Scalar(*static_cast<const int8_t *>(data_ptr));
    case at::ScalarType::Short:
      return at::Scalar(*static_cast<const int16_t *>(data_ptr));
    default:
      TORCH_CHECK(false, "_local_scalar_dense: unsupported dtype ", dtype);
  }
}

at::Tensor &DPUNativeFunctions::set_(at::Tensor &self, at::Storage source) {
  // Set the storage from source using TensorImpl API
  self.unsafeGetTensorImpl()->set_storage_keep_dtype(source);
  return self;
}

at::Tensor &DPUNativeFunctions::fill_(at::Tensor &self, const at::Scalar &value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "dpu fill_ only supports dpu tensors");

  // Get tensor properties
  int64_t numel = self.numel();
  auto dtype = self.scalar_type();
  void *data_ptr = self.mutable_data_ptr();

  // Fill tensor based on dtype
  switch (dtype) {
    case at::ScalarType::Float: {
      float val = value.to<float>();
      float *ptr = static_cast<float *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    case at::ScalarType::Double: {
      double val = value.to<double>();
      double *ptr = static_cast<double *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    case at::ScalarType::Int: {
      int32_t val = value.to<int32_t>();
      int32_t *ptr = static_cast<int32_t *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    case at::ScalarType::Long: {
      int64_t val = value.to<int64_t>();
      int64_t *ptr = static_cast<int64_t *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    case at::ScalarType::Bool: {
      bool val = value.to<bool>();
      bool *ptr = static_cast<bool *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    case at::ScalarType::Byte: {
      uint8_t val = value.to<uint8_t>();
      uint8_t *ptr = static_cast<uint8_t *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    case at::ScalarType::Char: {
      int8_t val = value.to<int8_t>();
      int8_t *ptr = static_cast<int8_t *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    case at::ScalarType::Short: {
      int16_t val = value.to<int16_t>();
      int16_t *ptr = static_cast<int16_t *>(data_ptr);
      for (int64_t i = 0; i < numel; ++i) {
        ptr[i] = val;
      }
      break;
    }
    default:
      TORCH_CHECK(false, "fill_: unsupported dtype ", dtype);
  }

  return self;
}

const at::Tensor &DPUNativeFunctions::resize_(const at::Tensor &self, at::IntArrayRef size,
                                              ::std::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "dpu resize_ only supports dpu tensors");

  // Remove const to perform in-place modification
  at::Tensor &self_mut = const_cast<at::Tensor &>(self);

  // Calculate new total number of elements
  int64_t new_numel = 1;
  for (int64_t dim : size) {
    new_numel *= dim;
  }

  // Get current storage size
  int64_t storage_size = self_mut.storage().nbytes();
  int64_t required_bytes = new_numel * self_mut.element_size();

  // Check if we need to reallocate storage
  if (required_bytes > storage_size) {
    // Use DPU empty operator to create a new tensor with required size
    // Note: must specify DPU device explicitly
    at::Tensor new_tensor = DPUNativeFunctions::empty(size, self_mut.scalar_type(), std::nullopt,
                                                      c10::Device(c10::DeviceType::PrivateUse1), std::nullopt,
                                                      memory_format);

    // Copy old data to new tensor
    if (storage_size > 0 && self_mut.storage().data()) {
      int64_t copy_bytes = std::min(storage_size, required_bytes);
      std::memcpy(new_tensor.mutable_data_ptr(), self_mut.storage().data(), copy_bytes);
    }

    // Swap storage with new tensor using set_
    DPUNativeFunctions::set_(self_mut, new_tensor.storage());
  }

  // Set the new sizes (strides are recomputed automatically)
  self_mut.unsafeGetTensorImpl()->set_sizes_contiguous(size);

  return self_mut;
}

at::Tensor DPUNativeFunctions::cat(const at::ITensorListRef &tensors, int64_t dim) {
  TORCH_CHECK(!tensors.empty(), "torch.cat: expected a non-empty list of tensors");

  // Check all tensors are on dpu device
  for (const auto &tensor : tensors) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, "dpu cat only supports dpu tensors");
  }

  const at::Tensor &first = tensors.front();
  int64_t ndim = first.dim();

  // Normalize dim
  if (dim < 0) {
    dim += ndim;
  }
  TORCH_CHECK(dim >= 0 && dim < ndim, "torch.cat: dim out of range");

  // Check all tensors have same shape except for cat dimension
  auto base_shape = first.sizes().vec();
  for (const auto &tensor : tensors) {
    TORCH_CHECK(tensor.dim() == ndim, "torch.cat: all tensors must have same number of dimensions");
    TORCH_CHECK(tensor.scalar_type() == first.scalar_type(), "torch.cat: all tensors must have same dtype");

    auto tensor_shape = tensor.sizes();
    for (int64_t d = 0; d < ndim; ++d) {
      if (d != dim) {
        TORCH_CHECK(tensor_shape[d] == base_shape[d], "torch.cat: sizes must match except in cat dimension");
      }
    }
  }

  // Calculate output shape
  auto output_shape = base_shape;
  output_shape[dim] = 0;
  for (const auto &tensor : tensors) {
    output_shape[dim] += tensor.size(dim);
  }

  // Create output tensor using DPU's empty
  at::Tensor result = DPUNativeFunctions::empty(output_shape, first.scalar_type(), std::nullopt,
                                                c10::Device(c10::DeviceType::PrivateUse1), std::nullopt,
                                                std::nullopt);

  // Copy data from each input tensor
  int64_t offset = 0;
  int64_t element_size = first.element_size();
  int64_t slice_nbytes = 1;
  for (int64_t d = 0; d < ndim; ++d) {
    if (d != dim) {
      slice_nbytes *= base_shape[d];
    }
  }

  for (const auto &tensor : tensors) {
    int64_t tensor_dim_size = tensor.size(dim);
    int64_t tensor_slice_bytes = tensor_dim_size * slice_nbytes * element_size;

    // Calculate source and destination pointers
    const void *src_ptr = tensor.const_data_ptr();
    void *dst_ptr = static_cast<uint8_t *>(result.mutable_data_ptr()) + offset * slice_nbytes * element_size;

    // Copy data using memcpy
    std::memcpy(dst_ptr, src_ptr, tensor_slice_bytes);

    offset += tensor_dim_size;
  }

  return result;
}

}  // namespace at_dpu::native
