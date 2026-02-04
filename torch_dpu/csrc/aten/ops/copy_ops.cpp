#include <torch/torch.h>

#include <cstring>

#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"

namespace at_dpu::native {

at::Tensor DPUNativeFunctions::_copy_from(const at::Tensor &self, const at::Tensor &dst, bool non_blocking) {
  // Create a non-const copy of dst for the copy operation
  at::Tensor dst_mut = const_cast<at::Tensor &>(dst);

  int64_t numel = self.numel();
  TORCH_CHECK(dst_mut.numel() == numel, "number of elements in source and destination tensors must match");

  TORCH_CHECK(self.scalar_type() == dst_mut.scalar_type(), "source and destination tensors must have the same dtype");

  // Use memcpy for fast byte-level copy
  int64_t nbytes = numel * self.element_size();
  void *dst_ptr = dst_mut.data_ptr();
  const void *src_ptr = self.data_ptr();

  std::memcpy(dst_ptr, src_ptr, nbytes);

  return dst_mut;
}

at::Tensor DPUNativeFunctions::_to_copy(const at::Tensor &self, ::std::optional<at::ScalarType> dtype,
                                        ::std::optional<at::Layout> layout, ::std::optional<at::Device> device,
                                        ::std::optional<bool> pin_memory, bool non_blocking,
                                        ::std::optional<at::MemoryFormat> memory_format) {
  // Determine target dtype
  at::ScalarType target_dtype = dtype.has_value() ? dtype.value() : self.scalar_type();

  // Determine target device (default to DPU if not specified)
  c10::Device target_device = device.has_value() ? device.value() : self.device();

  TORCH_CHECK(target_device.type() == c10::DeviceType::PrivateUse1 || target_device.type() == c10::DeviceType::CPU,
              "dpu _to_copy only supports copying to dpu or cpu devices, got: ", target_device);

  // Create empty target tensor
  at::Tensor result = at::empty_strided(self.sizes(), self.strides(),
                                        at::TensorOptions()
                                            .dtype(target_dtype)
                                            .device(target_device)
                                            .layout(layout.value_or(self.layout()))
                                            .pinned_memory(pin_memory.value_or(false)));

  // Handle dtype conversion + copy
  if (self.scalar_type() == target_dtype) {
    // Same dtype: use memcpy for fast byte-level copy
    int64_t nbytes = self.numel() * self.element_size();
    void *dst_ptr = result.data_ptr();
    const void *src_ptr = self.data_ptr();
    std::memcpy(dst_ptr, src_ptr, nbytes);
  } else {
    // Different dtype: need type conversion
    at::native::copy_(result, self, non_blocking);
  }

  return result;
}

}  // namespace at_dpu::native
