#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"

at::Tensor at_dpu::native::DPUNativeFunctions::to(
    const at::Tensor &self, c10::ScalarType dtype, bool non_blocking, bool copy,
    c10::optional<c10::MemoryFormat> memory_format) {
  if (self.scalar_type() == dtype && !copy) {
    return self;
  }

  c10::TensorOptions options = self.options().dtype(dtype);
  at::Tensor result = at::empty_like(self, options, memory_format);

  at::native::copy_(result, self, non_blocking);

  return result;
}

at::Tensor at_dpu::native::DPUNativeFunctions::to(
    const at::Tensor &self, c10::Device device, c10::ScalarType dtype,
    bool non_blocking, bool copy,
    c10::optional<c10::MemoryFormat> memory_format) {
  printf("[Info] call this one\n");
  fflush(stdout);
  if (self.device() == device && self.scalar_type() == dtype && !copy) {
    return self;
  }

  c10::TensorOptions options = self.options().device(device).dtype(dtype);
  at::Tensor result = at::empty_like(self, options, memory_format);

  at::native::copy_(result, self, non_blocking);

  return result;
}
