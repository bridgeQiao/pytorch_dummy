#include <torch/torch.h>

#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"

namespace at_dpu::native {

at::Tensor DPUNativeFunctions::view(const at::Tensor &self, c10::IntArrayRef size) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "dpu view only supports dpu tensors");

  int64_t numel = self.numel();

  // Handle -1 dimension inference
  c10::SmallVector<int64_t, 5> inferred_size(size.begin(), size.end());
  int64_t inferred_idx = -1;

  for (size_t i = 0; i < inferred_size.size(); ++i) {
    if (inferred_size[i] == -1) {
      TORCH_CHECK(inferred_idx == -1, "view() can only have one -1 dimension (got more)");
      inferred_idx = i;
    }
  }

  if (inferred_idx != -1) {
    // Calculate the inferred dimension
    int64_t known_numel = 1;
    for (size_t i = 0; i < inferred_size.size(); ++i) {
      if (i != static_cast<size_t>(inferred_idx)) {
        known_numel *= inferred_size[i];
      }
    }
    TORCH_CHECK(known_numel != 0, "view() cannot infer dimension when other dimensions are zero");
    TORCH_CHECK(numel % known_numel == 0, "view() size is not divisible by known dimensions");
    inferred_size[inferred_idx] = numel / known_numel;
  }

  // Check that number of elements matches
  int64_t new_numel = 1;
  for (int64_t dim : inferred_size) {
    new_numel *= dim;
  }
  TORCH_CHECK(numel == new_numel, "view() is only possible if the number of elements matches");

  // Calculate strides for contiguous layout
  c10::SmallVector<int64_t, 5> strides(inferred_size.size());
  strides[inferred_size.size() - 1] = 1;
  for (int64_t i = inferred_size.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * inferred_size[i + 1];
  }

  // Create a view that shares storage with self
  at::Tensor result = DPUNativeFunctions::empty_strided(inferred_size, strides, self.scalar_type(), std::nullopt,
                                                         c10::Device(c10::DeviceType::PrivateUse1), std::nullopt);

  // Share the storage
  result.set_(self.storage());

  // Set storage offset to match self
  result.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());

  return result;
}

at::Tensor DPUNativeFunctions::as_strided(const at::Tensor &self, at::IntArrayRef size, at::IntArrayRef stride,
                                          ::std::optional<int64_t> storage_offset) {
  return at::native::as_strided_tensorimpl(
      self, size, stride, storage_offset.has_value() ? std::make_optional(storage_offset.value()) : std::nullopt);
}

}  // namespace at_dpu::native
