#include <torch/torch.h>

#include <cmath>

#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"

namespace at_dpu::native {

at::Tensor DPUNativeFunctions::isfinite(const at::Tensor &self) {
  // Create output tensor with bool type
  at::Tensor result = DPUNativeFunctions::empty(self.sizes(), at::ScalarType::Bool, std::nullopt,
                                                 c10::Device(c10::DeviceType::PrivateUse1), std::nullopt,
                                                 std::nullopt);

  int64_t numel = self.numel();
  auto dtype = self.scalar_type();
  bool *out_ptr = result.data_ptr<bool>();

  // Check if each element is finite based on dtype
  switch (dtype) {
    case at::ScalarType::Float: {
      const float *self_ptr = self.data_ptr<float>();
      for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] = std::isfinite(self_ptr[i]);
      }
      break;
    }
    case at::ScalarType::Double: {
      const double *self_ptr = self.data_ptr<double>();
      for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] = std::isfinite(self_ptr[i]);
      }
      break;
    }
    case at::ScalarType::Half: {
      const at::Half *self_ptr = self.data_ptr<at::Half>();
      for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] = std::isfinite(static_cast<float>(self_ptr[i]));
      }
      break;
    }
    case at::ScalarType::BFloat16: {
      const at::BFloat16 *self_ptr = self.data_ptr<at::BFloat16>();
      for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] = std::isfinite(static_cast<float>(self_ptr[i]));
      }
      break;
    }
    case at::ScalarType::Int:
    case at::ScalarType::Long:
    case at::ScalarType::Bool:
    case at::ScalarType::Byte:
    case at::ScalarType::Char:
    case at::ScalarType::Short: {
      // Integer types are always finite
      for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] = true;
      }
      break;
    }
    default:
      TORCH_CHECK(false, "isfinite: unsupported dtype ", dtype);
  }

  return result;
}

}  // namespace at_dpu::native
