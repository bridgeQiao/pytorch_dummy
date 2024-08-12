#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"

template <typename T>
void AddImplDpu(T *out, T *self, T *other, T alpha, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    out[i] = self[i] + alpha * other[i];
  }
}
at::Tensor &at_dpu::native::DPUNativeFunctions::add_out(const at::Tensor &self,
                                                    const at::Tensor &other,
                                                    const c10::Scalar &alpha,
                                                    at::Tensor &out) {
  // 这里实现你的自定义 add 逻辑
  // printf("[Info]Using custom add for PrivateUse1 backend\n");
  // 获取张量的数据类型
  auto dtype = self.scalar_type();

  // 根据数据类型选择适当的加法实现
  switch (dtype) {
  case at::ScalarType::Float: {
    float *out_ptr = out.data_ptr<float>();
    float *self_ptr = self.data_ptr<float>();
    float *other_ptr = other.data_ptr<float>();
    float alpha_value = alpha.to<float>();

    AddImplDpu(out_ptr, self_ptr, other_ptr, alpha_value, self.numel());
    break;
  }
  case at::ScalarType::Double: {
    double *out_ptr = out.data_ptr<double>();
    double *self_ptr = self.data_ptr<double>();
    double *other_ptr = other.data_ptr<double>();
    double alpha_value = alpha.to<double>();

    AddImplDpu(out_ptr, self_ptr, other_ptr, alpha_value, self.numel());
    break;
  }
  // 可以添加其他数据类型的情况...
  default:
    TORCH_CHECK(false, "Unsupported data type for PrivateUse1 add.out");
  }

  return out;
}
