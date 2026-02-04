#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"

template <typename T>
void AddImplDpu(T *out, T *self, T *other, T alpha, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    out[i] = self[i] + alpha * other[i];
  }
}

at::Tensor &at_dpu::native::DPUNativeFunctions::add_out(const at::Tensor &self, const at::Tensor &other,
                                                        const c10::Scalar &alpha, at::Tensor &out) {
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
    case at::ScalarType::Int: {
      int32_t *out_ptr = out.data_ptr<int32_t>();
      int32_t *self_ptr = self.data_ptr<int32_t>();
      int32_t *other_ptr = other.data_ptr<int32_t>();
      int32_t alpha_value = alpha.to<int32_t>();

      AddImplDpu(out_ptr, self_ptr, other_ptr, alpha_value, self.numel());
      break;
    }
    // 可以添加其他数据类型的情况...
    default:
      TORCH_CHECK(false, "Unsupported data type for PrivateUse1 add.out");
  }

  return out;
}

template <typename T>
void EqImplDpu(bool *out, T *self, T *other, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    out[i] = (self[i] == other[i]);
  }
}

at::Tensor &at_dpu::native::DPUNativeFunctions::eq_out(const at::Tensor &self, const at::Tensor &other,
                                                       at::Tensor &out) {
  // Get tensor data type
  auto dtype = self.scalar_type();

  // Check output tensor is bool type
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Bool, "eq_out: output tensor must be bool type");

  // Check shapes match
  TORCH_CHECK(self.sizes() == other.sizes(), "eq_out: self and other must have the same shape");
  TORCH_CHECK(self.sizes() == out.sizes(), "eq_out: self and out must have the same shape");

  int64_t numel = self.numel();
  bool *out_ptr = out.data_ptr<bool>();

  // Select implementation based on data type
  switch (dtype) {
    case at::ScalarType::Float: {
      float *self_ptr = self.data_ptr<float>();
      float *other_ptr = other.data_ptr<float>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Double: {
      double *self_ptr = self.data_ptr<double>();
      double *other_ptr = other.data_ptr<double>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Int: {
      int32_t *self_ptr = self.data_ptr<int32_t>();
      int32_t *other_ptr = other.data_ptr<int32_t>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Long: {
      int64_t *self_ptr = self.data_ptr<int64_t>();
      int64_t *other_ptr = other.data_ptr<int64_t>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Bool: {
      bool *self_ptr = self.data_ptr<bool>();
      bool *other_ptr = other.data_ptr<bool>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Byte: {
      uint8_t *self_ptr = self.data_ptr<uint8_t>();
      uint8_t *other_ptr = other.data_ptr<uint8_t>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Char: {
      int8_t *self_ptr = self.data_ptr<int8_t>();
      int8_t *other_ptr = other.data_ptr<int8_t>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Short: {
      int16_t *self_ptr = self.data_ptr<int16_t>();
      int16_t *other_ptr = other.data_ptr<int16_t>();
      EqImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported data type for PrivateUse1 eq.out");
  }

  return out;
}

template <typename T>
void NeScalarImplDpu(bool *out, T *self, T other, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    out[i] = (self[i] != other);
  }
}

at::Tensor &at_dpu::native::DPUNativeFunctions::ne_out(const at::Tensor &self, const at::Scalar &other,
                                                       at::Tensor &out) {
  // Get tensor data type
  auto dtype = self.scalar_type();

  // Check output tensor is bool type
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Bool, "ne_out: output tensor must be bool type");

  // Check shapes match
  TORCH_CHECK(self.sizes() == out.sizes(), "ne_out: self and out must have the same shape");

  int64_t numel = self.numel();
  bool *out_ptr = out.data_ptr<bool>();

  // Select implementation based on data type
  switch (dtype) {
    case at::ScalarType::Float: {
      float *self_ptr = self.data_ptr<float>();
      float other_val = other.to<float>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    case at::ScalarType::Double: {
      double *self_ptr = self.data_ptr<double>();
      double other_val = other.to<double>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    case at::ScalarType::Int: {
      int32_t *self_ptr = self.data_ptr<int32_t>();
      int32_t other_val = other.to<int32_t>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    case at::ScalarType::Long: {
      int64_t *self_ptr = self.data_ptr<int64_t>();
      int64_t other_val = other.to<int64_t>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    case at::ScalarType::Bool: {
      bool *self_ptr = self.data_ptr<bool>();
      bool other_val = other.to<bool>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    case at::ScalarType::Byte: {
      uint8_t *self_ptr = self.data_ptr<uint8_t>();
      uint8_t other_val = other.to<uint8_t>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    case at::ScalarType::Char: {
      int8_t *self_ptr = self.data_ptr<int8_t>();
      int8_t other_val = other.to<int8_t>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    case at::ScalarType::Short: {
      int16_t *self_ptr = self.data_ptr<int16_t>();
      int16_t other_val = other.to<int16_t>();
      NeScalarImplDpu(out_ptr, self_ptr, other_val, numel);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported data type for PrivateUse1 ne.Scalar_out");
  }

  return out;
}

template <typename T>
void BitwiseAndImplDpu(T *out, T *self, T *other, int64_t numel) {
  for (int64_t i = 0; i < numel; ++i) {
    out[i] = self[i] & other[i];
  }
}

at::Tensor &at_dpu::native::DPUNativeFunctions::bitwise_and_out(const at::Tensor &self, const at::Tensor &other,
                                                               at::Tensor &out) {
  // Get tensor data type
  auto dtype = self.scalar_type();

  // Check output tensor matches input types
  TORCH_CHECK(out.scalar_type() == dtype, "bitwise_and_out: output tensor must have same dtype as inputs");

  // Check shapes match
  TORCH_CHECK(self.sizes() == other.sizes(), "bitwise_and_out: self and other must have the same shape");
  TORCH_CHECK(self.sizes() == out.sizes(), "bitwise_and_out: self and out must have the same shape");

  int64_t numel = self.numel();

  // Select implementation based on data type
  switch (dtype) {
    case at::ScalarType::Bool: {
      bool *out_ptr = out.data_ptr<bool>();
      bool *self_ptr = self.data_ptr<bool>();
      bool *other_ptr = other.data_ptr<bool>();
      BitwiseAndImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Byte: {
      uint8_t *out_ptr = out.data_ptr<uint8_t>();
      uint8_t *self_ptr = self.data_ptr<uint8_t>();
      uint8_t *other_ptr = other.data_ptr<uint8_t>();
      BitwiseAndImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Char: {
      int8_t *out_ptr = out.data_ptr<int8_t>();
      int8_t *self_ptr = self.data_ptr<int8_t>();
      int8_t *other_ptr = other.data_ptr<int8_t>();
      BitwiseAndImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Short: {
      int16_t *out_ptr = out.data_ptr<int16_t>();
      int16_t *self_ptr = self.data_ptr<int16_t>();
      int16_t *other_ptr = other.data_ptr<int16_t>();
      BitwiseAndImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Int: {
      int32_t *out_ptr = out.data_ptr<int32_t>();
      int32_t *self_ptr = self.data_ptr<int32_t>();
      int32_t *other_ptr = other.data_ptr<int32_t>();
      BitwiseAndImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    case at::ScalarType::Long: {
      int64_t *out_ptr = out.data_ptr<int64_t>();
      int64_t *self_ptr = self.data_ptr<int64_t>();
      int64_t *other_ptr = other.data_ptr<int64_t>();
      BitwiseAndImplDpu(out_ptr, self_ptr, other_ptr, numel);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported data type for PrivateUse1 bitwise_and.Tensor_out");
  }

  return out;
}
