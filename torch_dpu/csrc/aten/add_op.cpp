#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/Allocator.h>
#include <cstdio>
#include <torch/extension.h>

#include "torch_dpu/csrc/core/dpu/dpu_exception.h"
#include "torch_dpu/csrc/core/dpu/dpu_storage_impl.h"
#include "torch_dpu/csrc/core/dpu/dpu_tensor_impl.h"

at::Tensor empty_dummy(c10::IntArrayRef size,
                     c10::optional<at::ScalarType> dtype_opt,
                     c10::optional<c10::Layout> layout_opt,
                     c10::optional<c10::Device> device_opt,
                     c10::optional<bool> pin_memory_opt,
                     c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto device_ = c10::device_or_default(device_opt);
  AT_ASSERT(device_.type() == c10::DeviceType::PrivateUse1,
            OPS_ERROR(ErrCode::PARAM));
  c10::Allocator *allocator = c10::GetAllocator(c10::DeviceType::CPU);
  int64_t nelements = c10::multiply_integers(size);
  auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));
  int64_t size_bytes = nelements * dtype.itemsize();
  c10::intrusive_ptr<c10::StorageImpl> storage_impl =
      c10::make_intrusive<torch_dpu::DPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes,
          allocator->allocate(size_bytes), allocator, true);

  auto tensor =
      at::detail::make_tensor<torch_dpu::DPUTensorImpl>(storage_impl, dtype);

  // Default at::TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }
  auto memory_format =
      memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous,
              "Only c10::MemoryFormat::Contiguous is supported for creating a "
              "npu tensor",
              OPS_ERROR(ErrCode::NOT_SUPPORT));
  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  // StorageDescHelper::SetDesc(tensor, size, tensor.strides());

  return tensor;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.out", [](const at::Tensor &self, const at::Tensor &other, const c10::Scalar& scalar, at::Tensor &out) -> at::Tensor & {
    // 这里实现你的自定义 add 逻辑
    printf("    Using custom add for PrivateUse1 backend");
    out = self + other;
    return out; // 这里只是一个示例，你可以根据需要修改
  });
  m.impl("empty.memory_format", TORCH_FN(empty_dummy));
  m.impl("to.dtype",
         [](const at::Tensor &self, c10::ScalarType dtype, bool non_blocking,
            bool copy, c10::optional<c10::MemoryFormat> memory_format) {
           if (self.scalar_type() == dtype && !copy) {
             return self;
           }

           c10::TensorOptions options = self.options().dtype(dtype);
           at::Tensor result = at::empty_like(self, options, memory_format);

           at::native::copy_(result, self, non_blocking);

           return result;
         });

  m.impl("to.device", [](const at::Tensor &self, c10::Device device,
                         c10::ScalarType dtype, bool non_blocking, bool copy,
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
  });
}

// void custom_cpu_fallback(const c10::OperatorHandle &op,
//                          torch::jit::Stack *stack) {
//   // Add some hints about new devices that do not support and need to fall back
//   // to cpu
//   at::native::cpu_fallback(op, stack);
// }

// TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
//   m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
// }
