#include "torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h"

#include "torch_dpu/csrc/core/dpu/dpu_exception.h"
#include "torch_dpu/csrc/core/dpu/dpu_storage_impl.h"
#include "torch_dpu/csrc/core/dpu/dpu_tensor_impl.h"

at::Tensor at_dpu::native::DPUNativeFunctions::empty(
    c10::IntArrayRef size, c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt, c10::optional<bool> pin_memory_opt,
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

at::Tensor at_dpu::native::DPUNativeFunctions::empty_strided(
    c10::IntArrayRef size, c10::IntArrayRef stride,
    c10::optional<at::ScalarType> dtype_opt,
    c10::optional<c10::Layout> layout_opt,
    c10::optional<c10::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  auto device_ = c10::device_or_default(device_opt);
  AT_ASSERT(device_.type() == c10::DeviceType::PrivateUse1,
            OPS_ERROR(ErrCode::PARAM));

  TORCH_CHECK(size.size() == stride.size(),
              "size and stride must have the same length",
              OPS_ERROR(ErrCode::PARAM));

  auto layout = layout_opt.value_or(c10::kStrided);
  TORCH_CHECK(layout == c10::kStrided,
              "Only c10::kStrided is supported for creating a dpu tensor",
              OPS_ERROR(ErrCode::NOT_SUPPORT));

  c10::Allocator *allocator = c10::GetAllocator(c10::DeviceType::CPU);
  auto dtype = c10::scalarTypeToTypeMeta(dtype_or_default(dtype_opt));

  int64_t numel = c10::multiply_integers(size);
  int64_t storage_numel = 0;
  if (numel > 0) {
    int64_t max_offset = 0;
    for (size_t i = 0; i < size.size(); ++i) {
      TORCH_CHECK(stride[i] >= 0, "negative strides are not supported",
                  OPS_ERROR(ErrCode::NOT_SUPPORT));
      max_offset += (size[i] - 1) * stride[i];
    }
    storage_numel = max_offset + 1;
  }
  int64_t size_bytes = storage_numel * dtype.itemsize();

  c10::intrusive_ptr<c10::StorageImpl> storage_impl =
      c10::make_intrusive<torch_dpu::DPUStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes,
          allocator->allocate(size_bytes), allocator, true);

  auto tensor =
      at::detail::make_tensor<torch_dpu::DPUTensorImpl>(storage_impl, dtype);

  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  }

  return tensor;
}
