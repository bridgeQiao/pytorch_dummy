#include <c10/core/ScalarType.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
// #include "torch_dummy/csrc/core/dummy/DPUStream.h"

// #include "torch_dummy/csrc/framework/StorageDescHelper.h"
#include "torch_dummy/csrc/dummy_tensor_impl.h"
#include "torch_dummy/csrc/dummy_storage_impl.h"

namespace torch_dummy
{
  DPUTensorImpl::DPUTensorImpl(c10::Storage &&storage, const caffe2::TypeMeta &data_type)
      : c10::TensorImpl(std::move(storage),
                        c10::DispatchKeySet{c10::DispatchKey::PrivateUse1,
                                            c10::DispatchKey::AutogradPrivateUse1},
                        data_type)
  {
    is_non_overlapping_and_dense_ = false;
  }

  void DPUTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl)
  {
    copy_tensor_metadata(
        impl.get(),
        this,
        version_counter(),
        allow_tensor_metadata_change());
    refresh_numel();
    refresh_contiguous();
  }

  c10::intrusive_ptr<c10::TensorImpl> DPUTensorImpl::shallow_copy_and_detach(
      const c10::VariableVersion &version_counter,
      bool allow_tensor_metadata_change) const
  {
    auto impl = c10::make_intrusive<DPUTensorImpl>(c10::Storage(this->storage()), this->data_type_);
    copy_tensor_metadata(
        this,
        impl.get(),
        version_counter,
        allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  c10::intrusive_ptr<c10::TensorImpl> DPUTensorImpl::shallow_copy_and_detach(
      c10::VariableVersion &&version_counter,
      bool allow_tensor_metadata_change) const
  {
    auto impl = c10::make_intrusive<DPUTensorImpl>(c10::Storage(this->storage()), this->data_type_);
    copy_tensor_metadata(
        this,
        impl.get(),
        std::move(version_counter),
        allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }
  DPUTensorImpl::~DPUTensorImpl() {}
}
