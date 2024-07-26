#pragma once

#include "torch_dpu/csrc/core/dpu_storage_impl.h"
#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>

namespace torch_dpu {

// DPUTensorImpl class is derived from c10::TensorImpl, and it is only used to
// handle an DPU tensor. Its scope is just to handle an DPUTensor.
class DPUTensorImpl : public c10::TensorImpl {
public:
  explicit DPUTensorImpl(c10::Storage &&storage,
                         const caffe2::TypeMeta &data_type);

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl> &impl) final;

  c10::intrusive_ptr<c10::TensorImpl>
  shallow_copy_and_detach(const c10::VariableVersion &version_counter,
                          bool allow_tensor_metadata_change) const final;
  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<c10::TensorImpl>
  shallow_copy_and_detach(c10::VariableVersion &&version_counter,
                          bool allow_tensor_metadata_change) const final;

public:
  DPUTensorImpl(const DPUTensorImpl &) = delete;
  DPUTensorImpl &operator=(const DPUTensorImpl &) = delete;
  DPUTensorImpl(DPUTensorImpl &&) = default;
  DPUTensorImpl &operator=(DPUTensorImpl &&) = default;
  ~DPUTensorImpl();
};

} // namespace torch_dpu
