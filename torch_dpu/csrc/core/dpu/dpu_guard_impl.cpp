#include "torch_dpu/csrc/core/dpu_guard_impl.h"
#include "torch_dpu/csrc/core/dpu_storage_impl.h"

#include <c10/core/StorageImpl.h>

namespace c10_dummy {
namespace impl {

constexpr c10::DeviceType DPUGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(PrivateUse1, DPUGuardImpl);

int rename_privateuse1_backend() {
  c10::register_privateuse1_backend("dpu");
  c10::SetStorageImplCreate(
      c10::DeviceType::PrivateUse1,
      (c10::StorageImplCreateHelper)&torch_dpu::make_dummy_storage_impl);
  return 0;
}
static const int _temp_dummy = rename_privateuse1_backend();

} // namespace impl
} // namespace c10_dummy
