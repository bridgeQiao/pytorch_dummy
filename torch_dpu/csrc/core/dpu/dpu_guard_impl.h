#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include "torch_dpu/csrc/core/dpu_exception.h"
#include <cassert>

namespace c10_dummy {
namespace impl {

struct DPUGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::DeviceType::PrivateUse1;

  DPUGuardImpl() {}
  explicit DPUGuardImpl(c10::DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ", t,
                          PTA_ERROR(ErrCode::PARAM));
  }

  // override
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }

  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == c10::DeviceType::PrivateUse1,
                          "DeviceType must be NPU. Actual DeviceType is: ",
                          d.type(), PTA_ERROR(ErrCode::PARAM));
    return d;
  }
  c10::Device getDevice() const override {
      int device = 0;
      // mfrtGetDevice(&device);
      return c10::Device(c10::DeviceType::PrivateUse1, device);
  }
  void setDevice(c10::Device d) const override {
      // mfrtSetDevice(d.index());
  }
  void uncheckedSetDevice(c10::Device d) const noexcept override {}
  c10::Stream getStream(c10::Device d) const noexcept override {
      return c10::Stream(c10::Stream::Default::DEFAULT, d);
  }
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
      return s;
  }
  c10::DeviceIndex deviceCount() const noexcept override {
      int device_count = 0;
      // mfrtGetDeviceCount(&device_count);
      return 1;
  }
};

} // namespace impl
} // namespace c10_dummy
