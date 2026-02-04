#include "torch_dpu/csrc/core/dpu/dpu_exception.h"

#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <unordered_map>

std::unordered_map<SubModule, std::string> submoduleMap = {{SubModule::PTA, "PTA"},
                                                           {SubModule::OPS, "OPS"},
                                                           {SubModule::DIST, "DIST"},
                                                           {SubModule::GRAPH, "GRAPH"},
                                                           {SubModule::PROF, "PROF"}};

std::unordered_map<ErrCode, std::string> errCodeMap = {{ErrCode::SUC, "success"},
                                                       {ErrCode::PARAM, "invalid parameter"},
                                                       {ErrCode::TYPE, "invalid type"},
                                                       {ErrCode::VALUE, "invalid value"},
                                                       {ErrCode::PTR, "invalid pointer"},
                                                       {ErrCode::INTERNAL, "internal error"},
                                                       {ErrCode::MEMORY, "memory error"},
                                                       {ErrCode::NOT_SUPPORT, "feature not supported"},
                                                       {ErrCode::NOT_FOUND, "resource not found"},
                                                       {ErrCode::UNAVAIL, "resource unavailable"},
                                                       {ErrCode::SYSCALL, "system call failed"},
                                                       {ErrCode::TIMEOUT, "timeout error"},
                                                       {ErrCode::PERMISSION, "permission error"},
                                                       {ErrCode::ACL, "call acl api failed"},
                                                       {ErrCode::HCCL, "call hccl api failed"},
                                                       {ErrCode::GE, "call ge api failed"}};

std::string formatErrorCode(SubModule submodule, ErrCode errorCode) {
  std::ostringstream oss;
  int deviceIndex = -1;
  // c10_npu::GetDevice(&deviceIndex);
  // auto rank_id = c10_npu::option::OptionsManager::GetRankId();
  oss << "\n[ERROR] " << " (PID:" << getpid() << ", Device:" << deviceIndex << ") ";
  oss << "ERR" << std::setw(2) << std::setfill('0') << static_cast<int>(submodule);
  oss << std::setw(3) << std::setfill('0') << static_cast<int>(errorCode);
  oss << " " << submoduleMap[submodule] << " " << errCodeMap[errorCode];

  return oss.str();
}
