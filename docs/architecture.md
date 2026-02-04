# Architecture Overview

## Goals
- Provide a minimal PyTorch backend on top of `PrivateUse1`.
- Register a `dpu` device that can host custom kernels.
- Keep the backend small and easy to extend.

## Top-Level Layout
- `torch_dpu/__init__.py` registers the backend and device module.
- `torch_dpu/dpu/` is the Python-facing device module (currently empty).
- `torch_dpu/csrc/` contains all C++ backend code.
- `torch_dpu/csrc/aten/` implements ATen kernels for the backend.
- `torch_dpu/csrc/core/dpu/` implements tensor/storage/guard plumbing.
- `codegen/` contains codegen helpers for backend stubs.

## Registration Flow
1. Importing `torch_dpu` triggers device registration:
   - Rename `PrivateUse1` to `dpu`.
   - Register `torch_dpu.dpu` as the device module.
2. From that point, `device="dpu"` is valid in PyTorch APIs.

## Kernel Dispatch
- `torch_dpu/csrc/aten/dpu_native_functions.yaml` declares supported ops.
- Codegen produces `PrivateUse1NativeFunctions.h` and other glue.
- Implementations live in `torch_dpu/csrc/aten/ops/` and related files.

## Tensor and Storage
- `DPUStorageImpl` wraps CPU-allocated storage for backend tensors.
- `DPUTensorImpl` is the custom tensor implementation.
- This is sufficient to demonstrate a working `PrivateUse1` backend.

## Build and Codegen Entry Points
- `generate_code.sh` runs codegen for backend stubs.
- `codegen/gen_backend_stubs.py` is the Python codegen entry.

