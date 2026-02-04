# Adding New Operators

This guide describes the minimal workflow to add a new operator to the `dpu` backend.

## 1) Declare the Op
Edit `torch_dpu/csrc/aten/dpu_native_functions.yaml` and add the op name to `supported`.
Example:
```yaml
supported:
  - empty.memory_format
  - empty_strided
  - add.out
  - to.device
  - to.dtype
  - relu.out   # new
```

Notes:
- Use the ATen op name and overload (e.g., `add.out`).
- Keep the list minimal; only add ops you will implement.

## 2) Implement the Kernel
Add the implementation in C++.

Where to put code:
- `torch_dpu/csrc/aten/ops/` for most ops
- `torch_dpu/csrc/aten/` for shared or specialized ops

Example pattern (simplified):
```cpp
at::Tensor& at_dpu::native::DPUNativeFunctions::relu_out(
    const at::Tensor& self, at::Tensor& out) {
  // 1) Validate inputs
  // 2) Dispatch based on dtype
  // 3) Write output
  return out;
}
```

Tips:
- Follow existing patterns in `binary_ops.cpp` and `tensor_creation.cpp`.
- Add `TORCH_CHECK` guards for unsupported dtypes or layouts.
- Use `self.numel()` and raw `data_ptr<T>()` for simple elementwise ops.

## 3) Regenerate Backend Stubs
Run codegen with the Python from your target environment:
```bash
bash generate_code.sh python3
```
This regenerates the backend stubs used by dispatch.

## 4) Rebuild and Test
Rebuild the extension and verify the op works on `dpu` tensors.

Example smoke test:
```python
import torch
import torch_dpu

x = torch.randn(4, 4, device="dpu")
# call your new op here
```

## Common Pitfalls
- Declaring an op but forgetting to implement it.
- Supporting an op without handling all required overloads.
- Missing dtype checks for `out` or `self`.

