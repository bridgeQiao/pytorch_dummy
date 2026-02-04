# Skills

## Adding Ops

### Purpose
Provide a minimal, repeatable workflow for adding operators to the `dpu`
backend and rebuilding the extension so dispatch stays in sync.

### Scope
Use this workflow whenever you add or update operators declared in
`torch_dpu/csrc/aten/dpu_native_functions.yaml`.

### When To Run
Run the compile step after finishing any new operator implementation or any
change to the supported op list.

### Workflow
1. Add the operator to `torch_dpu/csrc/aten/dpu_native_functions.yaml`.

2. Regenerate backend stubs to update function declarations:

```bash
bash generate_code.sh python3
```

3. Implement the kernel in the appropriate C++ file based on the function declarations in `torch_dpu/csrc/aten/PrivateUse1NativeFunctions.h`.

4. Compile the extension:

```bash
pip install -e . --no-build-isolation
```

**Important**: If this step fails, report the compilation errors. Do not proceed to verification if the build fails.

### Verification
Run smoke tests to confirm the operators work correctly:

```bash
# Test view and as_strided
python scripts/smoke_view.py

# Test resize_ and cat
python scripts/smoke_cat.py

# Test _local_scalar_dense
python scripts/smoke_scalar.py
```

### Implemented Operators

The following operators have been implemented for the DPU backend:

| Operator | File | Description |
|----------|------|-------------|
| `empty` | `ops/tensor_creation.cpp` | Create uninitialized tensor |
| `empty_strided` | `ops/tensor_creation.cpp` | Create tensor with custom strides |
| `add.out` | `binary_ops.cpp` | Element-wise addition |
| `view` | `unary_ops.cpp` | Reshape tensor (view) |
| `as_strided` | `unary_ops.cpp` | Create tensor view with custom strides |
| `resize_` | `unary_ops.cpp` | Resize tensor in-place |
| `cat` | `unary_ops.cpp` | Concatenate tensors along dimension |
| `_local_scalar_dense` | `unary_ops.cpp` | Extract scalar from single-element tensor |
| `_copy_from` | `unary_ops.cpp` | Copy data from tensor |
| `_to_copy` | `unary_ops.cpp` | Copy tensor with options |

### Troubleshooting
- If you see missing kernel linker errors, ensure the op appears in
  `torch_dpu/csrc/aten/dpu_native_functions.yaml` and the corresponding C++
  implementation exists.
- If dispatch falls back to CPU, confirm the tensor device is `dpu` and
  rebuild the extension after codegen.
