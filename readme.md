# Overview

This project is a PyTorch backend modeled after [ascend/pytorch](https://github.com/ascend/pytorch), built on top of `PrivateUse1`.

**Notice:** Minimum supported PyTorch version is `2.7.0`.

# Usage

## Build the extension
```bash
cd pytorch_dummy
# Generate codegen stubs.
# Use the Python binary from your target environment (python or python3, or an absolute path).
# If this fails, compare codegen/gen_backend_stubs.py with torchgen/gen_backend_stubs.py.
bash generate_code.sh python3

# Build and install the extension (editable).
pip install -e . --no-build-isolation
```
You must install PyTorch first. A successful build produces `torch_dpu/_C.*.so`.

## Use it

Start Python and run:
```python
import torch
import torch_dpu
# Only aten::empty and aten::add are supported.
# Implement additional kernels if you need more ops.
x = torch.ones([3,3], dtype=torch.int32).to('dpu')
y = torch.ones([3,3], dtype=torch.int32).to('dpu')
x + y
```

## Troubleshooting

- `symbol not found in flat namespace '__xxxx'`
  - Ensure your build uses `--no-build-isolation` so it links against the PyTorch already installed in your environment.
  - Remove any old `torch_dpu/_C.*.so` artifacts and rebuild if the error persists.
