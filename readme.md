# PyTorch DPU Backend

A minimal PyTorch backend built on `PrivateUse1`, modeled after [ascend/pytorch](https://github.com/ascend/pytorch).

**Minimum PyTorch version:** 2.7.0

## Build

```bash
# Install PyTorch first
pip install torch==2.7.0

# Generate codegen stubs
bash generate_code.sh python3

# Build and install (editable)
pip install -e . --no-build-isolation
```

A successful build produces `torch_dpu/_C.*.so`.

## Quick Start

```python
import torch
import torch_dpu

# Create tensors on DPU device
x = torch.ones(3, 3, device="dpu")
y = torch.empty(3, 3, device="dpu").fill_(2.0)

# Basic operations
z = x + y                     # addition
result = torch.cat([x, y])    # concatenation
```

## Supported Operators

| Category | Operators |
|----------|-----------|
| Creation | `empty`, `empty_strided` |
| Math | `add`, `isfinite` |
| Comparison | `eq`, `ne` |
| Manipulation | `view`, `as_strided`, `resize_`, `cat`, `fill_`, `set_` |
| Copy | `to`, `copy_` |
| Access | `item()` |

See `docs/skills.md` for adding new operators.

## Troubleshooting

- **Symbol not found errors**: Use `--no-build-isolation` and remove old `torch_dpu/_C.*.so` artifacts before rebuilding
- **Compilation fails**: Report errors - do not proceed to verification
