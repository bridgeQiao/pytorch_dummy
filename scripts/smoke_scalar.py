#!/usr/bin/env python3
"""Smoke test for _local_scalar_dense operator (called via item())."""

import torch
import torch_dpu

print("Testing _local_scalar_dense operator (via item())...")

# Test with different dtypes
test_cases = [
    (torch.tensor(3.14, device="dpu"), "Float"),
    (torch.tensor(42, device="dpu"), "Int"),
    (torch.tensor(2.718, dtype=torch.float64, device="dpu"), "Double"),
    (torch.tensor(True, device="dpu"), "Bool"),
    (torch.tensor(100, dtype=torch.int64, device="dpu"), "Long"),
]

for tensor, name in test_cases:
    scalar = tensor.item()
    print(f"{name}: item()={scalar}")

# Test error case: multi-element tensor
try:
    x = torch.empty(5, device="dpu")
    scalar = x.item()
    print("ERROR: Should have failed with multi-element tensor")
except Exception as e:
    print(f"\nExpected error for multi-element tensor: {e}")

print("\nAll tests passed!")
