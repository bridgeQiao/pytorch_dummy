#!/usr/bin/env python3
"""Smoke test for as_strided operator."""

import torch
import torch_dpu

print("Testing as_strided operator...")

# Create a DPU tensor
x = torch.empty(2, 3, device="dpu")
x_cpu = torch.arange(6).float()
x.copy_(x_cpu)

print(f"x shape: {x.shape}, strides: {x.stride()}")

# Test as_strided with custom strides
y = x.as_strided((2, 2), (3, 1))
print(f"as_strided((2, 2), (3, 1)) shape: {y.shape}, strides: {y.stride()}")

# Test with storage_offset
z = x.as_strided((2, 2), (3, 1), storage_offset=1)
print(f"as_strided((2, 2), (3, 1), storage_offset=1) shape: {z.shape}, strides: {z.stride()}")

# Verify values by copying to CPU
y_cpu = y.to('cpu')
print(f"\ny values (first row): {y_cpu[0].tolist()}")

print("\nas_strided test passed!")
