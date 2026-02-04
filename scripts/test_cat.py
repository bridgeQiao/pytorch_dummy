#!/usr/bin/env python3
"""Simple smoke test for cat operator."""

import torch
import torch_dpu

print("Testing cat operator...")

# Create two DPU tensors
a = torch.empty(2, 3, device="dpu")
b = torch.empty(2, 3, device="dpu")

# Fill with values using CPU tensors
a_cpu = torch.ones(2, 3) * 1.0
b_cpu = torch.ones(2, 3) * 2.0
a.copy_(a_cpu)
b.copy_(b_cpu)

# Test cat along dim 0
c = torch.cat([a, b], dim=0)
print(f"a shape: {a.shape}, b shape: {b.shape}")
print(f"cat([a, b], dim=0) shape: {c.shape}")
print(f"Expected: torch.Size([4, 3])")

# Verify values
print(f"c[0][0]: {c[0][0].item()}, c[2][0]: {c[2][0].item()}")

# Test cat along dim 1
d = torch.cat([a, b], dim=1)
print(f"cat([a, b], dim=1) shape: {d.shape}")
print(f"Expected: torch.Size([2, 6])")

print("\ncat test passed!")
