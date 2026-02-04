#!/usr/bin/env python3
"""Smoke test for resize_ and cat operators."""

import torch
import torch_dpu

print("Testing resize_ operator...")
# Create a DPU tensor using empty (zeros not implemented)
x = torch.empty(5, device="dpu")
print(f"Original shape: {x.shape}")

# Resize in-place
x.resize_(10)
print(f"After resize_(10): {x.shape}")

x.resize_(2, 3)
print(f"After resize_(2, 3): {x.shape}")

print("\nTesting cat operator...")
# Test cat along dim 0
a = torch.empty(2, 3, device="dpu")
b = torch.empty(2, 3, device="dpu")
# Fill with some values
a.fill_(1.0)
b.fill_(2.0)

c = torch.cat([a, b], dim=0)
print(f"a shape: {a.shape}, b shape: {b.shape}")
print(f"cat([a, b], dim=0) shape: {c.shape}")

# Test cat along dim 1
d = torch.cat([a, b], dim=1)
print(f"cat([a, b], dim=1) shape: {d.shape}")

print("\nAll tests passed!")
