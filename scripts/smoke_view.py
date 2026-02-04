import torch
import torch_dpu

x = torch.empty(2, 3, device="dpu")
y = x.view(3, 2)
print(y.shape)

nc = torch.empty_strided((2, 3), (1, 2), device="dpu")
z = nc.reshape(3, 2)
print(z.shape)
