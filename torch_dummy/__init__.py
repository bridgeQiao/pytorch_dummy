import torch
import torch_dummy
import torch_dummy.dpu

torch.utils.rename_privateuse1_backend("dpu")
torch._register_device_module('dpu', torch_dummy.dpu)
