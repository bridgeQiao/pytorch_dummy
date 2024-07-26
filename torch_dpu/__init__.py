import torch
import torch_dpu
import torch_dpu.dpu

import torch_dpu._C

torch.utils.rename_privateuse1_backend("dpu")
torch._register_device_module('dpu', torch_dpu.dpu)
