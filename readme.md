# Brief

仿照[ascend/pytorch](https://github.com/ascend/pytorch)写的pytorch backend，基于PrivateUse1，torch==2.3，理论上 2.2 应该也可以跑，没试过。

# useage

## compile library
```bash
cd pytorch_dummy
python3 setup.py build_ext --inplace
```
需要预先安装好torch包，正常编译会得到`dummy_backend.*.so`。

## use it

启动python，然后输入：
```python
import torch
torch.ops.load_library('dummy_backend.cpython-312-darwin.so')
# only support aten::empty/aten::add, you need implement more kernels if you want
x = torch.empty((2,2,), device='privateuseone')
y = torch.empty((2,2,), device='privateuseone')
x + y
```

