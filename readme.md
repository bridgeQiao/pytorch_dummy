# Brief

仿照[ascend/pytorch](https://github.com/ascend/pytorch)写的pytorch backend，基于PrivateUse1，torch==2.3，理论上 2.2 应该也可以跑，没试过。吐槽下，ascend_pytorch的setup.py也太老了，不更新成最新的。

# useage

## compile library
```bash
cd pytorch_dummy
python3 setup.py build_ext --inplace
```
需要预先安装好torch包，正常编译会得到`torch_dpu/_C.*.so`。

## use it

启动python，然后输入：
```python
import torch
import torch_dummy
# only support aten::empty/aten::add, you need implement more kernels if you want
x = torch.empty((2,2,), device='privateuseone')
y = torch.empty((2,2,), device='privateuseone')
x + y
```

