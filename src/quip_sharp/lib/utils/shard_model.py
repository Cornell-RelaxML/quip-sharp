"""Simple sharded model
"""
import glog
import torch
from torch import nn

from . import graph_wrapper


def convert_args(args, kwargs, device, dtype):

    def convert_tensor(tensor):
        if tensor.dtype == torch.float16 or tensor.dtype == torch.float32:
            tensor = tensor.to(dtype)
        return tensor.to(device)

    dev_args = []
    for i in range(len(args)):
        dev_args.append(
            convert_tensor(args[i]) if isinstance(args[i], torch.Tensor
                                                  ) else args[i])
    for i in kwargs:
        if isinstance(kwargs[i], torch.Tensor):
            kwargs[i] = convert_tensor(kwargs[i])
    return dev_args, kwargs


class Shard(nn.Module):

    def __init__(self, layers, arg_fn):
        super().__init__()
        self.layers = layers
        self.arg_fn = arg_fn

    def forward(self, *args, **kwargs):
        for layer in self.layers:
            output = layer(*args, **kwargs)
            args, kwargs = self.arg_fn(output, args, kwargs)
        return args, kwargs


class ShardTransformer(nn.Module):

    def __init__(self,
                 shards,
                 output_layer,
                 grad_ckpt,
                 train_mode,
                 to_float=True):
        super().__init__()

        # shards is list of [(device, arg_fn, modulelist)]

        self.shards = nn.ModuleList([_['shard'] for _ in shards])
        self.devices = [_['device'] for _ in shards]

        from lib.linear.fused_quantized_linear import FusedQuantizedLinear
        from lib.linear.quantized_linear import QuantizedLinear
        for name, module in self.shards.named_modules():
            if isinstance(module, QuantizedLinear) or isinstance(
                    module, FusedQuantizedLinear):
                module.grad_ckpt = grad_ckpt
                module.train_mode = train_mode
        for i in range(len(shards)):
            device = self.devices[i]
            if to_float:
                self.shards[i].float()
            #self.shards[i] = graph_wrapper.get_graph_wrapper(Shard, device)(self.shards[i], shards[i]['arg_fn']).to(device)
            self.shards[i] = Shard(self.shards[i],
                                   shards[i]['arg_fn']).to(device)
        self.dtype = torch.float32 if to_float else torch.float16
        self.output_layer = output_layer['layer'].to(0)
        self.output_layer_fn = output_layer['fn']
        self.grad_ckpt = grad_ckpt

    def manifest(self, *args, **kwargs):
        for i in range(len(self.shards)):
            device = self.devices[i]
            glog.info(f'manifesting layers on gpu {device}')
            args, kwargs = convert_args(args, kwargs, device, self.dtype)
            self.shards[i](*args, **kwargs)

    def shard_wrapper(self, input):
        i, args, kwargs = input
        return self.shards[i](*args, **kwargs)

    def ckpt_shard(self, i, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(self.shard_wrapper,
                                                 (i, args, kwargs),
                                                 use_reentrant=False)

    def forward(self, *args, **kwargs):
        for i in range(len(self.shards)):
            device = self.devices[i]
            args, kwargs = convert_args(args, kwargs, device, self.dtype)
            if self.grad_ckpt:
                args, kwargs = self.ckpt_shard(i, *args, **kwargs)
            else:
                args, kwargs = self.shards[i](*args, **kwargs)

        return self.output_layer(self.output_layer_fn(args, kwargs).to(0))
