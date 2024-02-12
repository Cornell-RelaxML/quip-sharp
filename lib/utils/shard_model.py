"""Simple sharded model
"""
import glog
import torch
from torch import nn


class ShardModel(nn.Module):

    def __init__(self, quant_model, nshards, grad_ckpt, train_mode):
        super().__init__()
        self.shards = nn.ModuleList(
            [nn.ModuleList([]) for _ in range(nshards)])
        nlayers = len(quant_model.model.layers)
        for i in range(nshards):
            for j in range(int(nlayers * i / nshards),
                           int(nlayers * (i + 1) / nshards)):
                self.shards[i].append(quant_model.model.layers[j])
                # hack to get around circular import, fix later
                from lib.linear.fused_quantized_linear import \
                    FusedQuantizedLinear
                from lib.linear.quantized_linear import QuantizedLinear
                for name, module in self.shards[i][-1].named_modules():
                    if isinstance(module, QuantizedLinear) or isinstance(
                            module, FusedQuantizedLinear):
                        # need to set both quantized linear and decoder layer to use grad ckpt
                        module.grad_ckpt = grad_ckpt
                        module.train_mode = train_mode

        self.norm = quant_model.model.norm.to(0)
        self.lm_head = quant_model.lm_head.to(0)
        self.grad_ckpt = grad_ckpt

    def manifest(self, emb, position_ids, attention_mask):
        for i in range(len(self.shards)):
            glog.info(f'manifesting layers on gpu {i}')
            emb = emb.to(i)
            position_ids = position_ids.to(i)
            attention_mask = attention_mask.to(i)
            for j in range(len(self.shards[i])):
                self.shards[i][j].to(i)
                emb = self.shards[i][j](emb,
                                        position_ids=position_ids,
                                        attention_mask=attention_mask,
                                        output_attentions=False,
                                        use_cache=False)[0]
                self.shards[i][j].cpu()
            self.shards[i].to(i)

    def shard_wrapper(self, input):
        i, j, args, kwargs = input
        return self.shards[i][j](*args, **kwargs)

    def ckpt_shard(self, i, j, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(self.shard_wrapper,
                                                 (i, j, args, kwargs),
                                                 use_reentrant=False)

    def forward(self, emb, position_ids, attention_mask):
        for i in range(len(self.shards)):
            emb = emb.to(i)
            position_ids = position_ids.to(i)
            attention_mask = attention_mask.to(i)
            for j in range(len(self.shards[i])):
                if self.grad_ckpt:
                    emb = self.ckpt_shard(i,
                                          j,
                                          emb,
                                          position_ids=position_ids,
                                          attention_mask=attention_mask,
                                          output_attentions=False,
                                          use_cache=False)[0]
                else:
                    emb = self.shards[i][j](emb,
                                            position_ids=position_ids,
                                            attention_mask=attention_mask,
                                            output_attentions=False,
                                            use_cache=False)[0]
        return self.lm_head(self.norm(emb.to(0)))
