"""
Utilities for fine tuning
"""
import copy
from operator import attrgetter

import glog
import torch
from torch import nn

from lib import codebook, utils
from lib.linear import *

from . import quip


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, args):
    layer = layer.to(device)

    susv_params, params = utils.extract_susv_params(layer)
    optim = utils.get_susv_adam(susv_params, params, args)

    best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss}')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    worse_ct = 0
    position_ids = None

    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            with torch.autocast(device_type='cuda',
                                dtype=torch.float16,
                                enabled=True):
                output = layer(source.to(device), position_ids=position_ids)[0]
                loss = nn.MSELoss()(output, targets.to(device))
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
            if test_loss < best_loss:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(layer.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    del optim, train_dl, valid_dl

    layer.load_state_dict(best_sd)
    utils.clean()
    layer = layer.cpu()


def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)

    codebook_id = codebook.get_id(args.codebook)

    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype),
                   cb.version)
    shared_kwargs = {
        'rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'resid_scale_override': args.resid_scale_override,
        'bias': False,
        'train_mode': args.ft_train_mode,
        'grad_ckpt': args.ft_grad_ckpt,
    }

    for quant_i, (linear_attr, name) in enumerate(quant_order):
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        if orig_linear.bias is not None:
            # not implemented yet
            raise Exception
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        hessian_path = f'{args.hessian_path}/{idx}_{name}.pt'
        with torch.no_grad():
            if isinstance(orig_linear, FusedLinear):
                weights = torch.split(orig_linear.weight,
                                      orig_linear.fuse_sizes, 0)
            else:
                weights = [orig_linear.weight]
            quip.quantize_linear(weights, save_path, hessian_path, cb, args,
                                 device)
            saved_linear = torch.load(save_path,
                                      map_location=torch.device('cpu'))
            if saved_linear['fused']:
                quant_linear = FusedQuantizedLinear(
                    -1, [_[0] for _ in saved_linear['shapes']],
                    saved_linear['shapes'][0][1],
                    sum([_[0] for _ in saved_linear['shapes']]), *shared_args,
                    **shared_kwargs)
                for i in range(len(saved_linear['scales'])):
                    quant_linear.fuse_scales[i].copy_(
                        saved_linear['scales'][i])
            else:
                quant_linear = QuantizedLinear(saved_linear['shapes'][0][1],
                                               saved_linear['shapes'][0][0],
                                               *shared_args, **shared_kwargs)
            utils.unpack_quip(quant_linear, saved_linear, codebook_id,
                              cb.codesz)
        quant_linear.SU = nn.Parameter(quant_linear.SU.float(),
                                       requires_grad=True)
        quant_linear.SV = nn.Parameter(quant_linear.SV.float(),
                                       requires_grad=True)
        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            quant_linear)
        if quant_i < len(quant_order) - 1:
            finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
                                   train_dl, valid_dl, args)

    with torch.no_grad():
        utils.clean()
        for i, (linear_attr, name) in enumerate(quant_order):
            utils.save_susv(
                attrgetter(linear_attr)(mixed_layer),
                f'{args.save_path}/{idx}_{name}.pt')

    mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)


def finetune_susv_e2e(model, orig_logits, emb, position_ids, attention_mask,
                      save_fn, args):

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear) or isinstance(
                module, FusedQuantizedLinear):
            module.SU = nn.Parameter(module.SU.float(), requires_grad=True)
            module.SV = nn.Parameter(module.SV.float(), requires_grad=True)
    model.float()

    train_dl, valid_dl = utils.split_data(emb, orig_logits, args)

    susv_params, params = utils.extract_susv_params(model)
    optim = utils.get_susv_adam(susv_params, params, args)

    best_loss = utils.calculate_ce_loss(model, position_ids, attention_mask,
                                        valid_dl)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            with torch.autocast(device_type='cuda',
                                dtype=torch.float16,
                                enabled=True):
                output = model(
                    source,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )[:, :-1].contiguous()
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             targets.to(0).view(
                                                 -1, targets.shape[-1]))
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss(model, position_ids,
                                                attention_mask, valid_dl)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    with torch.no_grad():
        model.load_state_dict(best_sd)
        save_fn(model)
