import torch
from torch import nn


def extract_susv_params(module):
    susv_params = []
    params = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            if 'SU' in name or 'SV' in name:
                susv_params.append(param)
            else:
                params.append(param)
    return susv_params, params


def get_susv_adam(susv_params, params, args):
    return torch.optim.Adam([
        {
            'params': susv_params,
            'lr': args.ft_susv_lr
        },
        {
            'params': params,
            'lr': args.ft_lr
        },
    ])


def save_susv(module, path):
    saved_layer = torch.load(path, map_location=torch.device('cpu'))
    saved_layer['SU'] = module.SU.data.to(torch.float16)
    saved_layer['SV'] = module.SV.data.to(torch.float16)
    torch.save(saved_layer, path)


def calculate_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            total_loss += nn.MSELoss()(layer(source.to(device), position_ids=position_ids)[0],
                                       target.to(device))
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()


def calculate_ce_loss(layer, position_ids, attention_mask, dataloader):
    layer.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            output = layer(
                source,
                position_ids=position_ids,
                attention_mask=attention_mask.float())[:, :-1].contiguous()
            total_loss += nn.CrossEntropyLoss()(
                output.view(-1, output.shape[-1]),
                target.to(0).view(-1, target.shape[-1]),
            )
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()
