import torch
torch.set_grad_enabled(False)
from lib.utils.unsafe_import import model_from_hf_path
from lib.utils  import matmul_hadU, matmul_hadUt
from scipy.optimize import minimize_scalar

def opt_err_cvx(fn):
    res = minimize_scalar(fn, bounds=(0.1, 10000))
    scale = res.x.item()
    err = res.fun
    return err, scale

def round(x, grid):
    dist = (x.unsqueeze(-1) - grid.unsqueeze(0))**2
    idx = dist.argmin(-1)
    return grid[idx]

def round_smart(x, min, max):
    return torch.clamp(torch.round(x + 1/2) - 1/2, min=min, max=max)

path = '/mnt/desa_data/hfized/model_v1/1_7b_e8p_2bit_nolr'

model, ms = model_from_hf_path(path)

_2bit = (torch.arange(-2, 2) + 1/2).cuda()
_4bit = (torch.arange(-8, 8) + 1/2).cuda()
_6bit = (torch.arange(-32, 32) + 1/2).cuda()
_8bit = (torch.arange(-128, 128) + 1/2).cuda().to(torch.float16)

def hack_quantize(x, bits):
    x = x.to(torch.float16)
    m, n = x.shape
    SU = (torch.randn(n).sign() + 1e-5).sign().to(x.dtype).cuda()
    x = matmul_hadUt(x * SU)
    x = x.view(-1)
    
    def fn(i):
        return (round_smart(x*i, -2**(bits-1) + 1/2, 2**(bits-1) - 1/2)/i - x).norm().cpu()
    best_err, best_scale = opt_err_cvx(fn)
    print(best_scale, best_err)

    xh = round_smart(x*best_scale, -2**(bits-1) + 1/2, 2**(bits-1) - 1/2)/best_scale
    xh = xh.view(m, n)
    xh = matmul_hadU(xh) * SU
    return xh
 
model.model.embed_tokens.weight.copy_(hack_quantize(model.model.embed_tokens.weight.clone(), 6))
model.lm_head.weight.copy_(hack_quantize(model.lm_head.weight.clone(), 6))

model.save_pretrained(f'{path}_hackquant', safe_serialization=True)
