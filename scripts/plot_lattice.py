import math

import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize_scalar

torch.set_float32_matmul_precision('high')
torch.manual_seed(0)


def opt_err_cvx(fn):
    res = minimize_scalar(fn, bounds=(0.1, 100))
    scale = res.x.item()
    err = res.fun
    return err, scale


def round(X, grid, grid_norm):
    Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
    return grid[Xqidx]


def get_hint_curve(bit_cap=4, cols=1):

    def round_mvn(grid, dim=cols, nsamples=50000, sample_bs=500):
        X = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(dim),
            torch.eye(dim)).rsample([nsamples
                                     ]).to(grid.dtype).to(grid.device).abs()

        grid_norm = grid.norm(dim=-1)**2

        def test_s(s):
            total_err = 0
            for i in range(nsamples // sample_bs):
                sample_b = X[i * sample_bs:(i + 1) * sample_bs].cuda()
                total_err += (
                    round(sample_b * s, grid, grid_norm) / s -
                    sample_b).float().norm()**2 / torch.numel(sample_b)
            total_err = total_err / (nsamples // sample_bs)
            return total_err.cpu()

        return opt_err_cvx(test_s)

    bits = 0
    last_bits = 0
    cr = 1
    data = [[], [], []]
    while bits < bit_cap:
        base_grid = torch.arange(0, cr).to(torch.float16)
        grid = torch.cartesian_prod(*[base_grid + 1 / 2] * cols)
        if cols == 1:
            grid = grid.unsqueeze(-1)
        grid_norms = torch.sum(grid**2, dim=-1)
        norms = torch.unique(grid_norms)
        norms = norms[torch.where((norms >= (cr - 1)**2) * (norms < cr**2))[0]]
        for norm in norms[::4]:
            cb = grid[torch.where(grid_norms <= norm)[0]].cuda()
            bits = math.log(len(cb)) / math.log(2) / cols + 1
            if bits - last_bits < 0.1:
                continue
            last_bits = bits
            data[0].append(bits)
            err, scale = round_mvn(cb.cuda())
            data[1].append(err)
            data[2].append(scale)
            print(norm.item(), bits, err, scale)
            if bits > bit_cap:
                return data
        cr += 1
    return data


def get_D4_curve(bit_cap=4):

    def round_mvn(grid, nsamples=50000, sample_bs=1000):
        dim = grid.shape[-1]

        X = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(dim),
            torch.eye(dim)).rsample([nsamples]).to(grid.dtype).to(grid.device)

        grid_norm = grid.norm(dim=-1)**2

        def test_s(s):
            err = (round(X * s, grid, grid_norm) / s -
                   X).float().norm()**2 / torch.numel(X)
            return err.cpu()

        return opt_err_cvx(test_s)

    _D4_CODESZ = 4
    bits = 0
    last_bits = 0
    cr = 1
    data = [[], [], []]
    while bits < bit_cap:
        base_grid = torch.arange(-cr, cr).to(torch.float16)
        grid = torch.cartesian_prod(*[base_grid + 1 / 2] * _D4_CODESZ)
        grid = grid[torch.where(grid.sum(dim=-1) % 2 == 0)[0]]
        grid_norms = torch.sum(grid**2, dim=-1)
        norms = torch.unique(grid_norms)
        norms = norms[torch.where((norms >= (cr - 1)**2) * (norms < cr**2))[0]]
        for norm in norms[::4]:
            cb = grid[torch.where(grid_norms <= norm)[0]].cuda()
            bits = math.log(len(cb)) / math.log(2) / _D4_CODESZ
            if bits - last_bits < 0.1:
                continue
            last_bits = bits
            data[0].append(bits)
            err, scale = round_mvn(cb.cuda())
            data[1].append(err)
            data[2].append(scale)
            print(norm.item(), bits, err, scale)
            if bits > bit_cap:
                return data
        cr += 1
    return data


def get_E8_curve(bit_cap=4):

    def round_mvn(grid, nsamples=50000, sample_bs=250):
        dim = grid.shape[-1]

        X = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(dim),
            torch.eye(dim)).rsample([nsamples]).to(grid.dtype)
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 0] = -X_part[X_odd, 0]
        X = X_part

        grid_norm = grid.norm(dim=-1)**2

        def test_s(s):
            total_err = 0
            for i in range(nsamples // sample_bs):
                sample_b = X[i * sample_bs:(i + 1) * sample_bs].cuda()
                total_err += (
                    round(sample_b * s, grid, grid_norm) / s -
                    sample_b).float().norm()**2 / torch.numel(sample_b)
            total_err = total_err / (nsamples // sample_bs)
            return total_err.cpu()

        return opt_err_cvx(test_s)

    def flip_cb(cb, flips, batch_size=5000000):
        map = 1 - 2 * flips
        output = torch.zeros((len(cb), len(map), cb.shape[-1]),
                             dtype=cb.dtype,
                             device='cpu')
        map = map.unsqueeze(0)
        for i in range(math.ceil(len(cb) / batch_size)):
            next = min(len(cb), (i + 1) * batch_size)
            output[i *
                   batch_size:next] = (cb[i * batch_size:next].unsqueeze(1) *
                                       map).cpu()
        return output.reshape(-1, cb.shape[-1])

    def batched_unique(cpu_tensor, batch_size=10**9):
        res = []
        for i in range(math.ceil(len(cpu_tensor) / batch_size)):
            next = min(len(cpu_tensor), (i + 1) * batch_size)
            res.append(
                torch.unique(cpu_tensor[i * batch_size:next].cuda(),
                             dim=0).cpu())
        return torch.concat(res, dim=0)

    def combo(n, k):
        return ((n + 1).lgamma() - (k + 1).lgamma() -
                ((n - k) + 1).lgamma()).exp()

    _E8_CODESZ = 8

    int_map = 2**torch.arange(8)
    bitmap = torch.zeros(256, 8)
    for i in range(256):
        bitmap[i] = (i & int_map) != 0
    bitmap = bitmap[torch.where(bitmap.sum(dim=-1) % 2 == 0)[0]].cuda()

    bits = 0
    cr = 2
    data = [[], [], []]
    last_bits = 0
    while bits < bit_cap:
        base_grid = torch.arange(-1, cr).to(torch.float16)
        int_grid = torch.cartesian_prod(*[base_grid] * _E8_CODESZ)
        int_grid = int_grid[torch.where(int_grid.sum(dim=-1) % 2 == 0)[0]]
        hint_grid = torch.cartesian_prod(*[base_grid + 1 / 2] * _E8_CODESZ)
        hint_grid = hint_grid[torch.where(hint_grid.sum(dim=-1) % 2 == 0)[0]]
        grid = torch.concat([int_grid, hint_grid], dim=0)

        grid_norms = torch.sum(grid**2, dim=-1)
        norms = torch.unique(grid_norms)
        norms = norms[torch.where((norms >= (cr - 1)**2) * (norms < cr**2))[0]]
        for norm in norms[::4]:
            cb = grid[torch.where(grid_norms <= norm)[0]].cuda()
            cb = batched_unique(flip_cb(cb, bitmap))
            idxs = torch.where(
                ((cb[:, 1:] < 0).sum(dim=-1) <= 1) * \
                (cb[:, 1:].min(dim=-1).values >= -0.5)
            )[0]
            cb_part = cb[idxs]

            bits = math.log(len(cb)) / math.log(2) / _E8_CODESZ
            if bits - last_bits < 0.1:
                continue
            last_bits = bits
            data[0].append(bits)
            err, scale = round_mvn(cb_part.cuda())
            data[1].append(err)
            data[2].append(scale)
            print(norm.item(), bits, err, scale)
            if bits > bit_cap:
                return data
        cr += 1
    return data


def parse_cached(s):
    s = s.replace('\n', ' ')
    s = s.strip().rstrip().split(' ')
    bits = [float(_) for _ in s[1::3]]
    err = [float(_) for _ in s[2::3]]
    return bits, err


bit_cap = 3.5
hint_1c = get_hint_curve(bit_cap, 1)
hint_4c = get_hint_curve(bit_cap, 4)
hint_8c = get_hint_curve(bit_cap, 8)
D4 = get_D4_curve(bit_cap)
E8 = get_E8_curve(bit_cap)

import pickle as pkl

all_data = {
    'half_int_1col': hint_1c,
    'half_int_4col': hint_4c,
    'half_int_8col': hint_8c,
    'D4': D4,
    'E8': E8,
}

print(all_data)
pkl.dump(all_data, open('plot_data.pkl', 'wb'))
exit()

plt.rcParams["figure.figsize"] = (6, 5)
plt.cla()
box = plt.plot(hint_1c[0], hint_1c[1], 's', label='Half Integer 1 Column')[0]
plt.plot(hint_1c[0], hint_1c[1], '-', alpha=0.5, color=box._color)
box = plt.plot(hint_4c[0], hint_4c[1], 'o', label='Half Integer 4 Column')[0]
plt.plot(hint_4c[0], hint_4c[1], '-', alpha=0.5, color=box._color)
box = plt.plot(hint_8c[0], hint_8c[1], '+', label='Half Integer 8 Column')[0]
plt.plot(hint_8c[0], hint_8c[1], '-', alpha=0.5, color=box._color)
box = plt.plot(D4[0], D4[1], '*', label='D4')[0]
plt.plot(D4[0], D4[1], '-', alpha=0.5, color=box._color)
box = plt.plot(E8[0], E8[1], 'x', label='E8')[0]
plt.plot(E8[0], E8[1], '-', alpha=0.5, color=box._color)
plt.plot(2.0, 0.0915, 'yD', label='E8 Padded ($2^{16}$ entries)')
plt.legend()
plt.title('Lowest MSE Achievable for a Multivariate Gaussian')
plt.ylabel('MSE')
plt.yscale('log')
plt.xlabel('Bits')
plt.tight_layout()
plt.savefig('lattice_err.png', dpi=600)
